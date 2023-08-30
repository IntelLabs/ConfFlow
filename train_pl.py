import os
import random
import numpy as np
import pickle
import imageio
from typing import Optional, Union, Dict
from configargparse import Namespace
from omegaconf import DictConfig
from tqdm.auto import tqdm
from copy import deepcopy
import torch
from torch import optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch_geometric.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from models.flow import get_point_cnf, create_regularization_fns
from models.misc import BondEncoder, AtomEncoder
from datasets import get_mol_graph_data, VirtualDataset
from utils import truncated_normal, ComputeRMS, standard_normal_logprob, CosineAnnealingWarmRestarts, \
    apply_random_rotation, visualize_molecules, set_rdmol_positions
from args import get_args


class PointFlow_GCond(pl.LightningModule):
    def __init__(self, args: Optional[Union[Dict, Namespace, DictConfig]] = None):
        super().__init__()
        self.save_hyperparameters(args)
        global data
        self.data = data
        self.save_dir = os.path.join("checkpoints", self.hparams.log_name)
        self.temp_file = self.hparams.log_name.split('/')[-1]

        regularization_fns, self.regularization_coeffs = create_regularization_fns(self.hparams)
        self.point_cnf = get_point_cnf(self.hparams, regularization_fns, conditional=False)

        if self.hparams.new_features:
            edge_dims = data[0].edge_dims
            self.edge_proc = BondEncoder(128, edge_dims)
            atom_dims = data[0].atom_dims
            self.node_proc = AtomEncoder(128, atom_dims)
        else:
            self.edge_proc = nn.Sequential(
                nn.Linear(self.hparams.edge_dim + int(self.hparams.virtual_node), 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 128)
            )
            self.node_proc = nn.Sequential(
                nn.Linear(self.hparams.node_dim, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 128)
            )
        self.global_proc = nn.Sequential(
            nn.Linear(self.hparams.global_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 128)
        )
        self.pe_proc = nn.Linear(int(self.hparams.max_atoms * 2 / 3), 128, bias=True)

    def prepare_data(self):
        img_dir = os.path.join(self.save_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.tr_dataset, self.te_dataset = self.data[0], self.data[1]
            if self.hparams.train_subset:
                subset_ratio = 0.1
                self.tr_dataset = self.tr_dataset[:int(subset_ratio * len(self.tr_dataset))]
            if not self.global_rank:
                np.save(os.path.join(self.save_dir, "train_set_mean.npy"), self.tr_dataset.all_points_mean)
                np.save(os.path.join(self.save_dir, "train_set_std.npy"), self.tr_dataset.all_points_std)
                np.savez(os.path.join(self.save_dir, "feat_mean_std.npz"), self.tr_dataset.nodes_mean,
                         self.tr_dataset.edges_mean, self.tr_dataset.glbl_mean, self.tr_dataset.nodes_std,
                         self.tr_dataset.edges_std, self.tr_dataset.glbl_std)
        else:
            self.te_dataset = self.data[1]

    def train_dataloader(self):
        loader = DataLoader(dataset=self.tr_dataset, batch_size=self.hparams.batch_size,
                            shuffle=True, num_workers=self.hparams.num_workers, pin_memory=False,
                            drop_last=True, follow_batch=['edge_index'])
        return loader

    def val_dataloader(self):
        loader = DataLoader(dataset=self.te_dataset, batch_size=self.hparams.batch_size,
                            shuffle=False, num_workers=self.hparams.num_workers, pin_memory=False,
                            drop_last=False, follow_batch=['edge_index'])
        return loader

    def test_dataloader(self):
        loader = DataLoader(dataset=self.te_dataset, batch_size=self.hparams.batch_size,
                            shuffle=False, num_workers=self.hparams.num_workers, pin_memory=False,
                            drop_last=False, follow_batch=['edge_index'])
        return loader

    @staticmethod
    def sample_gaussian(size, mean=0., std=1., truncate_std=None, device=None):
        y = torch.randn(*size, device=device).float()
        y.data.mul_(std).add_(mean)
        if truncate_std is not None:
            truncated_normal(y, mean=mean, std=std, trunc_std=truncate_std)
        return y

    @staticmethod
    def sample_gaussian_subspace(size, batch, mean=0., std=1., truncate_std=None, device=None):
        y = torch.randn(*size, device=device).float()
        y.data.mul_(std).add_(mean)
        centroid = scatter_mean(y, batch, dim=0)
        y.data.sub_(centroid[batch])
        if truncate_std is not None:
            truncated_normal(y, mean=mean, std=std, trunc_std=truncate_std)
        return y

    def forward(self, graphs, sampling=False, truncate_std=None):
        context = None
        if "mpnn" in self.hparams.layer_type_point:
            context = (self.node_proc(graphs.x) + self.pe_proc(graphs.pe), self.edge_proc(graphs.edge_attr),
                       self.global_proc(graphs.y), graphs.edge_index, graphs.batch)  # , graphs.rad)
        if sampling:
            # Sample points conditioned on the shape code
            if self.hparams.masking:
                batch_size = graphs.num_graphs
                y = self.sample_gaussian((batch_size, self.hparams.max_atoms, self.hparams.input_dim), std=self.hparams.sigma,
                                         truncate_std=truncate_std, device=self.device)
            else:
                y = self.sample_gaussian((len(graphs.batch), self.hparams.input_dim), std=self.hparams.sigma,
                                         truncate_std=truncate_std, device=self.device)
            x = self.point_cnf(y, context, reverse=True).view(*y.size())

            return x
        else:
            reg_loss_x = torch.FloatTensor([0.0])
            reg_loss_xy = []
            if self.hparams.masking:
                # Collecting position vectors into single vector
                graph_list = graphs.to_data_list()
                x = []
                mask = []
                for graph in graph_list:
                    x.append(graph.gtpos)
                    mask.append(graph.mask)
                x = torch.stack(x, dim=0)
                mask = torch.stack(mask, dim=0)
                num_points = x.size(1)
                total_atoms = mask.sum()
                batch_size = graphs.num_graphs

                y, delta_log_py, y_reg_states = self.point_cnf(x, context, torch.zeros(batch_size, num_points, 1).type_as(x))
                log_py = standard_normal_logprob(y, std=self.hparams.sigma).view(batch_size, num_points, -1).sum(2, keepdim=True)
                log_px = ((log_py - delta_log_py).view(batch_size, num_points, 1) * mask).sum(1)
                if self.regularization_coeffs:
                    reg_loss_xy = [
                        (reg_state.view(batch_size, num_points, 1) * mask).sum(1) * coeff for reg_state, coeff in
                        zip(y_reg_states, self.regularization_coeffs) if coeff != 0]
                    reg_loss_x = sum(reg_loss_xy)
                    assert log_px.size(0) == reg_loss_x.size(0), 'sizes do not match'
            else:
                total_atoms = len(graphs.batch)

                y, delta_log_py, y_reg_states = self.point_cnf(graphs.gtpos, context,
                                                               torch.zeros(total_atoms, 1).type_as(graphs.gtpos))
                log_py = standard_normal_logprob(y, std=self.hparams.sigma).sum(1, keepdim=True)
                log_px = log_py - delta_log_py
                if self.regularization_coeffs:
                    reg_loss_xy = [
                        reg_state.view(total_atoms, 1) * coeff for reg_state, coeff in
                        zip(y_reg_states, self.regularization_coeffs) if coeff >= 0.]
                    reg_loss_x = sum(reg_loss_xy)
                    assert log_px.size(0) == reg_loss_x.size(0), 'sizes do not match'

            # Loss
            log_px = -log_px.sum() / total_atoms / self.hparams.input_dim
            reg_loss_total = reg_loss_x.sum() / total_atoms / self.hparams.input_dim
            loss = (log_px + reg_loss_total) * self.hparams.recon_weight
            reg_loss_indiv = [x.sum() / total_atoms / self.hparams.input_dim for x in reg_loss_xy]

            return loss, log_px, reg_loss_total, reg_loss_indiv

    def setup_optimization(self):
        if self.hparams.optimizer == 'adam':
            self._optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adamW':
            self._optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            self._optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum,
                                              weight_decay=self.hparams.weight_decay)
        else:
            assert 0, "args.optimizer should be either 'adam' or 'sgd' or 'adamw'"

        return self._optimizer

    def setup_scheduler(self):
        # initialize the learning rate scheduler
        if self.hparams.scheduler == 'exponential':
            self._scheduler = {
                "scheduler": optim.lr_scheduler.ExponentialLR(self._optimizer, self.hparams.lr_decay),
                "frequency": self.hparams.exp_decay_freq,
            }

        elif self.hparams.scheduler == 'step':
            self._scheduler = {
                "scheduler": optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[31], gamma=self.hparams.lr_decay),
                "frequency": self.hparams.exp_decay_freq,
            }

        elif self.hparams.scheduler == 'linear':
            def lambda_rule(ep):
                lr_l = 1.0 - max(0, ep - 0.5 * self.hparams.epochs) / float(0.5 * self.hparams.epochs)
                return lr_l
            self._scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda_rule),
                "frequency": self.hparams.exp_decay_freq,
            }

        elif self.hparams.scheduler == 'dynamic':
            if self.hparams.val_sampling:
                monitor = "val_sample_test/RMS"
            else:
                monitor = "val/loss"
            self._scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=self.hparams.lr_decay, threshold=0.01),
                "monitor": monitor,
                "frequency": self.hparams.exp_decay_freq,
            }

        elif self.hparams.scheduler == 'cosine':
            self._scheduler = {
                "scheduler": CosineAnnealingWarmRestarts(self._optimizer, 5, 2, eta_min=1e-5),
                "interval": "step",
            }

        elif self.hparams.scheduler == 'warm':
            def lambda_rule(itr):
                lr_l = min(min(float(itr + 1) / max(self.hparams.warmup_iters, 1), 1.0),
                           self.hparams.lr_decay ** int((itr + 1) / 25000))
                return lr_l
            self._scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda_rule),
                "interval": "step",
            }
        else:
            assert 0, "args.schedulers should be either 'exponential' or 'step' or 'linear' or 'dynamic'"

        return self._scheduler

    def configure_optimizers(self):
        self.setup_optimization()
        self.setup_scheduler()

        return [self._optimizer], [self._scheduler]

    def on_before_batch_transfer(self, data, dataloader_idx):
        if self.trainer.training and self.hparams.random_rotate:
            train_loader = self.train_dataloader()
            if self.hparams.masking:
                tr_batch = data.gtpos.view(data.num_graphs, -1, self.hparams.input_dim)
                tr_batch, rot, _ = apply_random_rotation(tr_batch, rot_axis=train_loader.dataset.gravity_axis)
                data.gtpos = tr_batch.view(-1, self.hparams.input_dim)
                tr_batch = data.pos.view(data.num_graphs, -1, self.hparams.input_dim)
                data.pos = torch.bmm(tr_batch, rot).view(-1, self.hparams.input_dim)
            else:
                lengths = torch.bincount(data.batch).tolist()
                tr_batch = pad_sequence(torch.split(data.gtpos, lengths), batch_first=True)
                tr_batch = tr_batch.view(data.num_graphs, -1, self.hparams.input_dim)

                tr_batch, rot, _ = apply_random_rotation(tr_batch, rot_axis=train_loader.dataset.gravity_axis)

                tr_batch = [seq[:l] for l, seq in zip(lengths, tr_batch)]
                data.gtpos = torch.cat(tr_batch, dim=0)

                tr_batch = pad_sequence(torch.split(data.pos, lengths), batch_first=True)
                tr_batch = tr_batch.view(data.num_graphs, -1, self.hparams.input_dim)

                tr_batch = torch.bmm(tr_batch, rot)
                tr_batch = [seq[:l] for l, seq in zip(lengths, tr_batch)]
                data.pos = torch.cat(tr_batch, dim=0)
        return data

    def training_step(self, batch, batch_idx):
        loss, log_px, reg_loss, reg_loss_indiv = self(batch)
        self.log_dict({'train/loss': loss,
                       'train/recon': log_px,
                       'train/regloss': reg_loss}, on_epoch=True, on_step=False)
        self.log_dict({'train/regloss_'+str(i): x for i, x in enumerate(reg_loss_indiv)}, on_epoch=True, on_step=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if self.global_rank == 0 and batch_idx == 0 and ((self.current_epoch + 1) % self.hparams.viz_freq == 0):
            self.trainer.model.eval()
            torch.set_grad_enabled(False)
            loader = self.train_dataloader()
            mollist = loader.dataset.mollist

            num_samples = min(10, batch.num_graphs)
            inputs_few = batch.to_data_list()
            random.shuffle(inputs_few)
            batch_input = Batch.from_data_list(inputs_few[:num_samples], follow_batch=['edge_index'])

            # samples
            samples = self(batch_input, sampling=True)
            if self.hparams.masking:
                results = []
                for idx in range(num_samples):
                    mask = inputs_few[idx].mask.bool()[:, 0]
                    pos = inputs_few[idx].gtpos[mask, :]
                    j = inputs_few[idx].idx
                    is_edge = inputs_few[idx].is_edge
                    res = visualize_molecules(samples[idx][mask, :], pos, mollist[j], is_edge, self.temp_file,
                                              self.hparams.virtual_node, pert_order=loader.dataset.display_axis_order)
                    results.append(res)
            else:
                samples = torch.split(samples, torch.bincount(batch_input.batch).tolist(), dim=0)
                results = []
                for idx in range(num_samples):
                    pos = inputs_few[idx].gtpos
                    j = inputs_few[idx].idx
                    is_edge = inputs_few[idx].is_edge
                    res = visualize_molecules(samples[idx], pos, mollist[j], is_edge, self.temp_file,
                                              self.hparams.virtual_node, pert_order=loader.dataset.display_axis_order)
                    results.append(res)

            res = np.concatenate(results, axis=0)
            imageio.imwrite(os.path.join(self.save_dir, 'images', 'tr_vis_sampled_epoch%d.png' % self.current_epoch),res)

            self.trainer.model.train()
            torch.set_grad_enabled(True)
        return

    def validation_step(self, batch, batch_idx):
        if self.hparams.val_sampling:
            if self.hparams.masking:
                out_pc = []
                for _ in range(self.hparams.val_sample_mol):
                    out = self(batch, sampling=True)
                    out_pc.append(out.unsqueeze(1))
                out_pc = torch.cat(out_pc, dim=1)

                # denormalize
                num_graphs = batch.num_graphs
                m, s = batch.mean.float(), batch.std.float()
                m, s = m.view(num_graphs, 1, 1, -1), s.view(num_graphs, 1, 1, -1)
                out_pc = out_pc * s + m

                return {"idx": batch.idx, "misc": batch.mask.view(num_graphs, self.hparams.max_atoms, -1), "out_pc": out_pc}
            else:
                out_pc = []
                for _ in range(self.hparams.val_sample_mol):
                    out = self(batch, sampling=True)
                    out_pc.append(out.unsqueeze(0))
                out_pc = torch.cat(out_pc, dim=0)

                # denormalize
                m, s = batch.mean.float(), batch.std.float()
                node_batch = batch.batch
                m, s = m[node_batch].view(1, len(node_batch), -1), s[node_batch].view(1, len(node_batch), -1)
                out_pc = out_pc * s + m

                num_graphs = batch.num_graphs
                atoms_per_mol = torch.bincount(node_batch, minlength=num_graphs).tolist()

                return {"idx": batch.idx, "misc": atoms_per_mol, "out_pc": out_pc}
        else:
            loss, log_px, reg_loss, reg_loss_indiv = self(batch, sampling=False)
            return {"val_loss": loss, "val_recon": log_px, "val_regloss": reg_loss,
                    "val_regloss0": reg_loss_indiv[0], "val_regloss1": reg_loss_indiv[1]}

    def validation_epoch_end(self, val_outputs):
        if self.hparams.val_sampling:
            all_sample = []
            all_idx = []
            all_lens = []
            mollist = self.val_dataloader().dataset.mollist
            if self.hparams.masking:
                for out in val_outputs:
                    all_sample.append(out["out_pc"])
                    all_idx.extend(out["idx"])
                    all_lens.append(out["misc"])
                sample_pcs = torch.cat(all_sample, dim=0)
                all_lens = torch.cat(all_lens, dim=0)
            else:
                for out in val_outputs:
                    all_sample.append(out["out_pc"])
                    all_idx.extend(out["idx"])
                    all_lens.extend(out["misc"])
                sample_pcs = torch.cat(all_sample, dim=1)

            rms, rms_std, rms_best, median_rms, median_rms_std, median_rms_best = \
                ComputeRMS(all_idx, sample_pcs, mollist, all_lens, self.hparams.val_sample_mol,
                           self.hparams.masking, self.hparams.mmff, self.hparams.onlyHeavy)
            self.log("val_sample_test/RMS", rms, prog_bar=True)
            self.log_dict({"val_sample_test/RMS Std": rms_std,
                           "val_sample_test/RMS Best": rms_best,
                           "val_sample_test/Median RMS": median_rms,
                           "val_sample_test/Median RMS Std": median_rms_std,
                           "val_sample_test/Median RMS Best": median_rms_best
                           })
        else:
            loss = []
            log_px = []
            regloss = []
            regloss0 = []
            regloss1 = []
            for out in val_outputs:
                loss.append(out["val_loss"])
                log_px.append(out["val_recon"])
                regloss.append(out["val_regloss"])
                regloss0.append(out["val_regloss0"])
                regloss1.append(out["val_regloss1"])
            loss = torch.stack(loss, dim=0)
            log_px = torch.stack(log_px, dim=0)
            regloss = torch.stack(regloss, dim=0)
            regloss0 = torch.stack(regloss0, dim=0)
            regloss1 = torch.stack(regloss1, dim=0)
            self.log("val/loss", loss.mean(), prog_bar=True)
            self.log_dict({'val/recon': log_px.mean(), 'val/regloss': regloss.mean(), 'val/regloss_0': regloss0.mean(),
                           'val/regloss_1': regloss1.mean()})

    def test_step(self, batch, batch_idx):
        out_pc = []
        for _ in range(self.hparams.val_sample_mol):
            out = self(batch, sampling=True)
            out_pc.append(out.unsqueeze(0))
        out_pc = torch.cat(out_pc, dim=0)

        m, s = batch.mean.float(), batch.std.float()
        if self.hparams.masking:
            num_graphs = batch.num_graphs
            m, s = m.view(1, num_graphs, 1, -1), s.view(1, num_graphs, 1, -1)
        else:
            node_batch = batch.batch
            m, s = m[node_batch].view(1, len(node_batch), -1), s[node_batch].view(1, len(node_batch), -1)
        out_pc = out_pc * s + m

        batch.to('cpu')
        out_pc.to('cpu')
        all_data_list = []
        for i in range(out_pc.shape[0]):
            batch.pos = out_pc[i].view(-1, self.hparams.input_dim)
            batch_list = batch.to_data_list()
            all_data_list += batch_list

        return all_data_list

    def test_epoch_end(self, outputs):
        gen_rdmols = []
        opt_rdmols = []
        mollist = self.test_dataloader().dataset.mollist
        for datalist in outputs:
            for data in datalist:
                rdmol = mollist[data.idx]
                pos = data.pos[data.mask.bool()[:, 0], :] if self.hparams.masking else data.pos
                rdmol = set_rdmol_positions(rdmol, pos)
                gen_rdmols.append(rdmol)
                opt_rdmols.append(mollist[data.idx])

        # Optimize using MMFF
        if self.hparams.mmff:
            opt_rdmols = []
            for mol in tqdm(gen_rdmols, desc='MMFF Optimize'):
                opt_mol = deepcopy(mol)
                MMFFOptimizeMolecule(opt_mol)
                opt_rdmols.append(opt_mol)

        # Save
        save_file = os.path.join(self.save_dir, 'generated.pkl')
        print('Saving to: %s' % save_file)
        with open(save_file, 'wb') as f:
            pickle.dump([gen_rdmols, opt_rdmols], f)

    def on_fit_end(self):
        try:
            os.remove('%s.png' % self.temp_file)
            os.remove('tmp/%s.ini' % self.temp_file)
            os.remove('tmp/%s.pov' % self.temp_file)
        except:
            print("File does not exist")

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


def main():
    args = get_args()
    save_dir = os.path.join("checkpoints", args.log_name)
    log_dir = os.path.join("runs", args.log_name)

    if args.seed is None:
        args.seed = random.randint(0, 2**32)

    pl.seed_everything(args.seed, workers=True)

    ######## Checkpoint ##########
    model_checkpoint = ModelCheckpoint(filename='checkpoint-{epoch:d}',
                                       save_top_k=-1,
                                       save_last=True,
                                       auto_insert_metric_name=False,
                                       save_on_train_epoch_end=False,
                                       every_n_epochs=args.save_freq)
    model_checkpoint.FILE_EXTENSION = ".pt"
    model_checkpoint.CHECKPOINT_NAME_LAST = "checkpoint-latest"

    if args.val_sampling:
        monitor="val_sample_test/RMS"
    else:
        monitor="val/loss"
    best_checkpoint = ModelCheckpoint(filename='checkpoint-best',
                                      monitor=monitor,
                                      save_last=False)
    best_checkpoint.FILE_EXTENSION = ".pt"

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [model_checkpoint, best_checkpoint, lr_monitor]

    ######### Logging ##########
    logger = TensorBoardLogger(save_dir=log_dir,
                               name=None,
                               version="")

    ######### Load Data ##########
    global data
    split = ['train', args.valsplit]
    data = get_mol_graph_data(args, split)
    if args.val_sample_mol < 0:
        args.val_sample_mol *= -1
        data = list(data)
        data[1] = VirtualDataset(data[1])
        data = tuple(data)

    ######### Model loading ##########
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoints', args.resume_checkpoint)

    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoints', 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoints', 'checkpoint-latest.pt')  # use the latest checkpoint

    if args.resume_checkpoint is not None and not args.resume_optimizer:
        model = PointFlow_GCond.load_from_checkpoint(args.resume_checkpoint, hparams_file=os.path.join(log_dir, 'hparams.yaml'))
        args.resume_checkpoint = None
    else:
        model = PointFlow_GCond(args)

    ######### Define trainer ##########
    if args.generate:
        acc = None
    else:
        acc = "ddp"
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         gradient_clip_val=args.clip_grad,
                         gradient_clip_algorithm="norm",
                         gpus=args.gpu,
                         auto_select_gpus=True,
                         max_epochs=args.epochs,
                         val_check_interval=args.val_freq,
                         log_every_n_steps=10,
                         accelerator=acc,
                         weights_save_path=save_dir,
                         resume_from_checkpoint=args.resume_checkpoint,
                         deterministic=True,
                         terminate_on_nan=True,
                         num_sanity_val_steps=0,
                         )

    # Fit model
    trainer.fit(model)

    # # Validate model on best model
    # trainer.validate(model)

    if args.generate:
        # (TODO): Currently it is limited to execution on single gpu run.
        # Generates samples for final evaluation on TEST set
        trainer.test(model)


if __name__ == '__main__':
    main()
