import os
import torch
import torch.distributed as dist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import warnings

from math import log, pi
from ase import Atoms
from ase.io import write
from ase.io.pov import set_high_bondorder_pairs
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdForceFieldHelpers
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import _LRScheduler


def apply_random_rotation(pc, rot_axis=1):
    B = pc.shape[0]

    theta = np.random.rand(B) * 2 * np.pi
    zeros = np.zeros(B)
    ones = np.ones(B)
    cos = np.cos(theta)
    sin = np.sin(theta)

    if rot_axis == 0:
        rot = np.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 1:
        rot = np.stack([
            cos, zeros, -sin,
            zeros, ones, zeros,
            sin, zeros, cos
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 2:
        rot = np.stack([
            ones, zeros, zeros,
            zeros, cos, -sin,
            zeros, sin, cos
        ]).T.reshape(B, 3, 3)
    else:
        raise Exception("Invalid rotation axis")
    rot = torch.from_numpy(rot).to(pc)

    # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
    pc_rotated = torch.bmm(pc, rot)
    return pc_rotated, rot, theta


def visualize_molecules(pts, gtr, mol, is_bond, save_file, virtual_node, pert_order=[0, 1, 2]):
    is_bond = is_bond.cpu().numpy()
    if virtual_node:
        pts = pts.cpu().detach().numpy()[:-1, pert_order].tolist()
        gtr = gtr.cpu().detach().numpy()[:-1, pert_order].tolist()
        numatoms = len(gtr)
        is_bond = is_bond[:numatoms, :numatoms]
    else:
        pts = pts.cpu().detach().numpy()[:, pert_order].tolist()
        gtr = gtr.cpu().detach().numpy()[:, pert_order].tolist()

    at = [x.GetSymbol() for x in mol.GetAtoms()]
    gtrmol = Atoms(at, positions=gtr)
    ptsmol = Atoms(at, positions=pts)

    # bondpairs = get_bondpairs(gtrmol, radius=1.1)
    is_bond = np.tril(is_bond)
    bondpairs = list(zip(*np.where(is_bond > 0), np.tile(np.array([[0, 0, 0]]), (np.sum(is_bond > 0), 1))))
    high_bondorder_pairs = {(i,j): ((0, 0, 0), 2, (0.17, 0.17, 0)) for i, j in zip(*np.where(is_bond > 1))}
    bondpairs = set_high_bondorder_pairs(bondpairs, high_bondorder_pairs)

    comlist = [ptsmol, gtrmol]
    sbplt = [121, 122]
    fig = plt.figure(figsize=(6, 3))
    for idx, com in enumerate(comlist):
        write(os.path.join('tmp', '%s.pov' % (save_file)), com, format='pov', run_povray=True, canvas_width=200,
              bondatoms=bondpairs, rotation="90y", radii=0.4)
        ax = fig.add_subplot(sbplt[idx])
        plt.axis('off')
        ax.imshow(mpimg.imread('%s.png' % (save_file)))

    fig.canvas.draw()
    res = np.array(fig.canvas.renderer._renderer)

    plt.close()
    return res


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0., std=1., trunc_std=2.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z, mean=0., std=1.):
    log_z = -0.5 * log(2 * pi * std**2)
    z = z - mean
    return log_z - z.pow(2) / (2 * std**2)


def getRMS(prb_mol, ref_pos, useFF=False, onlyHeavy=False):

    def optimizeWithFF(mol):

        # molf = Chem.AddHs(mol, addCoords=True)
        molf = deepcopy(mol)
        rdForceFieldHelpers.MMFFOptimizeMolecule(molf)
        # molf = Chem.RemoveHs(molf)

        return molf

    n_est = prb_mol.GetNumAtoms()

    ref_cf = Chem.rdchem.Conformer(n_est)
    for k in range(n_est):
        ref_cf.SetAtomPosition(k, ref_pos[k].tolist())

    ref_mol = deepcopy(prb_mol)
    ref_mol.RemoveConformer(0)
    ref_mol.AddConformer(ref_cf)

    if useFF:
        try:
            if onlyHeavy:
                res = rdMolAlign.AlignMol(Chem.RemoveAllHs(prb_mol), Chem.RemoveAllHs(optimizeWithFF(ref_mol)))
            else:
                res = rdMolAlign.AlignMol(prb_mol, optimizeWithFF(ref_mol))
        except:
            if onlyHeavy:
                res = rdMolAlign.AlignMol(Chem.RemoveAllHs(prb_mol), Chem.RemoveAllHs(ref_mol))
            else:
                res = rdMolAlign.AlignMol(prb_mol, ref_mol)
    else:
        if onlyHeavy:
            res = rdMolAlign.AlignMol(Chem.RemoveAllHs(prb_mol), Chem.RemoveAllHs(ref_mol))
        else:
            res = rdMolAlign.AlignMol(prb_mol, ref_mol)

    return res


# (TODO:) Yet to figure out refining using proximity graphs
def ComputeRMS(all_idx, all_sample, mollist, all_lens, num_sample_per_mol, mask, useFF=False, onlyHeavy=False):
    valres = []
    if mask:
        all_mask = all_lens.bool()

        for i, idx in enumerate(all_idx):
            mask = all_mask[i, :, 0]
            # order = all_order[i][:mask.sum()]
            sample = all_sample[i][:][:, mask, :]
            for j in range(num_sample_per_mol):
                valres.append(getRMS(deepcopy(mollist[idx]), sample[j], useFF=useFF, onlyHeavy=onlyHeavy))
    else:
        all_sample = torch.split(all_sample, all_lens, dim=1)
        # all_order = torch.split(all_order, all_lens)

        for i, idx in enumerate(all_idx):
            # order = all_order[i]
            sample = all_sample[i]
            for j in range(num_sample_per_mol):
                valres.append(getRMS(deepcopy(mollist[idx]), sample[j], useFF=useFF, onlyHeavy=onlyHeavy))

    valres = np.array(valres).reshape(-1, num_sample_per_mol)

    return np.mean(np.mean(valres, axis=1)), np.mean(np.std(valres, axis=1)), np.mean(np.min(valres, axis=1)), \
           np.median(np.mean(valres, axis=1)), np.median(np.std(valres, axis=1)), np.median(np.min(valres, axis=1))


def set_rdmol_positions(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def split_dataset_by_smiles(dset, mollist, smilist=None):
    split = {}

    for i, data in enumerate(tqdm(iter(dset))):
        if smilist is not None:
            smiles = smilist[data.idx]
        else:
            smiles = Chem.MolToSmiles(mollist[data.idx])
        if smiles in split:
            split[smiles].append(i)
        else:
            split[smiles] = [i]

    split = {k:torch.utils.data.Subset(dset, v) for k, v in split.items()}
    return split


### Adapted from pytorch. Slightly different version from that presented in pytorch
class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
