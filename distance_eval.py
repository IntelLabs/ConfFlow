import numpy as np
import torch
import os
import pickle
import copy
import argparse
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def split_dataset_by_smiles(dataset):
    split = {}

    if isinstance(dataset, MoleculeDataset):
        dset = dataset.dataset
    else:
        dset = dataset
    for i, data in enumerate(tqdm(dset)):
        smiles = data.smiles
        if smiles in split:
            split[smiles].append(i)
        else:
            split[smiles] = [i]

    split = {k:torch.utils.data.Subset(dataset, v) for k, v in split.items()}
    return split


def evaluate_distance(pos_ref, pos_gen, edge_index, atom_type, ignore_H=True):
    # compute generated length and ref length
    ref_lengths = (pos_ref[:, edge_index[0]] - pos_ref[:, edge_index[1]]).norm(dim=-1)  # (N, num_edge)
    gen_lengths = (pos_gen[:, edge_index[0]] - pos_gen[:, edge_index[1]]).norm(dim=-1)  # (M, num_edge)

    stats_single = []
    first = 1
    for i, (row, col) in enumerate(tqdm(edge_index.t())):
        if row >= col:
            continue
        if ignore_H and 1 in (atom_type[row].item(), atom_type[col].item()):
            continue
        gen_l = gen_lengths[:, i]
        ref_l = ref_lengths[:, i]
        if first:
            first = 0
        mmd = compute_mmd(gen_l.view(-1, 1).cuda(), ref_l.view(-1, 1).cuda()).item()
        stats_single.append({
            'edge_id': i,
            'nodes': (row.item(), col.item()),
            'gen_lengths': gen_l.cpu(),
            'ref_lengths': ref_l.cpu(),
            'mmd': mmd
        })

    first = 1
    stats_pair = []
    for i, (row_i, col_i) in enumerate(tqdm(edge_index.t())):
        if row_i >= col_i:
            continue
        if ignore_H and 1 in (atom_type[row_i].item(), atom_type[col_i].item()):
            continue
        for j, (row_j, col_j) in enumerate(edge_index.t()):
            if (row_i >= row_j) or (row_j >= col_j):
                continue
            if ignore_H and 1 in (atom_type[row_j].item(), atom_type[col_j].item()):
                continue

            gen_L = gen_lengths[:, (i, j)]  # (N, 2)
            ref_L = ref_lengths[:, (i, j)]  # (M, 2)
            if first:
                first = 0
            mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

            stats_pair.append({
                'edge_id': (i, j),
                'nodes': (
                    (row_i.item(), col_i.item()),
                    (row_j.item(), col_j.item()),
                ),
                'gen_lengths': gen_L.cpu(),
                'ref_lengths': ref_L.cpu(),
                'mmd': mmd
            })

    edge_filter = edge_index[0] < edge_index[1]
    if ignore_H:
        for i, (row, col) in enumerate(edge_index.t()):
            if 1 in (atom_type[row].item(), atom_type[col].item()):
                edge_filter[i] = False

    gen_L = gen_lengths[:, edge_filter]  # (N, Ef)
    ref_L = ref_lengths[:, edge_filter]  # (M, Ef)
    mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

    stats_all = {
        'gen_lengths': gen_L.cpu(),
        'ref_lengths': ref_L.cpu(),
        'mmd': mmd
    }
    return stats_single, stats_pair, stats_all


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.shape[0]) + int(target.shape[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    total1 = total.unsqueeze(1).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))

    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def compute_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        Params:
            source: (N, D)
            target: (M, D)
        Return:
            loss: MMD loss
    '''
    batch_size = int(source.shape[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)

    return loss


def _evaluate_distance(data, ignore_H):
    return evaluate_distance(data[0], data[1], data[2], data[3], ignore_H=ignore_H)


class DistEvaluator(object):

    def __init__(self, ignore_H=False, device='cuda'):
        super().__init__()
        self.device = device
        self.func = partial(_evaluate_distance, ignore_H=ignore_H)

    def __call__(self, ref_dset, gen_dset):
        ref_grouped = split_dataset_by_smiles(ref_dset)
        gen_grouped = split_dataset_by_smiles(gen_dset)

        pos_refs = []
        pos_gens = []
        edge_indexs = []
        atom_types = []

        for smiles, gen_mols in gen_grouped.items():
            if smiles not in ref_grouped:
                continue
            edge_indexs.append(gen_mols[0].edge_index)
            atom_types.append(gen_mols[0].node_type)
            ref_mols = ref_grouped[smiles]

            p_ref = []
            p_gen = []
            for mol in ref_mols:
                p_ref.append(mol.pos.reshape(1, -1, 3).to(self.device))
            for mol in gen_mols:
                p_gen.append(mol.pos.reshape(1, -1, 3).to(self.device))
            pos_refs.append(torch.cat(p_ref, dim=0))
            pos_gens.append(torch.cat(p_gen, dim=0))

        return self._run(pos_refs, pos_gens, edge_indexs, atom_types)

    def _run(self, pos_refs, pos_gens, edge_indexs, atom_types):
        """
        Args:
            pos_refs:  A list of numpy tensors of shape (num_refs, num_atoms, 3)
            pos_gens:  A list of numpy tensors of shape (num_gens, num_atoms, 3)
            edge_indexs:  A list of LongTensor(E, 2)
            atom_types:   A list of LongTensor(N, )
        """
        s_mmd_all = []
        p_mmd_all = []
        a_mmd_all = []

        for data in tqdm(zip(pos_refs, pos_gens, edge_indexs, atom_types), total=len(pos_refs)):
            stats_single, stats_pair, stats_all = self.func(data)
            s_mmd_all += [e['mmd'] for e in stats_single]
            p_mmd_all += [e['mmd'] for e in stats_pair]
            a_mmd_all.append(stats_all['mmd'])

        return s_mmd_all, p_mmd_all, a_mmd_all


def rdmol_to_data(mol:Mol):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float)

    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

    edge_index = torch.tensor([row, col], dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]

    smiles = Chem.MolToSmiles(mol)

    data = Data(node_type=z, pos=pos, edge_index=edge_index, rdmol=copy.deepcopy(mol), smiles=smiles)

    return data


def enumerate_conformers(mol):
    num_confs = mol.GetNumConformers()
    if num_confs == 1:
        yield mol
        return
    mol_templ = copy.deepcopy(mol)
    mol_templ.RemoveAllConformers()
    for conf_id in tqdm(range(num_confs), desc='Conformer'):
        conf = mol.GetConformer(conf_id)
        conf.SetId(0)
        mol_conf = copy.deepcopy(mol_templ)
        conf_id = mol_conf.AddConformer(conf)
        yield mol_conf


class MoleculeDataset(Dataset):

    def __init__(self, raw_path, force_reload=False, transform=None):
        super().__init__()
        self.raw_path = raw_path
        self.processed_path = raw_path + '.processed'
        self.transform = transform

        _, extname = os.path.splitext(raw_path)
        assert extname in ('.sdf', '.pkl'), 'Only supports .sdf and .pkl files'

        self.dataset = None
        if force_reload or not os.path.exists(self.processed_path):
            if extname == '.sdf':
                self.process_sdf()
            elif extname == '.pkl':
                self.process_pickle()
        else:
            self.load_processed()

    def load_processed(self):
        self.dataset = torch.load(self.processed_path)

    def process_sdf(self):
        self.dataset = []
        suppl = Chem.SDMolSupplier(self.raw_path, removeHs=False, sanitize=True)
        for mol in tqdm(suppl):
            if mol is None:
                continue
            for conf in enumerate_conformers(mol):
                self.dataset.append(rdmol_to_data(conf))
        torch.save(self.dataset, self.processed_path)

    def process_pickle(self):
        self.dataset = []
        with open(self.raw_path, 'rb') as f:
            mols, _ = pickle.load(f)
            for mol in tqdm(mols):
                for conf in enumerate_conformers(mol):
                    self.dataset.append(rdmol_to_data(conf))
            torch.save(self.dataset, self.processed_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data


parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, default='./generated.pkl')
parser.add_argument('--testdata', type=str)
parser.add_argument('--onlyHeavy', action='store_true')
args = parser.parse_args()

ref_dset = MoleculeDataset(args.testdata)
gen_dset = MoleculeDataset(args.out)

# Dist
evaluator = DistEvaluator(ignore_H=args.onlyHeavy)
# Run evaluation
results = evaluator(ref_dset, gen_dset)
s_mmd_all = np.asarray(results[0])
p_mmd_all = np.asarray(results[1])
a_mmd_all = np.asarray(results[2])
print('single(Mean) %.6f, single(Median) %.6f' % (
        np.mean(s_mmd_all, axis=0),
        np.median(s_mmd_all, axis=0),
    ))
print('pair(Mean) %.6f, pair(Median) %.6f' % (
        np.mean(p_mmd_all, axis=0),
        np.median(p_mmd_all, axis=0),
    ))
print('all(Mean) %.6f, all(Median) %.6f' % (
        np.mean(a_mmd_all, axis=0),
        np.median(a_mmd_all, axis=0),
    ))
