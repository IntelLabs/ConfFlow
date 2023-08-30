import os
import random
import pickle
import torch
import numpy as np
from torch.utils import data
from torch_geometric.data import InMemoryDataset, Data
from numpy import linalg as LA
from scipy import sparse
from scipy.sparse import csgraph
from utils import split_dataset_by_smiles


class MolGraphSampling(InMemoryDataset):
    def __init__(self, root, rawdata, procdata, mask=True, transform=None, pre_transform=None,
                 val_sample_size=10000, test_sample_size=10000, split='train',
                 normalize_per_shape=False, normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 node_dim=22, edge_dim=30, max_atoms=9, standardise=False,
                 nodes_mean=None, edges_mean=None, glbl_mean=None,
                 nodes_std=None, edges_std=None, glbl_std=None):
        self.eps = 1e-8
        self.root = root
        self.rawdata = rawdata
        self.procdata = procdata
        self.mask = mask
        self.val_sample_size = val_sample_size
        self.test_sample_size = test_sample_size
        self.split = split
        self.standardise = standardise
        self.nodes_mean = nodes_mean
        self.nodes_std = nodes_std
        self.edges_mean = edges_mean
        self.edges_std = edges_std
        self.glbl_mean = glbl_mean
        self.glbl_std = glbl_std
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.all_points_mean = all_points_mean
        self.all_points_std = all_points_std
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_max = max_atoms
        self.k = int(self.n_max * 2 / 3)
        self.pos_dim = 3
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(MolGraphSampling, self).__init__(root, transform, pre_transform)

        arr = np.load(self.processed_paths[1], allow_pickle=True)
        self.mollist, self.shuffle_idx, self.all_points_mean, self.all_points_std, self.nodes_mean, self.nodes_std, \
        self.edges_mean, self.edges_std, self.glbl_mean, self.glbl_std, self.atom_dims, self.edge_dims, self.smilist = \
            arr['arr_0'], arr['arr_1'], arr['arr_2'], arr['arr_3'], arr['arr_4'], arr['arr_5'], \
            arr['arr_6'], arr['arr_7'], arr['arr_8'], arr['arr_9'], arr['arr_10'], arr['arr_11'], arr['arr_12']
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.standardise and self.nodes_mean is None:
            raise ValueError('Standardization not applied. Delete previously processed data.')

    @property
    def raw_file_names(self):
        return self.rawdata

    @property
    def processed_file_names(self):
        return ['%s%s_%d_%d.pt' % (self.procdata, self.split, int(self.standardise), int(self.mask)),
                '%s%s_%d_%d_extra.npz' % (self.procdata, self.split, int(self.standardise), int(self.mask))]

    def download(self):
        # Download to `self.raw_dir`.
        if not os.path.exists(self.raw_paths[0]):
            raise ValueError('Oops!!! Dataset {} does not exist'.format(self.raw_paths[0]))

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.pos_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return torch.from_numpy(m).float(), torch.from_numpy(s).float()

        return torch.from_numpy(self.all_points_mean.reshape(1, -1)).float(), \
               torch.from_numpy(self.all_points_std.reshape(1, -1)).float()

    @staticmethod
    def compute_mean_std(a):
        return(
            np.mean(a, axis=0),
            np.maximum(np.std(a, axis=0), 1E-6),
        )

    def process(self):
        print("Reading " + self.raw_paths[0])
        with open(self.raw_paths[0], 'rb') as f:
            [D1, D2, D3, D4, D5] = pickle.load(f)

        print("Reading " + self.raw_paths[1])
        with open(self.raw_paths[1], 'rb') as f:
            [mollist, smilist, atom_feat_dims, edge_feat_dims] = pickle.load(f)

        total_mol = len(D5)
        tr_sample_size = total_mol - self.test_sample_size - self.val_sample_size
        if self.split == 'train':
            D1 = D1[:tr_sample_size]
            D2 = D2[:tr_sample_size]
            D3 = D3[:tr_sample_size]
            D4 = D4[:tr_sample_size]
            D5 = D5[:tr_sample_size]
            mollist = mollist[:tr_sample_size]
            smilist = smilist[:tr_sample_size]
        elif self.split == 'val':
            end_idx = tr_sample_size + self.val_sample_size
            D1 = D1[tr_sample_size:end_idx]
            D2 = D2[tr_sample_size:end_idx]
            D3 = D3[tr_sample_size:end_idx]
            D4 = D4[tr_sample_size:end_idx]
            D5 = D5[tr_sample_size:end_idx]
            mollist = mollist[tr_sample_size:end_idx]
            smilist = smilist[tr_sample_size:end_idx]
        elif self.split == 'test':
            start_idx = tr_sample_size + self.val_sample_size
            D1 = D1[start_idx:]
            D2 = D2[start_idx:]
            D3 = D3[start_idx:]
            D4 = D4[start_idx:]
            D5 = D5[start_idx:]
            mollist = mollist[start_idx:]
            smilist = smilist[start_idx:]

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(D5)))
        random.Random(38383).shuffle(self.shuffle_idx)
        D1 = [D1[idx] for idx in self.shuffle_idx]
        D2 = [D2[idx] for idx in self.shuffle_idx]
        D3 = [D3[idx] for idx in self.shuffle_idx]
        D4 = [D4[idx] for idx in self.shuffle_idx]
        D5 = [D5[idx] for idx in self.shuffle_idx]
        mollist = mollist[self.shuffle_idx]
        smilist = smilist[self.shuffle_idx]

        self.mollist = mollist
        self.smilist = smilist
        self.atom_dims = atom_feat_dims
        self.edge_dims = edge_feat_dims

        if self.standardise:
            if self.nodes_mean is None:
                nodes = np.vstack(D1)
                edges = np.vstack(D2)

                nodes_mean, nodes_std = self.compute_mean_std(nodes)
                edges_mean, edges_std = self.compute_mean_std(edges)

                self.nodes_mean = nodes_mean.reshape(1, -1).astype('float32')
                self.nodes_std = nodes_std.reshape(1, -1).astype('float32')
                self.edges_mean = edges_mean.reshape(1, -1).astype('float32')
                self.edges_std = edges_std.reshape(1, -1).astype('float32')

        if self.glbl_mean is None:
            D4 = np.vstack(D4)
            global_mean, global_std = self.compute_mean_std(D4)
            self.glbl_mean = global_mean.reshape(1, -1)
            self.glbl_std = global_std.reshape(1, -1)
            D4 = (D4 - self.glbl_mean) / self.glbl_std

        # Normalization of position vectors
        pos = np.vstack(D5)
        if self.all_points_mean is not None and self.all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = self.all_points_mean
            self.all_points_std = self.all_points_std
        else:  # normalize across the dataset
            self.all_points_mean = pos.mean(axis=0).reshape(1, self.pos_dim)
            if self.normalize_std_per_axis:
                self.all_points_std = pos.std(axis=0).reshape(1, self.pos_dim)
            else:
                self.all_points_std = pos.reshape(-1).std(axis=0).reshape(1, 1)

        np.savez(self.processed_paths[1], self.mollist, self.shuffle_idx, self.all_points_mean, self.all_points_std,
                 self.nodes_mean, self.nodes_std, self.edges_mean, self.edges_std, self.glbl_mean, self.glbl_std,
                 self.atom_dims, self.edge_dims, self.smilist)

        data_list = []
        for idx in range(len(D5)):
            total_atoms = len(D1[idx])
            if self.standardise:
                nodes = torch.from_numpy((D1[idx] - self.nodes_mean) / self.nodes_std).float()
                edges = torch.from_numpy((D2[idx] - self.edges_mean) / self.edges_std).float()
            else:
                nodes = torch.from_numpy(D1[idx]).long()
                edges = torch.from_numpy(D2[idx]).long()
            edge_index = D3[idx].long()
            glbl = torch.from_numpy(D4[idx].reshape(1, -1)).float()
            pos = torch.from_numpy(D5[idx]).float()
            m, s = self.get_pc_stats(idx)
            gtpos = (pos - m) / (s + self.eps)

            temp = D3[idx].numpy()
            adj = sparse.coo_matrix((np.ones(temp.shape[-1]), (temp[0], temp[1])), shape=(self.n_max, self.n_max)).tocsr()
            _, v = LA.eigh(csgraph.laplacian(adj.todense(), normed=True))
            pe = torch.from_numpy(v[:total_atoms, :self.k]).float()

            data_list.append(Data(x=nodes, edge_index=edge_index, edge_attr=edges, gtpos=gtpos,
                                  y=glbl, pos=pos, mean=m, std=s, idx=idx, pe=pe))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class VirtualDataset(torch.utils.data.Dataset):

    def __init__(self, dset):
        super().__init__()
        self.mollist = dset.mollist
        grouped = split_dataset_by_smiles(dset, self.mollist)
        self.grouped = [subset for _, subset in grouped.items()]

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        return self.grouped[idx][0]


def get_mol_graph_data(args, split=None):
    dim_edge = args.edge_dim

    if args.dataset_type == 'GEOM_QM9':
        n_max = 29
        nval = 25000 - 105 #deducting the number of fragmented molecules from the total validation set
        ntst = 24068 - 404
    elif args.dataset_type == 'GEOM_Drugs':
        n_max = 181
        nval = 25000 - 38
        ntst = 14324

    data_dir = os.path.join(args.data_dir, args.dataset_type)
    suffix = ''
    if args.new_features:
        suffix = 'new_'
        args.standardise = False

    filenm = []
    filenm.append('%s_molvec_graph_%s%s.p' % (args.dataset_type, suffix, str(n_max)))
    filenm.append('%s_molset_graph_%s.p' % (args.dataset_type, str(n_max)))
    proc_filenm = ''

    if args.virtual_node:
        filenm = []
        filenm.append('%s_molvec_graph_%s%s_vn.p' % (args.dataset_type, suffix, str(n_max)))
        filenm.append('%s_molset_graph_%s_vn.p' % (args.dataset_type, str(n_max)))
        proc_filenm = 'vn_'
        n_max += 1
        dim_edge += 1

    dataset = []

    if split is None:
        split = ['train', 'val']

    all_points_mean = None
    all_points_std = None
    nodes_mean = None
    nodes_std = None
    edges_mean = None
    edges_std = None
    glbl_mean = None
    glbl_std = None

    for set in split:
        if (set == 'test' or set == 'val') and nodes_mean is None:
            save_dir = os.path.join("checkpoints", args.log_name)
            if not args.normalize_per_shape:
                all_points_mean = np.load(os.path.join(save_dir, "train_set_mean.npy"))
                all_points_std = np.load(os.path.join(save_dir, "train_set_std.npy"))
            if args.standardise:
                if not os.path.exists(os.path.join(save_dir, "feat_mean_std.npz")):
                    raise ValueError('File {} do not exist'.format("feat_mean_std.npz"))
                arr = np.load(os.path.join(save_dir, "feat_mean_std.npz"), allow_pickle=True)
                nodes_mean, edges_mean, glbl_mean, nodes_std, edges_std, glbl_std = \
                    arr['arr_0'], arr['arr_1'], arr['arr_2'], arr['arr_3'], arr['arr_4'], arr['arr_5']

        dataset.append(MolGraphSampling(
            root=data_dir,
            rawdata=filenm,
            procdata=proc_filenm,
            split=set,
            mask=args.masking,
            val_sample_size=nval,
            test_sample_size=ntst,
            normalize_per_shape=args.normalize_per_shape,
            normalize_std_per_axis=args.normalize_std_per_axis,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            node_dim=args.node_dim,
            edge_dim=dim_edge,
            max_atoms=n_max,
            standardise=args.standardise,
            nodes_mean=nodes_mean,
            nodes_std=nodes_std,
            edges_mean=edges_mean,
            edges_std=edges_std,
            glbl_mean=glbl_mean,
            glbl_std=glbl_std,
        ))
        if set == 'train':
            if not args.normalize_per_shape:
                all_points_mean = dataset[-1].all_points_mean
                all_points_std = dataset[-1].all_points_std
            nodes_mean = dataset[-1].nodes_mean
            nodes_std = dataset[-1].nodes_std
            edges_mean = dataset[-1].edges_mean
            edges_std = dataset[-1].edges_std
            glbl_mean = dataset[-1].glbl_mean
            glbl_std = dataset[-1].glbl_std

    return tuple(dataset)
