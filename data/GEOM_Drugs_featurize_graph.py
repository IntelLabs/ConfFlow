from __future__ import print_function
import os
import numpy as np
import torch
import pickle as pkl
import argparse
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt


parser = argparse.ArgumentParser()
parser.add_argument('--loaddir', type=str, default='./')
parser.add_argument('--savedir', type=str, default='./')

args = parser.parse_args()


# allowable multiple choice node and edge features
allowable_features = {
    'possible_atoms_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_number_valence_e_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 'misc'],
    'possible_hybridization_list' : [
        'S','SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
        'misc'
    ],
    'possible_is_conjugated_list': [False, True, 'misc'],
    'possible_same_ring': [False, True],
    'possible_shortest_path': [1, 2, 3, 'misc'],
    'possible_ring_size': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom, ri_a):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atoms_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_valence_e_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_number_valence_e_list'], atom.GetTotalValence()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atoms_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_valence_e_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_number_valence_e_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

# Rings
MAX_RING_SIZE = 9
RING_SIZES = range(3, MAX_RING_SIZE + 1)
def bond_to_feature_vector(bond, samering, shortpath):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    NOT_IN_RING = tuple(0 for _ in RING_SIZES)

    if len(bond) == 1:
        bond = bond[0]
        bond_feature = [
                    safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                    allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                    allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                    allowable_features['possible_same_ring'].index(samering),
                    safe_index(allowable_features['possible_shortest_path'], shortpath),
                ] + [allowable_features['possible_ring_size'].index(bond.IsInRingSize(size)) for size in RING_SIZES]
    else:
        bond_feature = [
                    allowable_features['possible_bond_type_list'].index('misc'),
                    allowable_features['possible_bond_stereo_list'].index('misc'),
                    allowable_features['possible_is_conjugated_list'].index('misc'),
                    allowable_features['possible_same_ring'].index(samering),
                    safe_index(allowable_features['possible_shortest_path'], shortpath),
                ] + list(NOT_IN_RING)

    return bond_feature


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list'],
        allowable_features['possible_same_ring'],
        allowable_features['possible_shortest_path'],
        ])) + [2] * len(RING_SIZES)


data = 'GEOM_Drugs'
n_min = 2
n_max = 181
atom_dim = 11
edge_dim = 12

mollist, smilist = pkl.load(open(args.loaddir+data+'_molset_all.p','rb'))

excl = []
mollist2 = []
smilist2 = []
nodes = []
edges = []
edge_index = []
glbl = []
pos = []

for i in range(len(mollist)):
    if i % 1000 == 0:
        print(i, len(mollist), flush=True)
    mol = mollist[i]

    frags = Chem.GetMolFrags(mol)
    if len(frags) > 1:
        excl.append(i)
        continue

    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistry(mol)

    n = mol.GetNumAtoms()

    if n < n_min or n > n_max:
        print('error')
        break

    ri = mol.GetRingInfo()
    ri_a = ri.AtomRings()

    temp = mol.GetConformer(0).GetPositions()
    assert n == temp.shape[0]

    mollist2.append(mol)
    smilist2.append(smilist[i])
    pos.append(temp)

    # Global Feature
    molwt = MolWt(mol)
    avgwt = molwt / n
    avgbond = float(mol.GetNumBonds(onlyHeavy=False)) / n
    glbl.append(np.array([avgwt, avgbond]))

    node_feat = np.zeros((n, atom_dim))
    for j in range(n):
        atom = mol.GetAtomWithIdx(j)
        node_feat[j, :] = atom_to_feature_vector(atom, ri_a)

    nodes.append(node_feat)

    edge = []
    start = []
    end = []
    for j in range(n-1):
        for k in range(j+1, n):
            molpath = Chem.GetShortestPath(mol, j, k)
            shortpath = len(molpath) - 1
            assert shortpath > 0

            samering = 0
            for alist in ri_a:
                if j in alist and k in alist:
                    samering = 1

            bond = [mol.GetBondBetweenAtoms(molpath[mm], molpath[mm+1]) for mm in range(shortpath)]
            if len(bond) < 4:
                edge.extend(2 * [bond_to_feature_vector(bond, samering, shortpath)])
                start.extend([j, k])
                end.extend([k, j])

    edges.append(np.array(edge))
    edge_index.append(torch.tensor([start, end], dtype=torch.long))


atom_feat_dims = get_atom_feature_dims()
edge_feat_dims = get_bond_feature_dims()

os.makedirs(args.savedir, exist_ok=True)

molvec_fname = args.savedir + data + '_molvec_graph_new_' + str(n_max) + '.p'
molset_fname = args.savedir + data + '_molset_graph_' + str(n_max) + '.p'

print(molvec_fname)
print(molset_fname)
print("Total fragmented molecules that are excluded:", len(excl))
print(excl)

with open(molvec_fname,'wb') as f:
    pkl.dump([nodes, edges, edge_index, glbl, pos], f, protocol=4)

mollist2 = np.array(mollist2)
smilist2 = np.array(smilist2)

with open(molset_fname,'wb') as f:
    pkl.dump([mollist2, smilist2, atom_feat_dims, edge_feat_dims], f)
