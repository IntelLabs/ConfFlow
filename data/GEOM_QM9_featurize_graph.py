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


def to_onehot(val, cat, etc=0):

    onehot=np.zeros(len(cat))
    for ci, c in enumerate(cat):
        if val == c:
            onehot[ci]=1

    if etc==1 and np.sum(onehot)==0:
        print(val)

    return onehot


def atomFeatures(a, ri_a):

    def _ringSize_a(a, rings):
        onehot = np.zeros(6)
        aid = a.GetIdx()
        for ring in rings:
            if aid in ring and len(ring) <= 8:
                onehot[len(ring) - 3] += 1

        return onehot

    v1 = to_onehot(a.GetSymbol(), ['H','C','N','O','F'], 1)
    v2 = to_onehot(str(a.GetHybridization()), ['S','SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'], 1)

    v3 = [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(), a.GetImplicitValence(), a.GetNumRadicalElectrons(), int(a.GetIsAromatic())]
    v4 = _ringSize_a(a, ri_a)

    v5 = np.zeros(3)
    v6 = to_onehot(a.GetChiralTag(), [Chem.CHI_UNSPECIFIED, Chem.CHI_TETRAHEDRAL_CW, Chem.CHI_TETRAHEDRAL_CCW], 1)
    try:
        tmp = to_onehot(a.GetProp('_CIPCode'), ['R','S'], 1)
        v5[0] = tmp[0]
        v5[1] = tmp[1]
    except:
        v5[2]=1

    v5 = v5[:2]

    return np.concatenate([v1,v2,v3,v4,v5,v6], axis=0)


# Rings
MAX_RING_SIZE = 9
RING_SIZES = range(3, MAX_RING_SIZE + 1)
NOT_IN_RING = tuple(0 for _ in RING_SIZES)
def bondFeatures(bbs, samering, shortpath):
    if len(bbs) == 1:
        v0 = [1, 0]
        v1 = to_onehot(str(bbs[0].GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'], 1)
        v2 = to_onehot(str(bbs[0].GetStereo()), ['STEREOZ', 'STEREOE', 'STEREOANY', 'STEREONONE', 'STEREOTRANS', 'STEREOCIS'], 1)
        v3 = [int(bbs[0].GetIsConjugated()), samering, shortpath] + [int(bbs[0].IsInRingSize(size)) for size in RING_SIZES]
    else:
        v0 = [0, 1]
        v1 = np.zeros(4)
        v2 = np.zeros(6)
        v3 = [0, samering, shortpath] + list(NOT_IN_RING)

    return np.concatenate([v0, v1, v2, v3], axis=0)


data = 'GEOM_QM9'
n_min = 2
n_max = 29
atom_dim = 29
edge_dim = 22

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
        node_feat[j, :] = atomFeatures(atom, ri_a)

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
                edge.extend(2 * [bondFeatures(bond, samering, shortpath)])
                start.extend([j, k])
                end.extend([k, j])

    edges.append(np.array(edge))
    edge_index.append(torch.tensor([start, end], dtype=torch.long))

os.makedirs(args.savedir, exist_ok=True)

molvec_fname = args.savedir + data + '_molvec_graph_new_' + str(n_max) + '.p'
molset_fname = args.savedir + data + '_molset_graph_new_' + str(n_max) + '.p'

print(molvec_fname)
print(molset_fname)
print(excl)

with open(molvec_fname,'wb') as f:
    pkl.dump([nodes, edges, edge_index, glbl, pos], f, protocol=4)

mollist2 = np.array(mollist2)
smilist2 = np.array(smilist2)

with open(molset_fname,'wb') as f:
    pkl.dump([mollist2, smilist2, [], []], f)
