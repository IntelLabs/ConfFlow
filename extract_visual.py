import os
import sys
import numpy as np
import torch
import pickle
from copy import deepcopy
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.rdMolAlign import GetBestRMS, AlignMol
from rdkit.Chem import RemoveHs

from datasets import get_mol_graph_data
from args import get_args
from utils import split_dataset_by_smiles


def group_mol_by_smiles(dset, smilist=None):
    split = {}

    for i, data in enumerate(tqdm(dset)):
        if smilist is not None:
            smiles = smilist[i]
        else:
            smiles = Chem.MolToSmiles(data)
        if smiles in split:
            split[smiles].append(i)
        else:
            split[smiles] = [i]

    split = {k:torch.utils.data.Subset(dset, v) for k, v in split.items()}
    return split


def main(args, model_dir):
    test_dset = get_mol_graph_data(args, split=[args.valsplit])[0]
    mollist = test_dset.mollist

    grouped = split_dataset_by_smiles(test_dset, mollist)

    with open(os.path.join(model_dir, 'generated_2samp.pkl'), 'rb') as f:
        generated, y = pickle.load(f)

    if args.mmff:
        generated = y

    grouped_generated = group_mol_by_smiles(generated)

    gen_ind_list = []
    gt_ind_list = []
    smile_list = []
    MAT = []
    not_found = 0
    for smiles in grouped:
        if smiles not in grouped_generated:
            not_found += 1
            continue
        total_conf = len(grouped[smiles])
        if total_conf < 3:
            continue
        smile_list.append(smiles)
        min_rms = []
        min_arg = []
        for data in grouped[smiles]:
            prbmol = mollist[data.idx]
            rms = []
            for mol in grouped_generated[smiles]:
                try:
                    if args.onlyHeavy:
                        rms.append(GetBestRMS(RemoveHs(deepcopy(prbmol)), RemoveHs(mol)))
                    else:
                        rms.append(AlignMol(deepcopy(prbmol), mol))
                except:
                    print("Substructure exception")
            min_rms.append(min(rms))
            min_arg.append(np.argmin(rms))
        rms_arr = np.array(min_rms)
        arg_arr = np.array(min_arg)
        ind = np.unravel_index(np.argsort(rms_arr, axis=None), rms_arr.shape)
        rms_best = rms_arr[ind][:3]
        arg_best = arg_arr[ind][:3]
        gen_ind_list.append(arg_best)
        gt_ind_list.append(ind[0][:3])
        MAT.append(rms_best.sum() / 3)

    MAT = np.array(MAT)
    gen_ind_list = np.array(gen_ind_list)
    gt_ind_list = np.array(gt_ind_list)

    ind = np.unravel_index(np.argsort(MAT, axis=None), MAT.shape)
    indbest = (ind[0][:10]).tolist()
    smile_best = []
    gt_best = []
    gen_best = []
    for index in indbest:
        smile_best.append(smile_list[index])
        gt_mol = grouped[smile_best[-1]]
        gen_mol = grouped_generated[smile_best[-1]]
        gt_ind = gt_ind_list[index]
        gen_ind = gen_ind_list[index]
        gt_conf = []
        gen_conf = []
        for i in range(3):
            data = gt_mol[gt_ind[i]]
            mol = deepcopy(mollist[data.idx])
            gt_conf.append(mol)
            gen_conf.append(gen_mol[gen_ind[i]])
        gt_best.append(np.array(gt_conf))
        gen_best.append(np.array(gen_conf))

    pickle.dump([smile_best, gt_best], open(os.path.join(model_dir, 'gt.pkl'), 'wb'))
    pickle.dump(gen_best, open(os.path.join(model_dir, 'generated.pkl'), 'wb'))


if __name__ == '__main__':
    # command line args
    args = get_args()

    model_dir = os.path.join("checkpoints", args.log_name)
    assert os.path.exists(model_dir), "Model directory do not exist!!"

    with open(os.path.join(model_dir, 'evaluate_cmd.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    print("Arguments:")
    print(args)

    main(args, model_dir)
