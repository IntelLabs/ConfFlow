import os
import sys
import numpy as np
import torch
import pickle
from tqdm.auto import tqdm
from copy import deepcopy
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

    with open(os.path.join(model_dir, 'generated.pkl'), 'rb') as f:
        generated, y = pickle.load(f)

    if args.mmff:
        generated = y

    grouped_generated = group_mol_by_smiles(generated)

    conf_count = []
    rms_list = []
    MAT = []
    not_found = 0
    for smiles in grouped:
        if smiles not in grouped_generated:
            not_found += 1
            continue
        total_conf = len(grouped[smiles])
        conf_count.append(total_conf)
        min_rms = []
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
        rms_arr = np.array(min_rms)
        rms_list.append(rms_arr)
        MAT.append(rms_arr.sum() / total_conf)

    MAT = np.array(MAT)
    print("Mean MAT: %1.4f, Median MAT: %1.4f" % (np.mean(MAT), np.median(MAT)))

    print('Not found:', not_found)
    print("Thres, Mean COV, Median COV")
    mean = []
    std = []
    for thres in np.arange(0.0, 2.0, 0.1).tolist() + [1.25]:
        COV = []
        for a,b in zip(rms_list, conf_count):
            COV.append((a < thres).sum() / b)
        COV = np.array(COV)
        mean.append(np.mean(COV))
        std.append(np.std(COV))
        print("%1.1f    %1.4f   %1.4f" % (thres, np.mean(COV), np.median(COV)))
    print(mean[:-1])
    print(std[:-1])

    conf_count = []
    rms_list = []
    not_found = 0
    for smiles in grouped_generated:
        if smiles not in grouped:
            not_found += 1
            continue
        total_conf = len(grouped_generated[smiles])
        conf_count.append(total_conf)
        min_rms = []
        for mol in grouped_generated[smiles]:
            rms = []
            for data in grouped[smiles]:
                refmol = mollist[data.idx]
                try:
                    if args.onlyHeavy:
                        rms.append(GetBestRMS(RemoveHs(deepcopy(mol)), RemoveHs(refmol)))
                    else:
                        rms.append(AlignMol(deepcopy(mol), refmol))
                except:
                    print("Substructure exception")
            min_rms.append(min(rms))
        rms_arr = np.array(min_rms)
        rms_list.append(rms_arr)

    print('Not found:', not_found)
    print("Thres, Mean JUNK, Median JUNK")
    for thres in np.arange(0.0, 2.0, 0.1).tolist() + [1.25]:
        JUNK = []
        for a,b in zip(rms_list, conf_count):
            JUNK.append((a > thres).sum() / b)
        JUNK = np.array(JUNK)
        print("%1.1f    %1.4f   %1.4f" % (thres, np.mean(JUNK), np.median(JUNK)))


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
