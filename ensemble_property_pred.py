import argparse
import pickle
import psi4
import random
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

psi4.core.be_quiet()
psi4.set_memory('270 GB')
psi4.set_num_threads(90)


class Subset(object):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def mol2xyz(mol):
    # mol = Chem.AddHs(mol)
    # AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    # AllChem.MMFFOptimizeMolecule(mol)
    atoms = mol.GetAtoms()
    string = "\n"
    for i, atom in enumerate(atoms):
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    string += "units angstrom\n"
    return string


def group_mol_by_smiles(dset):
    split = {}

    for i, data in enumerate(tqdm(dset)):
        smiles = Chem.MolToSmiles(data)
        if smiles in split:
            split[smiles].append(i)
        else:
            split[smiles] = [i]

    split = {k: Subset(dset, v) for k, v in split.items()}
    return split


def property_eval(mol):
    xyz = mol2xyz(mol)

    mol = psi4.geometry(xyz)
    psi4.set_options({'guess': 'sad', 'basis_guess': '3-21g', 'scf_type': 'df', 'ints_tolerance': 1.0E-8})
    scf_e, scf_wfn = psi4.energy("scf/cc-pvdz", molecule=mol, return_wfn=True)
    dipole_x, dipole_y, dipole_z = psi4.variable('SCF DIPOLE X'), psi4.variable('SCF DIPOLE Y'), psi4.variable(
        'SCF DIPOLE Z')
    dipole_moment = np.sqrt(dipole_x ** 2 + dipole_y ** 2 + dipole_z ** 2)

    HOMO = scf_wfn.epsilon_a_subset('AO', 'ALL').np[scf_wfn.nalpha() - 1]
    LUMO = scf_wfn.epsilon_a_subset('AO', 'ALL').np[scf_wfn.nalpha()]
    return HOMO, LUMO, scf_e, dipole_moment


def psi4_clean():
    """Function to put Psi4 back to a clean state.  In particular deletes
       scratch files and resets Psi variables.  Call as last part of the real
       deriv call to Psi4.
    """
    psi4.core.clean_variables()
    psi4.core.clean()


parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, default='./generated.pkl')
parser.add_argument('--model', type=str)
parser.add_argument('--gt', type=str)
parser.add_argument('--index', type=int)
args = parser.parse_args()

mollist,_ = pickle.load(open(args.out, 'rb'))
grouped = group_mol_by_smiles(mollist)
if 'GT' in args.model:
    filtered = list(grouped.items())
    random.Random(38383).shuffle(filtered)
    grouped = dict(filtered[args.index:args.index+1])
else:
    gt = pickle.load(open(args.gt, 'rb'))
    filtered = list(gt.keys())[args.index:args.index+1]
    grouped = {k: v for k, v in grouped.items() if k in filtered}

prop = dict()
error = []
for smile in tqdm(grouped):
    E = []
    H = []
    L = []
    DP = []
    HL_gap = []
    excl = 0
    for data in tqdm(grouped[smile]):
        psi4_clean()
        try:
            h, l, e, d = property_eval(data)
        except:
            excl += 1
            print('Smile %s: %d' % (smile, excl))
            continue
        H.append(h)
        L.append(l)
        HL_gap.append(h-l)
        E.append(e)
        DP.append(d)

    H = np.array(H)
    L = np.array(L)
    E = np.array(E)
    DP = np.array(DP)
    HL_gap = np.array(HL_gap)
    prop[smile] = np.array([E.mean(), HL_gap.mean(), E.min(), HL_gap.min(), HL_gap.max(), H.mean(), L.mean(), DP.mean()])
    if 'GT' not in args.model:
        error.append(np.abs(gt[smile] - prop[smile]))

pickle.dump(prop, open('results/'+args.model+str(args.index)+'.pkl', 'wb'))

if 'GT' not in args.model:
    error = np.vstack(error)
    MAE = np.mean(error, axis=0)
    MedAE = np.median(error, axis=0)
    print('Mean Absolute Error:')
    print("Mean E \t HOMO-LUMO gap \t E_min \t min-HOMO-LUMO gap \t max-HOMO-LUMO gap \t Mean HOMO \t Mean LUMO \t Dipole Moment")
    print("%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f" %(MAE[0], MAE[1], MAE[2], MAE[3], MAE[4], MAE[5], MAE[6], MAE[7]))
    print('Median of Absolute Error:')
    print("Mean E \t HOMO-LUMO gap \t E_min \t min-HOMO-LUMO gap \t max-HOMO-LUMO gap \t Mean HOMO \t Mean LUMO \t Dipole Moment")
    print("%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f" %(MedAE[0], MedAE[1], MedAE[2], MedAE[3], MedAE[4], MedAE[5], MedAE[6], MedAE[7]))
