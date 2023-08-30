import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PyMol


smiles, gt = pickle.load(open("gt.pkl", 'rb'))
cf = pickle.load(open("conflow.pkl", 'rb'))
cgcf = pickle.load(open("cgcf.pkl", 'rb'))
confgf = pickle.load(open("confgf.pkl", 'rb'))

index = 0
for smile in smiles:
    mol = Chem.MolFromSmiles(smile)
    Draw.MolToFile(mol, str(index) + '.png')
    index += 1

v = PyMol.MolViewer()

for ind in range(len(gt)):
    print(ind)
    v.DeleteAll()
    gtmol = gt[ind]
    cfmol = cf[ind]
    confgfmol = confgf[ind]
    cgcfmol = cgcf[ind]

    index = 0
    for conf in gtmol:
        v.ShowMol(conf, name='gt_' + str(index), showOnly=False)
        index += 1

    index = 0
    for conf in cfmol:
        v.ShowMol(conf, name='gcgf_' + str(index), showOnly=False)
        index += 1

    index = 0
    for conf in confgfmol:
        v.ShowMol(conf, name='confgf_' + str(index), showOnly=False)
        index += 1

    index = 0
    for conf in cgcfmol:
        v.ShowMol(conf, name='cgcf_' + str(index), showOnly=False)
        index += 1

    v.server.do('set grid_mode, 3')
    input("Press Enter to continue...")
