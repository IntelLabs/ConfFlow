import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PyMol

cf, gt = pickle.load(open("sequence.pkl", 'rb'))

print(len(gt))

smile = Chem.MolToSmiles(gt[0])
mol = Chem.MolFromSmiles(smile)
Draw.MolToFile(mol, '0.png')

v = PyMol.MolViewer()
v.DeleteAll()

for ind in range(15, len(cf), 15):
    v.ShowMol(cf[ind], name='traj' + str(ind), showOnly=False)

v.ShowMol(cf[-1], name=str(len(gt)-1), showOnly=False)

v.server.do('set grid_mode, 3')
