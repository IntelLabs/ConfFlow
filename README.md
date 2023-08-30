# Conformation Generation using Transformer Flows #
[Sohil Atul Shah](https://sites.google.com/site/sas21587/) and [Vladlen Koltun](http://vladlen.info/)

[[`arXiv`](http://arxiv.org/abs/)] [[`BibTeX`](#CitingConfFlow)] 

![sequence](./assets/movie.gif)

## Features
* Fast synthesis of 3D molecular conformers for a given input 2D graph.
* Highly interpretable procedure akin to force field updates in molecular dynamics simulation.

## Requirements
- Linux with Python 3.8
- PyTorch >= 1.13.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- Create and install all dependencies using `conda env create -f environment.yml`

## Getting Started
To train a model, first setup the corresponding datasets following [scripts/data_generation.sh](./scripts/data_generation.sh).
### Datasets
The offical raw GEOM dataset is available [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

#### Dataset split and preprocessing
For ease of benchmarking and comparison, we use the default split of GEOM as provided by [[ConfGF]](https://github.com/DeepGraphLearning/ConfGF/). For each category of GEOM we first retrieve the corresponding dataset compressed file from [[google drive folder]](https://drive.google.com/drive/folders/10dWaj5lyMY0VY4Zl0zDPCa69cuQUGb-6?usp=sharing) and unpack it into the corresponding folder under [data](./data/) folder, eg: `data/GEOM_QM9/raw/`. Following this, we pack all the molecules and their smiles structures for all splits into a single raw dataset file as shown below.   
```python
import pickle

mols,smiles,mollist = [],[], []
mollist += pickle.load(open('train_data_40k.pkl', 'rb'))
mollist += pickle.load(open('val_data_5k.pkl', 'rb'))
mollist += pickle.load(open('test_data_200.pkl', 'rb'))
for mol in mollist:
  mols.append(mol.rdmol)
  smiles.append(mol.smiles)
pickle.dump([mols, smiles], open('GEOM_QM9_molset_all.p', 'wb'))
```

With split taken care of, we generate the preprocessed data by running their featurization code under the [[data]](./data/) folder. You can run the featurization code using [[data_generation]](./scripts/data_generation.sh) script,
```commandline
sh scripts/data_generation.sh GEOM_QM9
```

During the first training / evaluation run on the corresponding dataset, the `torch_geometric` package packs each split into its compressed form under the `./data/processed` folder for faster in-memory future retrievals.

The final dataset folder structure will look like this.
```
GEOM_QM9
|___train_data_40k.pkl
|___val_data_5k.pkl
|___test_data_200.pkl
|___GEOM_QM9_molset_all.p
|___raw
|   |___GEOM_QM9_molvec_graph_29.p
|   |___GEOM_QM9_molset_graph_29.p
|___processed
|   |___train_1_0.pt
|   |___val_1_0.pt
|   |___test_1_0.pt
|   |___train_1_0_extra.npz
|   |___val_1_0_extra.npz
|   |___test_1_0_extra.npz
|   |___pre_transform.pt
|   |___pre_filter.pt
|
...
```

### Training
All hyper-parameters and training details are provided in script files (`./scripts/train_*.sh`), and free feel to tune these parameters.

You can train the model with the following command:

```commandline
sh train_geom_qm9.sh output_dir_name
```
The folder `output_dir_name` is created under the `./checkpoints/${dataset}/` directory. 

### Generation
We provide generation / evaluation scripts for the GEOM_Drugs data. Same can be replicated for other dataset by referring to the config listed in the training scripts. 

The 3d structures of the test set is generated with the following command: 
```commandline
sh test_geom_drugs.sh output_dir_name
```
These generated structures are stored in file `generated.pkl` under the `output_dir_name` folder.

### Evaluation
Following the generation, one can run various evaluation with the following commands:
```commandline
sh evaluate.sh output_dir_name
sh property_eval.sh output_dir_name
sh distance_eval.sh output_dir_name test_data_200
```
The `evaluate.sh` computes three scores `COV, MAT, MIS` w.r.t. to RMSD to GT structures (Table 1) using `evaluate.py`.
Whereas the `property_eval.sh` evaluates the median of absolute prediction errors of various ensemble properties (Table 3) using `ensemble_property_pred.py`.

### Visualization
Following command generates the visual 3D structure of few sampled test data molecules. Feel free to modify files in order to generate different set of molecules and baselines.   

```commandline
sh extract_visual.sh output_dir_name
python pymol_visual.py
```

## Model Zoo and Baselines
We provide a trained models for both GEOM_Drugs and GEOM_QM9 available in the [checkpoints](./checkpoints/) folder.


## License
Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The source code and dataset are licensed under a [MIT License](LICENSE). In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

## <a name="CitingConfFlow"></a>Citing ConfFlow
If you use ConfFlow code or trained models in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{shah2023ConfFlow,
  title={Conformation Generation using Transformer Flows},
  author={Sohil Atul Shah and Vladlen Koltun},
  journal={arXiv:},
  year={2023}
}
```

## Contact

[Sohil Shah](sohil.iitb@gmail.com)