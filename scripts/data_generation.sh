#!/bin/bash

cd data/
data=${1}

python ${data}_featurize_graph.py --loaddir="./${data}/" --savedir="./${data}/raw/"
