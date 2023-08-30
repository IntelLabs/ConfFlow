#!/bin/bash

ds=GEOM_Drugs
output_dir=${1}
testdata=${2}

python distance_eval.py \
      --out=./checkpoints/${ds}/${output_dir}/generated.pkl \
      --testdata=./data/${ds}/${testdata}.pkl \
      --onlyHeavy
