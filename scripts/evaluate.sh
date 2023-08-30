#!/bin/bash

#data options
ds=GEOM_Drugs
data_dir="data/"
node_dim=11
edge_dim=12
glbl_dim=2
max_atoms=181

#evaluation options
split='test'

#logging options
output_dir=${1}
log_name="${ds}/${output_dir}"

python evaluate.py \
    --onlyHeavy \
    --standardise \
    --valsplit ${split} \
    --dataset_type ${ds} \
    --data_dir ${data_dir} \
    --node_dim ${node_dim} \
    --edge_dim ${edge_dim} \
    --global_dim ${glbl_dim} \
    --log_name ${log_name} \
    --new_features #(TODO:) only for GEOM_Drugs
    #--mmff

echo "Done"
exit 0
