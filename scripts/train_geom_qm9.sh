#!/bin/bash

#model architecture options
input_dim=3
dims="32-32"
num_blocks=2
point_layer="mpnnpointTx"
act="swish"
atol=1e-3
rtol=1e-3
ke=0.2
jn=0.2
std=1.0

#training options
zdim=0
batch_size=125 # per gpu
lr=1e-3
lr_decay=0.1
sch='warm'
optim='adam'
warm_iter=1730
total_epochs=45
CG=0.05
wd=0.

#data options
ds=GEOM_QM9
data_dir="data/"
valsplit="val"
node_dim=29
edge_dim=22
glbl_dim=2
max_atoms=29

#logging options
output_dir=${1}
log_name="${ds}/${output_dir}"

#misc options
nworker=12
seed=380847

python train_pl.py --valsplit ${split} \
    --val_sample_mol 2 \
    --val_sampling \
    --gpu 1 \
    --jacobian_norm2 ${jn} \
    --kinetic_energy ${ke} \
    --weight_decay ${wd} \
    --optimizer ${optim} \
    --sigma ${std} \
    --clip_grad ${CG} \
    --standardise \
    --seed ${seed} \
    --atol ${atol} \
    --rtol ${rtol} \
    --warmup_iters ${warm_iter} \
    --nonlinearity ${act} \
    --input_dim ${input_dim} \
    --dims ${dims} \
    --num_blocks ${num_blocks} \
    --layer_type_point ${point_layer} \
    --zdim ${zdim} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --scheduler ${sch} \
    --lr_decay ${lr_decay} \
    --epochs ${total_epochs} \
    --num_workers ${nworker} \
    --dataset_type ${ds} \
    --data_dir ${data_dir} \
    --node_dim ${node_dim} \
    --edge_dim ${edge_dim} \
    --global_dim ${glbl_dim} \
    --max_atoms ${max_atoms} \
    --log_name ${log_name} \
    --save_freq 1 \
    --val_freq 1.0 \
    --viz_freq 10 \
    --resume_optimizer

echo "Done"
exit 0
