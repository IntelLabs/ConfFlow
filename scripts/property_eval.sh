#!/bin/bash
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH -c 20

echo $SLURM_JOB_NODELIST
echo $PATH
export PSI_SCRATCH=$TMPDIR

ds=GEOM_Drugs
output_dir=${1}
out="./checkpoints/${ds}/${output_dir}/generated.pkl"
modelname=ConfFlow_QM9 #name describing the model and dataset
gt="./results/${ds}_GT.pkl"

python ensemble_property_pred.py --out=$out --model=modelname --gt=$gt --index=$SLURM_ARRAY_TASK_ID
