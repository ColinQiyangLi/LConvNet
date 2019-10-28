#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --array=0-1%12
#SBATCH -c 2

list=(
    "python -m lconvnet.run --cfg runs/baselines/kw-mnist/large/cfg.yaml"
    "python -m lconvnet.run --cfg runs/baselines/kw-mnist/small/cfg.yaml"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
