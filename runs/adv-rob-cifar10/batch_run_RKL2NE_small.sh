#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --array=0-4%5
#SBATCH -c 2

list=(
    "python -m lconvnet.run --cfg runs/adv-rob-cifar10/small/L2Nonexpansive/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg runs/adv-rob-cifar10/small/L2Nonexpansive/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg runs/adv-rob-cifar10/small/L2Nonexpansive/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg runs/adv-rob-cifar10/small/L2Nonexpansive/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg runs/adv-rob-cifar10/small/L2Nonexpansive/multi-trial-C/cfg.yaml"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
