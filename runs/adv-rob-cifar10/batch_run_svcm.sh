#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --array=0-9%10
#SBATCH -c 2

list=(
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/small/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/small/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/small/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/small/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/small/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/large/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/large/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/large/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/large/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg runs/release/adv-rob-cifar10/multi-margin/large/SVCM/proj-50/non-ortho/lr-0.001/multi-trial-C/cfg.yaml"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
