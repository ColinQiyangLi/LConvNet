#!/bin/bash
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --array=0-9%10
#SBATCH -c 2

list=(
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg runs/wde/conv/OSSN-old/n_iters-10/relu/multi-trial-C/cfg.yaml"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
