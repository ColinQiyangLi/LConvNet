#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --array=0-119%4
#SBATCH -c 2

list=(
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/BCOP/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/RKO/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.0001/conv/OSSN/n_iters-10/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/OSSN/n_iters-10/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/RKO/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.01/conv/BCOP/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/BCOP/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/OSSN/n_iters-10/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.00001/conv/RKO/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/OSSN/n_iters-10/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/maxmin/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/BCOP/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/relu/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/relu/multi-trial-B/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/relu/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/relu/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/relu/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/maxmin/multi-trial-C/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/maxmin/multi-trial-D/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/maxmin/multi-trial-A/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/maxmin/multi-trial-E/cfg.yaml"
    "python -m lconvnet.run --cfg ./runs/wde_cifar10_3/lr-0.001/conv/RKO/maxmin/multi-trial-B/cfg.yaml"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
