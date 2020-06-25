# Introduction
This is the codebase for the paper "[Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks](https://arxiv.org/abs/1911.00937)" by Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger Grosse, Jörn-Henrik Jacobsen to appear in NeurIPS 2019.

# Setup [Anaconda, CUDA10, Python-3.6, Pytorch 1.0.1.post2]
```
conda create -n test python=3.6
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

# Training and Generating Results
```
python -m lconvnet.run --cfg [dir to cfg.yaml] [--resume] [--test]
```
Put `--resume` when you want to resume from the best checkpoint (and continue training if the training was interrupted)
Put `--test` when you want to generate the results from the checkpoint

## Adversarial Robust Classification Experiments
For example, the following command launches an adversarial robustness experiment with BCOP on CIFAR10 and large architecture:
```
python -m lconvnet.run --cfg runs/adv-rob-cifar10/large/BCOP-Bjorck/multi-trial-A/cfg.yaml
```

After the training is finished, the following command will load the best checkpoint from the experiment and generate the provable/empirical robustness results in the experiment folder:
```
python -m lconvnet.run --cfg runs/adv-rob-cifar10/large/BCOP-Bjorck/multi-trial-A/cfg.yaml --resume --test
```

To evaluate the empirical robustness of baseline models that we are comparing with, the resume flag should be ommitted as the weights are implicitly baked in. For example, the following command will generate the results for KW-Large model on CIFAR10: 
```
python -m lconvnet.run --cfg runs/baselines/kw-cifar10/large/cfg.yaml --test
```

For L2Nonexpansive model, the pre-trained model needs to be downloaded from the authors' official website. This could be done by the following commands:
```
cd checkpoints/qian_models
chmod +x download.sh
./download.sh
```
Upon the completion of the downloads, the downaloded models could be used for evaluation. For example,
```
python -m lconvnet.run --cfg runs/baselines/qian-cifar10/model-3/cfg.yaml --test
```

## Wasserstein Distance Estimation Experiments
To launch a Wasserstein distance estimation experiment with BCOP, MaxMin activation function with a learning rate of 0.0001:
```
python -m lconvnet.run --cfg runs/wde_stl/lr-0.0001/conv/BCOP/maxmin/multi-trial-A/cfg.yaml
```

To launch the same experiment on the cifar10 dataset: 
```
python -m lconvnet.run --cfg runs/wde_cifar10/lr-0.0001/conv/BCOP/maxmin/multi-trial-A/cfg.yaml
```

The following command will report the lower-bound estimate of the model loaded from the checkpoint:
```
python -m lconvnet.run --cfg runs/wde_stl/lr-0.0001/conv/BCOP/maxmin/multi-trial-A/cfg.yaml --resume --test
```

## Table Generation
We also provide a convenient script to generate all the tables in the paper (with additional dependency on `pylatex`). After all the experiments have been completed, the following commands will generate a `table.tex` file under `runs/` (the consolidate command makes an copy of the results in each experiment folder and saves it as a unified file name: `results.yaml` and the export command walk through each individual experiment folders under `runs` to grab results from these files):
```
python -m lconvnet.consolidate --dir runs
python -m lconvnet.export --dir runs
```

# Pre-trained Models and Weights
All the model weights used in reporting can be downloaded from [here (Google Drive)](https://drive.google.com/file/d/1n0k9fLzgoq0y6A6Jc9gOr3FVjis7zfyG/view?usp=sharing) or [here (Dropbox)](https://www.dropbox.com/s/lxbg78ktdgwujfj/LConvNet-2020-04-25-release.zip?dl=0). It contains the `runs` folder with all the model weights and evaluation results placed under the appropriate experiment folders. 

# Generate a Batch of Experiment Configs from Template
```
./generate_experiments.sh [dir to where template.yaml is located]
```

To launch a batch of experiments on slurm:
```
sbatch [dir to where template.yaml is located]/batch_run.sh
```

To launch a batch of evaluation using the existing checkpoints:
```
sbatch [dir to where template.yaml is located]/batch_run_resume_test.sh
```

# Repo Structure
The repo is structured as follows
```
lconvnet
├── tasks
│   ├── adversarial
│   |   └── attackers.py                    "PGD, FGSM, Pointwise, Boundary attack" 
│   |   └── eval_robustness.py              "Robust accuracy upperbound by running the attacks
|   |                                        Robust accuracy lowerbound by certifying the Lipschitz network"
│   ├── gan                                 "Training the GAN for Wasserstein distance estimation experiments"
│   ├── wde                                 "GAN sampler for Wasserstein distance estimation"
│   └── common.py                           "Training step for different tasks (similar to train_step in pytorch lightning)"
├── layers 
│   ├── bcop.py                             "BCOP convolution"
│   ├── rko.py                              "RKO convolution"
│   ├── svcm.py                             "SVCM convolution"
│   ├── ossn.py                             "OSSN convolution"
│   ├── rkl2ne.py                           "RK-L2NE convolution"
│   └── ...py                               "Other GNP components"
├── external                                "Baselines"
│   ├── kw.py                               
│   └── qian.py                             
├── experiment                              "The main training loop and experiment management"
├── networks.py                             "FC/Small/Large/DCGANDiscriminator"
├── run.py                                  "Entry point"
├ ...
...
```

# Some Naming Conventions
- lr: learning rate
- small/large: the small/large network that is used by https://arxiv.org/abs/1805.12514
- fc: fully connected network
- conv: convolutional neural network
- `x`-layer: neural network with `x` hidden layers. Usually only used in describing fc network

# Citation
If you use this work, please cite with
```
@article{li2019lconvnet,
  title={Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks},
  author={Li, Qiyang and Haque, Saminul and Anil, Cem and Lucas, James and Grosse, Roger and Jacobsen, J{\"o}rn-Henrik},
  journal={Conference on Neural Information Processing Systems},
  year={2019}
}
```
