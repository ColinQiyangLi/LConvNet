import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops.layers.torch import Rearrange, Reduce
from yaml import load, dump
from spaghettini import quick_register

from munch import Munch

import types


###########################################
#     Register Customized Modules Here    #
###########################################
from lconvnet.networks import MLP, SmallConvNet, LargeConvNet, DCGANDiscriminator

quick_register(MLP)
quick_register(SmallConvNet)
quick_register(LargeConvNet)
quick_register(DCGANDiscriminator)

from lconvnet.layers import (
    BCOP,
    RKO,
    OSSN,
    SVCM,
    NonexpansiveConv2d,
    LipschitzConv2d,
    l2_lipschitz_constant_checker,
    BjorckLinear,
    GroupSort,
    PixelUnshuffle2d,
)

quick_register(LipschitzConv2d)
quick_register(l2_lipschitz_constant_checker)

quick_register(BCOP)
quick_register(RKO)
quick_register(OSSN)
quick_register(SVCM)
quick_register(NonexpansiveConv2d)

quick_register(BjorckLinear)
quick_register(GroupSort)
quick_register(PixelUnshuffle2d)

from lconvnet.datasets import cifar10, mnist, default_dataloader

quick_register(cifar10)
quick_register(mnist)
quick_register(default_dataloader)

from lconvnet.experiment import (
    Experiment,
    Trainer,
    default_loss_comparator,
    default_accuracy_comparator,
)

quick_register(Experiment)
quick_register(Trainer)
quick_register(default_loss_comparator)
quick_register(default_accuracy_comparator)

from lconvnet.external import kw, qian

quick_register(kw)
quick_register(qian)

from lconvnet.tasks.adversarial import (
    FoolBoxAttacker,
    PGDAttacker,
    FGSMAttacker,
    eval_adv_robustness_batch,
)
from lconvnet.tasks.wde import GANSampler, DistributionGANLoader, repeated_loaders
from lconvnet.tasks.gan.models import WGAN_GP
from lconvnet.tasks.common import (
    classification_step,
    wasserstein_distance_estimation_step,
    multi_margin_loss_eps,
    diff_loss,
)

quick_register(FoolBoxAttacker)
quick_register(PGDAttacker)
quick_register(FGSMAttacker)
quick_register(eval_adv_robustness_batch)

quick_register(GANSampler)
quick_register(DistributionGANLoader)
quick_register(repeated_loaders)
quick_register(WGAN_GP)

quick_register(classification_step)
quick_register(wasserstein_distance_estimation_step)
quick_register(multi_margin_loss_eps)
quick_register(diff_loss)

#########################################
#     Register Library Modules Here     #
#########################################
quick_register(nn.Conv2d)
quick_register(nn.BatchNorm2d)
quick_register(nn.CrossEntropyLoss)
quick_register(nn.AdaptiveAvgPool2d)
quick_register(nn.AdaptiveMaxPool2d)
quick_register(nn.MaxPool2d)
quick_register(nn.Linear)
quick_register(nn.Sequential)
quick_register(nn.ReLU)
quick_register(F.relu)
quick_register(F.cross_entropy)
quick_register(F.multi_margin_loss)
quick_register(F.sigmoid)
quick_register(optim.lr_scheduler.MultiStepLR)
quick_register(optim.lr_scheduler.ExponentialLR)
quick_register(optim.SGD)
quick_register(optim.Adam)
quick_register(optim.RMSprop)

quick_register(Rearrange)
quick_register(Reduce)

quick_register(torch.tensor)
quick_register(torch.arange)
quick_register(torch.clamp)

quick_register(Munch)
