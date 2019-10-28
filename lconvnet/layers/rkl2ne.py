"""
An implementation of reshaped kernel method using the spectral norm bound described in 
https://arxiv.org/pdf/1802.07896.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lconvnet.layers.core import LipschitzModuleL2
from lconvnet.layers.utils import conv2d_cyclic_pad, conv_singular_values_numpy
import numpy as np


class NonexpansiveConv2d(LipschitzModuleL2, nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        nn.Conv2d.__init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    
    def singular_values(self):
        svs = torch.from_numpy(conv_singular_values_numpy(self.buffer_weight.detach().cpu().numpy(), self._input_shape)).to(device=self.buffer_weight.device)
        return svs

    def linf_norm(self, weight):
        return weight.abs().sum(dim=1).max()

    # compute a bound on the spectral norm of the reshaped kernel
    def get_factor(self):
        weight = self.weight.view(self.weight.size(0), -1)
        return (torch.min(self.linf_norm(weight @ weight.t()), self.linf_norm(weight.t() @ weight))) ** 0.5

    def forward(self, x):
        self._input_shape = x.shape[-2:]
        factor = self.get_factor()

        # scale the reshape kerenl down by the upperbound of its spectral norm and 
        # additionally sacle it down by its kernel size to make sure the spectral norm 
        # of the convolution is at most 1
        self.buffer_weight = self.weight / factor / ((self.kernel_size[0] * self.kernel_size[1]) ** 0.5)
        
        # cyclic-padded convolution using the scaled weight
        return conv2d_cyclic_pad(
            x, self.buffer_weight, self.bias
        )
