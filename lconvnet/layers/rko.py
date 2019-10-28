"""
An implementation of reshaped kernel orthogonalization (RKO) method using Bjorck
"""

from lconvnet.layers.utils import conv_clip_2_norm_numpy, conv_singular_values_numpy
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch
import numpy as np

from lconvnet.layers.core import LipschitzModuleL2
from lconvnet.layers.linear import BjorckLinear
from lconvnet.layers.utils import conv2d_cyclic_pad

class RKO(BjorckLinear):
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
        bjorck_beta=0.5,
        bjorck_iters=20,
        bjorck_order=1,
        power_iteration_scaling=True,
    ):
        self.stride = stride
        self.kernel_size = kernel_size
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        super().__init__(
            in_features=in_channels * kernel_size[0] * kernel_size[1],
            out_features=out_channels,
            bias=bias,
            bjorck_beta=bjorck_beta,
            bjorck_iters=bjorck_iters,
            bjorck_order=bjorck_order,
            power_iteration_scaling=power_iteration_scaling,
        )

    def singular_values(self):
        svs = torch.from_numpy(
            conv_singular_values_numpy(
                self.buffer_weight.detach().cpu().numpy(), self._input_shape
            )
        ).to(device=self.buffer_weight.device)
        return svs

    def forward(self, x):
        self._input_shape = x.shape[-2:]

        # orthogonalize the unconstrained matrix using bjorck
        if not self.streamline:
            self.bjorck_weight = self.ortho_w()
        else:
            if self.bjorck_weight is None:
                with torch.no_grad():
                    self.bjorck_weight = self.ortho_w()

        # reshape the kernel back and scale it down by the kernel size
        self.buffer_weight = self.bjorck_weight.view(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        ) / ((self.kernel_size[0] * self.kernel_size[1]) ** 0.5)
        
        # detach the weight from the computation graph if we are using the cached weight
        if self.streamline:
            buffer_weight = self.buffer_weight.detach()
        else:
            buffer_weight = self.buffer_weight
        return conv2d_cyclic_pad(x, buffer_weight, self.bias)

    def extra_repr(self):
        "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, power_iteraion_scaling={power_iteration_scaling}, bias={enable_bias}".format(
            super().extra_repr(), **self.__dict__
        )
