"""
A wrapper of the numpy implementation of singular value clipping and masking method as described in
https://arxiv.org/pdf/1805.10408.pdf. 
"""

import torch
import torch.nn as nn

from lconvnet.layers.utils import conv_clip_2_norm_numpy, conv_singular_values_numpy, conv2d_cyclic_pad
from lconvnet.layers.core import LipschitzModuleL2

class SVCM(nn.Conv2d, LipschitzModuleL2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_projections=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        orthogonal=False,
        zero_padding=False,
        run_forward=True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias
        )
        self.num_projections = num_projections
        self.orthogonal = orthogonal
        self.zero_padding = zero_padding
        self.run_forward = run_forward

    def singular_values(self):
        return torch.from_numpy(conv_singular_values_numpy(
            self.weight.data.cpu().numpy(), self._input_shape
        ))

    # run a projection step with self.num_projections iterations
    def _project(self):
        input_shape = self._input_shape
        print("===Projecting===")
        for i in range(self.num_projections):
            if self.orthogonal:  # project onto orthogonal convolution kernel space
                self.weight.data = torch.from_numpy(
                    conv_clip_2_norm_numpy(
                        self.weight.data.cpu().numpy(), (self.weight.size(-2) * 2 - 1, self.weight.size(-1) * 2 - 1), clip_to=1, force_same=True
                    )
                ).to(device=self.weight.device, dtype=self.weight.dtype)
            else:                # project onto 1-Lipschitz convolution kernel space
                self.weight.data = torch.from_numpy(
                    conv_clip_2_norm_numpy(
                        self.weight.data.cpu().numpy(), input_shape, clip_to=1
                    )
                ).to(device=self.weight.device, dtype=self.weight.dtype)
            self.svs = torch.from_numpy(conv_singular_values_numpy(
                            self.weight.data.cpu().numpy(), input_shape
                        ))
        print("Projection Result: {}".format(self.lipschitz_constant()))
        print("===Finished===")

    def forward(self, x):
        self._input_shape = x.shape[2:]
        self.buffer_weight = self.weight
        return conv2d_cyclic_pad(x, self.weight, self.bias)

    def extra_repr(self):
        return "{}, num_projections={num_projections}".format(
            super().extra_repr(), **self.__dict__
        )