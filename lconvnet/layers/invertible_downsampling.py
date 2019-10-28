"""
Invertible Downsampling is implemented as described in https://arxiv.org/pdf/1802.07088.pdf.
"""
import torch
import torch.nn as nn
from einops import rearrange

class PixelUnshuffle2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def lipschitz_constant(self):
        return 1

    def forward(self, x):
        return rearrange(
            x,
            "b c (w k1) (h k2) -> b (c k1 k2) w h",
            k1=self.kernel_size[0],
            k2=self.kernel_size[1],
        )

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)