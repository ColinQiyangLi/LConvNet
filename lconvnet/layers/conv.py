"""
A wrapper for LipschitzConv2d. It uses invertible downsampling to mimic striding.
"""
import torch.nn as nn
from lconvnet.layers.invertible_downsampling import PixelUnshuffle2d


class LipschitzConv2d(nn.Module):
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
        conv_module=None,
    ):
        super().__init__()
        self.enable_bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # compute the in_channels based on stride
        # the invertible downsampling is applied before the convolution
        self.true_in_channels = in_channels * stride * stride
        self.true_out_channels = out_channels
        self.true_stride = 1
        assert kernel_size % stride == 0
        # compute the kernel size of the actual convolution based on stride
        self.true_kernel_size = kernel_size // stride
        self.shuffle = PixelUnshuffle2d(stride)
        self.conv = conv_module(
            self.true_in_channels,
            self.true_out_channels,
            kernel_size=self.true_kernel_size,
            stride=1,
            padding=self.true_kernel_size // 2,
            bias=bias,
        )
        
    def forward(self, x):
        x = self.shuffle(x)
        x = self.conv(x)
        return x

    def extra_repr(self):
        return "{in_channels} ({true_in_channels}), {out_channels} ({true_out_channels}), kernel_size={kernel_size} ({true_kernel_size}), stride={stride}, bias={enable_bias}".format(
            **self.__dict__
        )
