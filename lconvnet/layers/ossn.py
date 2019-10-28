"""
A reimplementation of spectral normalization for convolution that is inspired by
the official pytorch implementation of spectral normalization on linear layers:
https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

from lconvnet.layers.utils import conv2d_cyclic_pad, conv_singular_values_numpy
from lconvnet.layers.core import LipschitzModuleL2

class SpectralNorm:
    def __init__(self, coeff=1.0, n_iters=1, one_sided=False, during_forward_pass=False, eps=1e-12, run_forward=True, through_scaling=False):
        self.coeff = coeff
        self.n_iters = n_iters
        self.one_sided = one_sided
        self.during_forward_pass = during_forward_pass
        self.eps = eps
        self.weight_buff = None
        self.run_forward = run_forward
        self.through_scaling = through_scaling

    @property
    def data_dim(self):
        raise NotImplementedError

    def _normal_forward(self, x, w):
        raise NotImplementedError

    def _transpose_forward(self, x, w):
        raise NotImplementedError

    def _project_weight(self, w, factor):
        return w / factor

    def _initialize(self, x):
        with torch.no_grad():
            u = self._normal_forward(x, self.weight)
        u = torch.randn(*u.shape, device=x.device, dtype=x.dtype)
        self.register_buffer("u", u)
        
    def _power_iteration(self, w):
        u = self.u.clone()

        # power iteration using the stored vector u
        with torch.no_grad():
            for _ in range(self.n_iters):
                vs = self._transpose_forward(u, w)
                v = normalize(vs.contiguous().view(-1), dim=0,
                            eps=self.eps).view(vs.shape)
                us = self._normal_forward(v, w)
                u = normalize(us.view(-1), dim=0, eps=self.eps).view(us.shape)

        # compute an estimate for the maximum singular value sigma
        weight_v = self._normal_forward(v.detach(), w)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.detach().view(-1), weight_v)

        # compute the scaling factor
        factor = (sigma / self.coeff)
        if self.one_sided:
            factor = factor.clamp(min=1.0)
        else:
            factor = factor.clamp(min=1e-5)  # for stability
        self.u.data = u
        self.factor.data = factor.data
        if self.through_scaling:  # allowing the gradient to follow through the scaling factor
            return factor
        return factor.detach()

    def forward(self, x):
        # initialize the scaling factor and its corresponding vector
        if not hasattr(self, "factor"):
            factor = torch.tensor(1.0)
            self.register_buffer("factor", factor)
        if not hasattr(self, "u"):
            assert x.dim() == self.data_dim or x.dim() == self.data_dim + 1
            if x.dim() == self.data_dim + 1:
                self._initialize(x[0])
            else:
                self._initialize(x)

        # projection during the forward pass or explicit projection (PGD style)
        if self.during_forward_pass:
            if self.training: # run power iteration only during training
                factor = self._power_iteration(self.weight)
            else:
                factor = self.factor
            weight = self._project_weight(self.weight, factor)
            self.weight_buff = weight
        else:
            with torch.no_grad():
                self._power_iteration(self.weight)
            self.weight.data = self._project_weight(self.weight, self.factor)
            self.weight_buff = self.weight
        return self._forward(x, self.weight_buff)

class SpectralNormLinear(LipschitzModuleL2, SpectralNorm, nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 coeff=1.0,
                 n_iters=1,
                 one_sided=False,
                 during_forward_pass=False,
                 eps=1e-12,
                 run_forward=True
                 ):
        self.enable_bias = bias
        SpectralNorm.__init__(self,
                              coeff=coeff,
                              n_iters=n_iters,
                              one_sided=one_sided,
                              during_forward_pass=during_forward_pass,
                              eps=eps,
                              run_forward=run_forward,
                              )
        nn.Linear.__init__(self,
                           in_features=in_features,
                           out_features=out_features,
                           bias=bias
                           )

    def singular_values(self):
        return torch.svd(self.weight_buff)[1]

    @property
    def data_dim(self):
        return 1

    def _forward(self, x, w):
        return F.linear(x, w, self.bias)

    def _normal_forward(self, x, w):
        return F.linear(x, w)

    def _transpose_forward(self, x, w):
        return F.linear(x, w.t())

    def extra_repr(self):
        return "{in_features}, {out_features}, bias={enable_bias}, n_pow_iters={n_iters}, one_sided={one_sided}, during_forward_pass={during_forward_pass}".format(
            **self.__dict__
        )


class SpectralNormConv2d(LipschitzModuleL2, SpectralNorm, nn.Conv2d):
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
        coeff=1.0,
        n_iters=1,
        one_sided=False,
        during_forward_pass=False,
        eps=1e-12,
        run_forward=True,
        through_scaling=True,
    ):
        assert stride == 1
        assert padding == kernel_size // 2
        self.enable_bias = bias
        SpectralNorm.__init__(self,
                              coeff=coeff,
                              n_iters=n_iters,
                              one_sided=one_sided,
                              during_forward_pass=during_forward_pass,
                              eps=eps,
                              run_forward=run_forward,
                              through_scaling=through_scaling,
                              )
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

    @property
    def data_dim(self):
        return 3

    def singular_values(self):
        svs = torch.from_numpy(conv_singular_values_numpy(self.weight_buff.detach().cpu().numpy(), self._input_shape)).to(device=self.weight_buff.device)
        return svs

    # define the forward affine operator (with bias)
    def _forward(self, x, w):
        self._input_shape = x.shape[-2:]
        return conv2d_cyclic_pad(
            x, w, self.bias
        )

    # define the transpose linear operator (used in power iteration)
    def _transpose_forward(self, x, w):
        return conv2d_cyclic_pad(
            x, w.flip(
                [2, 3]).transpose(1, 0)
        )

    # define the forward linear operator (used in power iteration)
    def _normal_forward(self, x, w):
        return conv2d_cyclic_pad(
            x, w
        )

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, bias={enable_bias}, n_pow_iters={n_iters}, one_sided={one_sided}, during_forward_pass={during_forward_pass}".format(
            **self.__dict__
        )

class OSSN(SpectralNormConv2d):
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
        coeff=1.0,
        n_iters=1,
        during_forward_pass=True,
        eps=1e-12,
        run_forward=True,
        through_scaling=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            coeff,
            n_iters,
            one_sided=True,
            during_forward_pass=during_forward_pass,
            eps=eps,
            run_forward=run_forward,
            through_scaling=through_scaling,
        )
