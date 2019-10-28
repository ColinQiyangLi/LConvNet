"""
BjorckLinear is adapted from https://github.com/cemanil/LNets.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

from lconvnet.utils import StreamlinedModule
from lconvnet.layers.core import LipschitzModuleL2
from lconvnet.layers.utils import power_iteration, bjorck_orthonormalize

class BjorckLinear(nn.Linear, StreamlinedModule, LipschitzModuleL2):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bjorck_beta=0.5,
        bjorck_iters=20,
        bjorck_order=1,
        bjorck_iters_scheduler=None,
        power_iteration_scaling=True,
    ):
        self.enable_bias = bias
        super().__init__(in_features, out_features, bias=bias)
        self.bjorck_beta = bjorck_beta
        self.bjorck_iters = bjorck_iters
        self.bjorck_order = bjorck_order
        self.bjorck_weight = None
        self.bjorck_iters_scheduler = bjorck_iters_scheduler
        self.power_iteration_scaling = power_iteration_scaling

    def singular_values(self):
        svs = torch.svd(self.bjorck_weight.detach())[1]
        return svs

    def reset_parameters(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def ortho_w(self):
        return bjorck_orthonormalize(
            self.weight,
            beta=self.bjorck_beta,
            iters=self.bjorck_iters,
            order=self.bjorck_order,
            power_iteration_scaling=self.power_iteration_scaling,
            default_scaling=not self.power_iteration_scaling,
        )

    def set_streamline(self, streamline=False):
        super().set_streamline(streamline=streamline)
        if streamline == True:
            self.bjorck_weight = None

    def forward(self, x):
        if not self.streamline:
            if self.bjorck_iters_scheduler is not None:
                self.bjorck_iters_scheduler.update()
                self.bjorck_iters = self.bjorck_iters_scheduler.get()
            self.bjorck_weight = self.ortho_w()
        else:
            if self.bjorck_weight is None:
                with torch.no_grad():
                    self.bjorck_weight = self.ortho_w()
        return F.linear(x, self.bjorck_weight, self.bias)

    def extra_repr(self):
        return "{in_features}, {out_features}, bias={enable_bias}, bjorck_beta={bjorck_beta}, bjorck_iters={bjorck_iters}, bjorck_order={bjorck_order}".format(
            **self.__dict__
        )


if __name__ == "__main__":
    pass
