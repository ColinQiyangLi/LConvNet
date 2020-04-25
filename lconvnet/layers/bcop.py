"""
BCOP parameterization with block convolution procedure adapted from the official Tensorflow repo:
https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/ops/init_ops.py#L683
"""

from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from lconvnet.utils import StreamlinedModule
from lconvnet.layers.core import LipschitzModuleL2
from lconvnet.layers.invertible_downsampling import PixelUnshuffle2d
from lconvnet.layers.linear import BjorckLinear
from lconvnet.layers.utils import (
    conv_clip_2_norm_numpy,
    conv_singular_values_numpy,
    conv2d_cyclic_pad,
    bjorck_orthonormalize,
)


def orthogonal_matrix(n):
    a = torch.randn((n, n))
    q, r = torch.qr(a)
    return q * torch.sign(torch.diag(r))


def symmetric_projection(n, ortho_matrix, mask=None):
    """Compute a n x n symmetric projection matrix.
    Args:
      n: Dimension.
    Returns:
      A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
    q = ortho_matrix
    # randomly zeroing out some columns
    if mask is None:
        mask = (torch.randn(n) > 0).float()
    c = q * mask
    return c.mm(c.t())


def block_orth(p1, p2):
    """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
    Args:
      p1: A symmetric projection matrix.
      p2: A symmetric projection matrix.
    Returns:
      A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                      [(1-p1)p2, (1-p1)(1-p2)]].
    Raises:
      ValueError: If the dimensions of p1 and p2 are different.
    """
    assert p1.shape == p2.shape
    n = p1.size(0)
    kernel2x2 = {}
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    kernel2x2[0, 0] = p1.mm(p2)
    kernel2x2[0, 1] = p1.mm(eye - p2)
    kernel2x2[1, 0] = (eye - p1).mm(p2)
    kernel2x2[1, 1] = (eye - p1).mm(eye - p2)

    return kernel2x2


def matrix_conv(m1, m2):
    """Matrix convolution.
    Args:
      m1: A k x k dictionary, each element is a n x n matrix.
      m2: A l x l dictionary, each element is a n x n matrix.
    Returns:
      (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """

    n = (m1[0, 0]).size(0)
    if n != (m2[0, 0]).size(0):
        raise ValueError(
            "The entries in matrices m1 and m2 " "must have the same dimensions!"
        )
    k = int(np.sqrt(len(m1)))
    l = int(np.sqrt(len(m2)))
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
        for j in range(size):
            result[i, j] = torch.zeros(
                (n, n), device=m1[0, 0].device, dtype=m1[0, 0].dtype
            )
            for index1 in range(min(k, i + 1)):
                for index2 in range(min(k, j + 1)):
                    if (i - index1) < l and (j - index2) < l:
                        result[i, j] += m1[index1, index2].mm(
                            m2[i - index1, j - index2]
                        )
    return result


def dict_to_tensor(x, k1, k2):
    return torch.stack([torch.stack([x[i, j] for j in range(k2)]) for i in range(k1)])


def convolution_orthogonal_generator_projs(ksize, cin, cout, ortho, sym_projs):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
        ortho = ortho.t()
    if ksize == 1:
        return ortho.unsqueeze(-1).unsqueeze(-1)
    p = block_orth(sym_projs[0], sym_projs[1])
    for _ in range(1, ksize - 1):
        p = matrix_conv(p, block_orth(sym_projs[_ * 2], sym_projs[_ * 2 + 1]))
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = ortho.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)


def convolution_orthogonal_generator(ksize, cin, cout, P, Q):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
    orth = orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
        return orth.unsqueeze(0).unsqueeze(0)

    p = block_orth(symmetric_projection(cout, P[0]), symmetric_projection(cout, Q[0]))
    for _ in range(ksize - 2):
        temp = block_orth(
            symmetric_projection(cout, P[_ + 1]), symmetric_projection(cout, Q[_ + 1])
        )
        p = matrix_conv(p, temp)
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = orth.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)


def convolution_orthogonal_initializer(ksize, cin, cout):
    P, Q = [], []
    cmax = max(cin, cout)
    for i in range(ksize - 1):
        P.append(orthogonal_matrix(cmax))
        Q.append(orthogonal_matrix(cmax))
    P, Q = map(torch.stack, (P, Q))
    return convolution_orthogonal_generator(ksize, cin, cout, P, Q)


class BCOP(StreamlinedModule, LipschitzModuleL2):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        bias=True,
        mask_half=True,       # deprecated, TODO: remove
        projection=False,     # deprecated, TODO: remove
        ortho_mode="bjorck",  # deprecated, TODO: remove
        bjorck_iters=20,
        power_iteration_scaling=True,
        frozen=False,
    ):
        super().__init__()
        assert stride == 1, "BCOP convolution only supports stride 1."
        assert not projection and ortho_mode == "bjorck"  # these options are deprecated
        assert mask_half  # this option is deprecated
        assert padding is None or padding == kernel_size // 2, "BCOP convolution only supports k // 2 padding. actual - {}, required - {}".format(padding, kernel_size // 2)

        self.mask_half = mask_half
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.max_channels = max(self.in_channels, self.out_channels)
        self.num_kernels = 2 * (kernel_size - 1) + 1
        self.ortho_mode = ortho_mode
        self.bjorck_iters = bjorck_iters
        self.power_iteration_scaling = power_iteration_scaling
        self.frozen = frozen

        # Define the unconstrained matrices Ms and Ns for Ps and Qs
        self.param_matrices = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels, self.max_channels),
            requires_grad=not self.frozen,
        )

        # The mask controls the rank of the symmetric projectors (full half rank).
        self.mask = nn.Parameter(
            torch.cat(
                (
                    torch.ones(self.num_kernels - 1, 1, self.max_channels // 2),
                    torch.zeros(
                        self.num_kernels - 1,
                        1,
                        self.max_channels - self.max_channels // 2,
                    ),
                ),
                dim=-1,
            ).float(),
            requires_grad=False,
        )

        # Bias parameters in the convolution
        self.enable_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels), requires_grad=not self.frozen
            )
        else:
            self.bias = None

        # Initialize the weights (self.weight is set to zero for streamline module)
        self.reset_parameters()
        self.weight = None

    def set_streamline(self, streamline=False):
        # Implements interface required by StreamlineModule
        super().set_streamline(streamline=streamline)
        if streamline == True:
            self.weight = None

    def singular_values(self):
        # Implements interface required by LipschitzModuleL2
        svs = torch.from_numpy(
            conv_singular_values_numpy(
                self.buffer_weight.detach().cpu().numpy(), self._input_shape
            )
        ).to(device=self.buffer_weight.device)
        return svs

    def reset_parameters(self):
        ortho_weights = [
            torch.empty(self.max_channels, self.max_channels)
            for i in range(self.num_kernels)
        ]
        stdv = 1.0 / (self.max_channels ** 0.5)
        for index, ortho_weight in enumerate(ortho_weights):
            nn.init.orthogonal_(ortho_weight, gain=stdv)
            self.param_matrices.data[index] = ortho_weight

        std = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, x):
        self._input_shape = x.shape[
            2:
        ]  # cache the input shape for self.singular_values()

        # streamline controls whether the weight from previous steps are being used
        if not self.streamline or self.weight is None:
            # orthognoalize all the matrices using Bjorck
            ortho = bjorck_orthonormalize(
                self.param_matrices,
                iters=self.bjorck_iters,
                power_iteration_scaling=self.power_iteration_scaling,
                default_scaling=not self.power_iteration_scaling,
            )

            # compute the symmetric projectors
            H = ortho[0, : self.in_channels, : self.out_channels]
            PQ = ortho[1:]
            PQ = PQ * self.mask
            PQ = PQ @ PQ.transpose(-1, -2)

            # compute the resulting convolution kernel using block convolutions
            self.weight = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, H, PQ
            )
        self.buffer_weight = self.weight

        # detach the weight when we are using the cached weights from previous steps
        if self.streamline: weight = self.weight.detach()
        else: weight = self.weight

        # apply cyclic padding to the input and perform a standard convolution
        return conv2d_cyclic_pad(x, weight, self.bias)

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, bias={enable_bias}, mask_half={mask_half}, ortho_mode={ortho_mode}".format(
            **self.__dict__
        )

