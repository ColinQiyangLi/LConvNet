"""
GroupSort implementation is directly taken from https://github.com/cemanil/LNets
"""

import numpy as np
import torch
import torch.nn as nn


class GroupSort(nn.Module):
    def __init__(self, group_size, axis=-1, new_impl=False):
        super(GroupSort, self).__init__()
        self.group_size = group_size
        self.axis = axis
        self.new_impl = new_impl

    def lipschitz_constant(self):
        return 1

    def forward(self, x):
        group_sorted = group_sort(x, self.group_size, self.axis, self.new_impl)
        return group_sorted

    def extra_repr(self):
        return "group_size={group_size}, axis={axis}".format(**self.__dict__)

def group_sort(x, group_size, axis=-1, new_impl=False):
    if new_impl and group_size == 2:
        a, b = x.split(x.size(axis) // 2, axis)
        a, b = torch.max(a, b), torch.min(a, b)
        return torch.cat([a, b], dim=axis)
    shape = list(x.shape)
    num_channels = shape[axis]
    assert num_channels % group_size == 0
    shape[axis] = num_channels // group_size
    shape.insert(axis, group_size)
    if axis < 0:
        axis -= 1
    assert shape[axis] == group_size
    return x.view(*shape).sort(dim=axis)[0].view(*x.shape)
