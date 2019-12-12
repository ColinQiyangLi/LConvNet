from collections import defaultdict
from functools import reduce
import torch
import torch.nn as nn
from copy import deepcopy
import sys
import math


def default_collate_op(x, y):
    if x is None:
        return [y]
    if y is None:  # avoid appending None and nan
        return x
    if type(y) == list:
        x.extend(y)
    else:
        x.append(y)
    return x


def default_summarize_op(x, dtype):
    if dtype == "scalar":
        if len(x) == 0:
            return 0
        return sum(x) / len(x)
    if dtype == "histogram":
        return torch.tensor(x)
    return x


def default_display_op(x, dtype):
    if dtype == "scalar":
        return "{:.4f}".format(x)
    if dtype == "histogram":
        return "histogram[n={}]".format(len(x))
    return x


def prod(x):
    return reduce(lambda a, b: a * b, x)


class StreamlinedModule(nn.Module):
    def __init__(self):
        self.streamline = False
        super(StreamlinedModule, self).__init__()

    def set_streamline(self, streamline=False):
        self.streamline = streamline
        return streamline


def streamline_model(model, streamline=False):
    for m in model.modules():
        if isinstance(m, StreamlinedModule):
            m.set_streamline(streamline)


# Context manager that streamlines the module of interest in the context only
class Streamline:
    def __init__(self, module, new_flag=True, old_flag=False):
        self.module = module
        self.new_flag = new_flag
        self.old_flag = old_flag

    def __enter__(self):
        streamline_model(self.module, self.new_flag)

    def __exit__(self, *args, **kwargs):
        streamline_model(self.module, self.old_flag)


# A helper object for logging all the data
class Accumulator:
    def __init__(self):
        self.data = defaultdict(list)
        self.data_dtype = defaultdict(None)

    def __call__(
        self,
        name,
        value=None,
        dtype=None,
        collate_op=default_collate_op,
        summarize_op=None,
    ):
        if value is None:
            if summarize_op is not None:
                return summarize_op(self.data[name])
            return self.data[name]
        self.data[name] = default_collate_op(self.data[name], value)
        if dtype is not None:
            self.data_dtype[name] = dtype
        assert dtype == self.data_dtype[name]

    def summarize(self, summarize_op=default_summarize_op):
        for key in self.data:
            self.data[key] = summarize_op(self.data[key], self.data_dtype[key])

    def collect(self):
        return {key: self.__call__(key) for key in self.data}

    def filter(self, dtype=None, level=None, op=None):
        if op is None:
            op = lambda x: x
        if dtype is None:
            return self.collect()
        return {
            key: op(self.__call__(key))
            for key in filter(
                lambda x: self.data_dtype[x] == dtype
                and (x.count("/") <= level if (level is not None) else True),
                self.data,
            )
        }

    def latest_str(self):
        return ", ".join(
            "{}={:.4f}".format(key, value[-1] if len(value) > 0 else math.nan)
            for key, value in self.collect().items()
        )

    def summary_str(self, dtype=None, level=None):
        return ", ".join(
            "{}={}".format(
                key, default_display_op(self.__call__(key), self.data_dtype[key])
            )
            for key in self.filter(dtype=dtype, level=level)
        )

    def __str__(self):
        return self.summary_str()


# A logger that sync terminal output to a logger file
class Logger(object):
    def __init__(self, logdir):
        self.terminal = sys.stdout
        self.log = open(logdir, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
