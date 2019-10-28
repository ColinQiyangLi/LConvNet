import os 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def repeated_loaders(train_batch_size, test_batch_size, loader_init):
    return loader_init(
        batch_size=train_batch_size), loader_init(
        batch_size=test_batch_size), loader_init(
        batch_size=test_batch_size)

class DistributionLoader(DataLoader):
    def __init__(self, dist1, dist2, batch_size=1, n_examples_per_epoch=100):
        self.dist1, self.dist2 = dist1, dist2
        self.batch_size = batch_size
        self.n_examples_per_epoch = n_examples_per_epoch

    def __iter__(self):
        for i in range(self.n_examples_per_epoch):
            yield self.dist1(self.batch_size), self.dist2(self.batch_size)

    def __len__(self):
        return self.n_examples_per_epoch


class DistributionGANLoader(DistributionLoader):
    def __init__(self, gan_sampler_module, batch_size=1, n_examples_per_epoch=100):
        super().__init__(
            gan_sampler_module(generate_type="generated"),
            gan_sampler_module(generate_type="real"),
            batch_size=batch_size,
            n_examples_per_epoch=n_examples_per_epoch,
        )
