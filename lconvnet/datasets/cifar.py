import torchfile
from lconvnet.datasets.common import FirstNDataset

import torchvision
import torchvision.transforms as transforms

import torch.utils.data as data
from PIL import Image

import numpy as np
import random

import sys
import os

def cifar_dataset(name, no_scaling=False, mini_test_size=1000, mini_train_size=None):
    transform_lst_test = [transforms.ToTensor()]
    assert no_scaling and name == "cifar10"
    transform_lst_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    transform_train = transforms.Compose(transform_lst_train)  # meanstd transformation
    transform_test = transforms.Compose(transform_lst_test)

    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True,
        transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=False, download=False,
        transform=transform_test)
    mini_testset = FirstNDataset(testset, n=mini_test_size)
    if mini_train_size is not None:
        trainset = FirstNDataset(trainset, n=mini_train_size)
    return trainset, testset, mini_testset

def cifar10(no_scaling=False, mini_test_size=1000, mini_train_size=None):
    return cifar_dataset("cifar10", no_scaling=no_scaling, mini_test_size=mini_test_size, mini_train_size=mini_train_size)
