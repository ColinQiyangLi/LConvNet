from .common import FirstNDataset

import torchvision
import torchvision.transforms as transforms

import sys

def mnist(mini_test_size=1000, mini_train_size=None):
    print("| Preparing MNIST dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True, transform=transforms.ToTensor()
    )
    if mini_train_size is not None:
        trainset = FirstNDataset(trainset, n=mini_train_size)
    testset = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=False, transform=transforms.ToTensor()
    )
    mini_testset = FirstNDataset(testset, n=mini_test_size)
    return trainset, testset, mini_testset
