import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

class FirstNDataset(Dataset):
    def __init__(self, source_dataset, n):
        self.source_dataset = source_dataset
        self.n = n
        assert n <= len(self.source_dataset)

    def __getitem__(self, index):
        assert index < self.n
        return self.source_dataset[index]

    def __len__(self):
        return self.n

def default_dataloader(dataset, train_batch_size, test_batch_size):
    trainset, testset, mini_testset = dataset
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=0
    )
    mini_testloader = torch.utils.data.DataLoader(
        mini_testset, batch_size=test_batch_size, shuffle=False, num_workers=0
    )
    return trainloader, testloader, mini_testloader
