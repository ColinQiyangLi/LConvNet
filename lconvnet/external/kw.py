import os
import torch
import torch.nn as nn
from lconvnet.external.kw_ext.examples.problems import cifar_model, cifar_model_large, cifar_model_resnet, mnist_model, mnist_model_large

model_paths = {
    ("cifar10", "small"): ("cifar_small_36px.pth", cifar_model),
    ("cifar10", "large"): ("cifar_large_36px.pth", cifar_model_large),
    ("cifar10", "resnet"): ("ciar_resnet_36px.pth", cifar_model_resnet),
    ("mnist", "small"): ("mnist_small.pth", mnist_model),
    ("mnist", "large"): ("mnist_large.pth", mnist_model_large)
}

class WrapperMNIST(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def preprocess(self, x):
        return x

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

    def __getitem__(self, item):
        return self.model[item]


class WrapperCIFAR10(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.225, 0.225, 0.225)

    def preprocess(self, x):
        x = torch.stack([
            (x[:, 0] - self.mean[0]) / self.std[0],
            (x[:, 1] - self.mean[1]) / self.std[1],
            (x[:, 2] - self.mean[2]) / self.std[2], ], dim=1)
        return x


    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

    def __getitem__(self, item):
        return self.model[item]


def kw(dataset_name, model_name, index=0):
    path = "lconvnet/external/kw_ext/models_scaled_l2"
    assert (dataset_name, model_name) in model_paths
    suffix, model_loader = model_paths[(dataset_name, model_name)]
    model_path = os.path.join(path, suffix)
    model = model_loader()
    s = torch.load(model_path)["state_dict"][index]
    model.load_state_dict(s)

    if dataset_name == "cifar10":
        return WrapperCIFAR10(model)
    if dataset_name == "mnist":
        return WrapperMNIST(model)
