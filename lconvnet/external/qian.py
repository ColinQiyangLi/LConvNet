from lconvnet.external.qian_ext.script_cifar10 import cifar10_qian_model
from lconvnet.external.qian_ext.script_mnist import mnist_qian_model

def qian(dataset_name, model_index=3):
    assert model_index in [3, 4]
    assert dataset_name in ["cifar10", "mnist"]
    if dataset_name == "mnist":
        return mnist_qian_model(model_index, normalized=True)
    return cifar10_qian_model(model_index, normalized=True)
