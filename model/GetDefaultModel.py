from .Config import loadConfig
from .SimpleCNN_MNIST import trainSimpleCNNWithMNIST
from .AlexNet_CIFAR10 import trainAlexNetWithCIFAR10
from .ResNet18_CelebA import trainResNet18WithCelebA

def get_defualt_model(config_path='./model/config.json'):
    config = loadConfig(config_path)
    if config.dataset == "MNIST":
        model = trainSimpleCNNWithMNIST(get_default_model=True)
    elif config.dataset == "CIFAR10":
        model = trainAlexNetWithCIFAR10(get_default_model=True)
    elif config.dataset == "CelebA":
        model = trainResNet18WithCelebA(data_folder=None, label_file=None, get_default_model=True)
    else:
        raise Exception(f"Unsupported dataset: {config.dataset}")
    return model