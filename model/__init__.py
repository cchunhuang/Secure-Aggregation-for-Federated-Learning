from .SimpleCNN_MNIST import trainSimpleCNNWithMNIST
from .AlexNet_CIFAR10 import trainAlexNetWithCIFAR10
from .ResNet18_CelebA import trainResNet18WithCelebA
from .MachineLearningGeneralFunction import trainModel, trainModelWithValidation, testModel

__all__ = ['trainSimpleCNNWithMNIST', 'trainAlexNetWithCIFAR10', 'trainResNet18WithCelebA', 'trainModel', 'trainModelWithValidation', 'testModel']