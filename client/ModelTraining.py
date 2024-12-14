import os
import json

from client.Config import loadConfig
from client.model import trainSimpleCNNWithMNIST, trainAlexNetWithCIFAR10, trainResNet18WithCelebA

def modelTraining(config_path:str, input_model=None):
    if not os.path.exists(config_path):
        raise Exception(f"Config file not found: {config_path}")
    
    config = loadConfig(config_path)
    params = config[config.dataset]
    
    if config.dataset == "MNIST":
        result = trainSimpleCNNWithMNIST(data_folder=params.data_folder, input_model=input_model, input_model_path=params.input_model_path, output_model_path=params.output_model_path,
                                         train_num=params.train_num, test_num=params.test_num, learning_rate=params.learning_rate, batch_size=params.batch_size, epoch=params.epochs)
    elif config.dataset == "CIFAR10":
        result = trainAlexNetWithCIFAR10(data_folder=params.data_folder, input_model=input_model, input_model_path=params.input_model_path, output_model_path=params.output_model_path,
                                         train_num=params.train_num, test_num=params.test_num, learning_rate=params.learning_rate, batch_size=params.batch_size, epoch=params.epochs)
    elif config.dataset == "CelebA":
        result = trainResNet18WithCelebA(data_folder=params.data_folder, label_file=params.label_file, input_model=input_model, input_model_path=params.input_model_path, output_model_path=params.output_model_path,
                                         train_num=params.train_num, test_num=params.test_num, learning_rate=params.learning_rate, batch_size=params.batch_size, epoch=params.epochs)
    else:
        raise Exception(f"Unsupported dataset: {config.dataset}")
    
    
if __name__ == "__main__":    
    config_path = "./client/config.json"
    modelTraining(config_path)