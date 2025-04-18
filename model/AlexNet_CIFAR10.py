import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from .MachineLearningGeneralFunction import trainModel, testModel

def trainAlexNetWithCIFAR10(data_folder='./dataset', input_model=None, input_model_path=None, output_model_path=None, 
                            train_num=None, test_num=None, learning_rate=None, batch_size=None, epoch=None, get_default_model=False):
    """
    Train an AlexNet model on the CIFAR-10 dataset.

    Parameters:
        data_folder (str): Path to the folder containing the dataset.
        input_model (nn.Module): Pre-trained model to use for training (optional).
        input_model_path (str): Path to the pre-trained model file (optional).
        output_model_path (str): Path to save the trained model (optional). Default is None.
        train_num (int): Number of training samples to use (optional). Default is None (use all samples).
        test_num (int): Number of testing samples to use (optional). Default is None (use all samples).
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        batch_size (int): Batch size for training. Default is 64.
        epoch (int): Number of training epochs. Default is 10.
        get_default_model (bool): If True, return the default model without training. Default is False.
        
    Returns:
        dict: A dictionary containing the training accuracy, training loss, test accuracy, and model.
    """
    
    if get_default_model:
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, 10) # Modify the classifier to match CIFAR-10 (10 classes)
        return model
    
    if learning_rate == None:
        learning_rate = 0.001
    if batch_size == None:
        batch_size = 64
    if epoch == None:
        epoch = 10

    # Data preprocessing for CIFAR-10
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # Adjust size for AlexNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=data_folder, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_folder, train=False, download=True, transform=transform)
    
    if train_num != None:
        train_indices = np.random.choice(len(train_dataset), train_num, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    if test_num != None:
        test_indices = np.random.choice(len(test_dataset), test_num, replace=False)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load AlexNet model
    if input_model:
        model = input_model
        print("Using customed pre-trained model.")
    elif input_model_path:
        model = torch.load(input_model_path)  # Load custom pre-trained model
        print(f"Loaded pre-trained model from {input_model_path}")
    else:
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)  # Use ImageNet pre-trained model
        model.classifier[6] = nn.Linear(4096, 10) # Modify the classifier to match CIFAR-10 (10 classes)
        print("Using default pre-trained AlexNet model.")
    
    

    # Set device and define loss and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    result = trainModel(model, train_loader, criterion, optimizer, epoch)
    model = result["model"]
    
    # Test the model
    test_accuracy = testModel(model, test_loader)
    result["test_accuracy"] = test_accuracy
    
    # Save the model
    if output_model_path:
        torch.save(model, output_model_path)
        print(f"Model saved to {output_model_path}")
        
    return result
