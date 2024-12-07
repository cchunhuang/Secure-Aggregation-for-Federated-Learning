import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from MachineLearningGeneralFunction import trainModel, testModel

def trainAlexNetWithCIFAR10(data_folder='./dataset', input_model_path=None, output_model_path=None, learning_rate=0.001, batch_size=64, epoch=20):
    """
    Train an AlexNet model on the CIFAR-10 dataset.

    Parameters:
        input_model_path (str): Path to the pre-trained model file (optional).
        output_model_path (str): Path to save the trained model (optional). Default is None.
        learning_rate (float): Learning rate. Default is 0.001.
        batch_size (int): Batch size. Default is 64.
        epoch (int): Number of training epochs. Default is 20.
        
    Returns:
        dict: A dictionary containing the training accuracy, training loss, test accuracy, and model.
    """

    # Data preprocessing for CIFAR-10
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # Adjust size for AlexNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=data_folder, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root=data_folder, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load AlexNet model
    if input_model_path:
        model = torch.load(input_model_path)  # Load custom pre-trained model
        print(f"Loaded pre-trained model from {input_model_path}")
    else:
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)  # Use ImageNet pre-trained model
        print("Using default pre-trained AlexNet model.")
    
    # Modify the classifier to match CIFAR-10 (10 classes)
    model.classifier[6] = nn.Linear(4096, 10)

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
