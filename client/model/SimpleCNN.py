import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from MachineLearningGeneralFunction import trainModel, testModel

def trainSimpleCNNWithMNIST(data_folder='./dataset',input_model_path=None, output_model_path=None, learning_rate=0.001, batch_size=64, epoch=10):
    """
    Train a simple Convolutional Neural Network (CNN) on the MNIST dataset.

    This function defines a basic CNN architecture, trains the model using the MNIST dataset, 
    evaluates its performance, and provides options to load a pre-trained model and save the trained model.

    Parameters:
        input_model_path : str, optional
            Path to the pre-trained model file. If specified, the model will be initialized with these weights (default is None).
        output_model_path : str, optional
            Path to save the trained model. If specified, the trained model will be saved to this location (default is None).
        learning_rate : float, optional
            The learning rate for the Adam optimizer (default is 0.001).
        batch_size : int, optional
            Batch size for training and testing (default is 64).
        epoch : int, optional
            Number of training iterations over the dataset (default is 10).

    Returns:
        dict: A dictionary containing the training accuracy, training loss, test accuracy, and model.
    """
    
    # Define the CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.fc1 = nn.Linear(7 * 7 * 64, 128)
            self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST
            self.dropout = nn.Dropout(0.25)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)  # Pooling layer
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(-1, 7 * 7 * 64)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_folder, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = CNN()
    if input_model_path:
        model.load_state_dict(torch.load(input_model_path, weights_only=True))
        print(f"Loaded pre-trained model from {input_model_path}")
    else:
        print("Using default model initialization.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    result = trainModel(model, train_loader, criterion, optimizer, epoch)
    model = result["model"]
    
    # Test the model
    test_accuracy = testModel(model, test_loader)
    result["test_accuracy"] = test_accuracy

    # Save the trained model
    if output_model_path:
        torch.save(model.state_dict(), output_model_path)
        print(f"Model saved to {output_model_path}")

    return result
