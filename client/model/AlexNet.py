import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

def trainAlexNetWithCIFAR10(data_folder='./dataset', input_model_path=None, output_model_path=None, learning_rate=0.001, batch_size=64, epoch=10):
    """
    Train an AlexNet model on the CIFAR-10 dataset.

    Parameters:
        input_model_path (str): Path to the pre-trained model file (optional).
        output_model_path (str): Path to save the trained model (optional). Default is None.
        learning_rate (float): Learning rate. Default is 0.001.
        batch_size (int): Batch size. Default is 64.
        epoch (int): Number of training epochs. Default is 10.
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

    # Training function
    def train_model(model, train_loader, criterion, optimizer, epochs):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation function
    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    # Train and evaluate
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, epoch)
    print("Testing model...")
    evaluate_model(model, test_loader)
    
    # Save the model
    if output_model_path:
        torch.save(model, output_model_path)
        print(f"Model saved to {output_model_path}")
        
    return model
