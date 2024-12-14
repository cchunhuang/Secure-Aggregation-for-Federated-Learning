import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

from .MachineLearningGeneralFunction import trainModel, testModel

# Function to train ResNet18 model on the CelebA dataset
def trainResNet18WithCelebA(data_folder, label_file, input_model=None, input_model_path=None, output_model_path=None, 
                            train_num=None, test_num=None, learning_rate=0.001, batch_size=32, epoch=10):
    """
    Train a ResNet18 model on the CelebA dataset.

    Parameters:
        data_folder (str): Path to the folder containing the images.
        label_file (str): Path to the label file.
        input_model (torch.nn.Module): Pre-loaded model (optional).
        input_model_path (str): Path to the pre-trained model file (optional).
        output_model_path (str): Path to save the trained model (optional).
        train_num (int): Number of training samples.
        test_num (int): Number of testing samples.
        learning_rate (float): Learning rate. Default is 0.001.
        batch_size (int): Batch size. Default is 32.
        epoch (int): Number of training epochs. Default is 10.

    Returns:
        dict: A dictionary containing the training accuracy, training loss, test accuracy, and model.
    """

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the label file
    labels_df = pd.read_csv(label_file, sep=r'\s+', skiprows=2, header=None)
    labels_df.columns = ['image_id'] + [f'attr_{i}' for i in range(1, labels_df.shape[1])]
    labels_df.replace(-1, 0, inplace=True)
    
    # Shuffle and split into train and test
    all_files = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset
    train_files = all_files.iloc[:train_num]
    test_files = all_files.iloc[train_num:train_num + test_num]

    # Custom Dataset Class
    class CelebADataset(Dataset):
        """
        Custom dataset class for CelebA images.
        """
        def __init__(self, data_folder, label, transform=None):
            self.data_folder = data_folder
            self.label = label
            self.transform = transform

        def __len__(self):
            return len(self.label)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_folder, self.label.iloc[idx]['image_id'])
            image = Image.open(img_name).convert("RGB")
            label = torch.tensor(self.label.iloc[idx, 1:].astype(float).values, dtype=torch.float32)

            if self.transform:
                image = self.transform(image)

            return image, label

    # Create datasets and dataloaders
    train_dataset = CelebADataset(data_folder, train_files, transform)
    test_dataset = CelebADataset(data_folder, test_files, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load ResNet18 model
    if input_model:
        model = input_model
        print("Using custom pre-trained model.")
    elif input_model_path:
        model = torch.load(input_model_path)
        print(f"Loaded pre-trained model from {input_model_path}")
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 40)  # Output size matches number of attributes
        print("Using default pre-trained ResNet18 model.")

    # Define device, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    result = trainModel(model, train_loader, criterion, optimizer, epoch)
    model = result["model"]
    
    # Test model
    test_accuracy = testModel(model, test_loader)
    result["test_accuracy"] = test_accuracy

    # Save the model
    if output_model_path:
        torch.save(model, output_model_path)
        print(f"Model saved to {output_model_path}")

    return result
