import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd

from MachineLearningGeneralFunction import trainModelWithValidation, testModel

class CelebADataset(Dataset):
    '''
    Custom dataset class for CelebA dataset.
    '''
    def __init__(self, data_folder, label_file, partition_file, partition, transform=None):
        '''
        Initialize the CelebA dataset.
        
        Parameters:
            data_folder (str): Path to the folder containing the images.
            label_file (str): Path to the label file.
            partition_file (str): Path to the partition file.
            partition (int): Partition of the dataset (0: training, 1: validation, 2: testing).
            transform (torchvision.transforms): Transformations to apply to the images.
        '''
        self.data_folder = data_folder
        self.labels = pd.read_csv(label_file, sep=r'\s+', skiprows=1)
        self.labels.replace(-1, 0, inplace=True)
        self.partition = pd.read_csv(partition_file, sep=r'\s+', header=None, names=["image_id", "partition"])
        self.image_ids = self.partition[self.partition["partition"] == partition]["image_id"]
        self.transform = transform

    def __len__(self):
        '''
        Get the number of samples in the dataset.
        '''
        return len(self.image_ids)

    def __getitem__(self, idx):
        '''
        Get a sample from the dataset.
        
        Parameters:
            idx (int): Index of the sample.
            
        Returns:
            tuple: Tuple containing the image and label.
        '''
        image_id = self.image_ids.iloc[idx]
        img_path = os.path.join(self.data_folder, image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels.loc[self.labels.index == image_id].values[0][0:], dtype=torch.float32)
        return image, label

def trainResNet18WithCelebA(data_folder, label_file, eval_file, input_model_path=None, 
                            output_model_path=None, learning_rate=0.001, batch_size=32, epoch=10):
    '''
    Train a ResNet18 model on the CelebA dataset.
    
    Parameters:
        data_folder (str): Path to the folder containing the images.
        label_file (str): Path to the label file.
        eval_file (str): Path to the evaluation file.
        input_model_path (str): Path to the pre-trained model file (optional).
        output_model_path (str): Path to save the trained model (optional). Default is None.
        learning_rate (float): Learning rate. Default is 0.001.
        batch_size (int): Batch size. Default is 32.
        epoch (int): Number of training epochs. Default is 10.
        
    Returns:
        dict: A dictionary containing the training accuracy, training loss, validation accuracy, validation loss, test accuracy, and model.
    '''
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = CelebADataset(data_folder, label_file, eval_file, partition=0, transform=transform)
    val_dataset = CelebADataset(data_folder, label_file, eval_file, partition=1, transform=transform)
    test_dataset = CelebADataset(data_folder, label_file, eval_file, partition=2, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load ResNet18 model
    if input_model_path and os.path.exists(input_model_path):
        print(f"Loading pre-trained model from {input_model_path}")
        model = torch.load(input_model_path)
    else:
        print("Using pre-trained ResNet18 model from torchvision")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 40)  # 40 attributes in CelebA

    # Define device, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model with validation
    result = trainModelWithValidation(model, train_loader, val_loader, criterion, optimizer, epoch)
    model = result["model"]
    
    # Test model
    test_accuracy = testModel(model, test_loader)
    result["test_accuracy"] = test_accuracy
    
    # Save trained model
    if output_model_path:
        torch.save(model, output_model_path)
        print(f"Model saved to {output_model_path}")
    
    return result
