import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score

class CelebADataset(Dataset):
    def __init__(self, data_folder, label_file, partition_file, partition, transform=None):
        self.data_folder = data_folder
        self.labels = pd.read_csv(label_file, sep=r'\s+', skiprows=1)
        self.labels.replace(-1, 0, inplace=True)
        self.partition = pd.read_csv(partition_file, sep=r'\s+', header=None, names=["image_id", "partition"])
        self.image_ids = self.partition[self.partition["partition"] == partition]["image_id"]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids.iloc[idx]
        img_path = os.path.join(self.data_folder, image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels.loc[self.labels.index == image_id].values[0][0:], dtype=torch.float32)
        return image, label

def trainResNet18WithCelebA(data_folder, label_file, eval_file, input_model_path=None, 
                            output_model_path=None, learning_rate=0.001, batch_size=32, epoch=10):
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

    # Training loop
    for e in range(epoch):
        print(f"Training epoch {e+1}/{epoch}")
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch [{e+1}/{epoch}], Training Loss: {train_loss/len(train_loader):.4f}")

        # Validation
        print(f"Validation epoch {e+1}/{epoch}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        print(f"Epoch [{e+1}/{epoch}], Validation Loss: {val_loss/len(val_loader):.4f}")

    # Save trained model
    if output_model_path:
        torch.save(model, output_model_path)
        print(f"Model saved to {output_model_path}")

    # Test model
    print("Testing model")   
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            all_preds.append((outputs.cpu() > 0.5).int())
            all_labels.append(labels.cpu().int())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")

    return model
