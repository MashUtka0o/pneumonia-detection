# Class: Deep Learning: Pneumonia Classification
# This Script Trains a ResNet18 model on the Chest X-Ray dataset to classify pneumonia types.
# This script assumes you have already preprocessed the dataset and split it into train, validation, and test sets.

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop the image to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset Directory in the format:
# datasetdirectory
# ├── train
# │   ├── NORMAL
# │   ├── PNEUMONIA_BACTERIA
# │   └── PNEUMONIA_VIRUS
# ...
data_dir = './chest_xray_split2'

# Load datasets
train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=validation_transform)
test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=validation_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

"""
Main Function
Training Parameters:
    Epochs: 20
    Batch Size: 32
    Learning Rate: 1e-4
    Weight Decay: 1e-4
    Optimizer: Adam
    Scheduler: StepLR with step size 7 and gamma 0.1
"""

if __name__ == "__main__":
    # Load pre-trained ResNet18 model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    criterion = nn.CrossEntropyLoss()
    epoch = 20

    # Replace final FC layer (3 output classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    # ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS']

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = model.to(device)

    for epoch in range(epoch):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / total_val

        scheduler.step()

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Testing
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

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "./models/resnet50_pneumonia3.pth")
    print("Model saved as resnet50_pneumonia3.pth")

