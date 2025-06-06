"""
Tests the trained model on the test set and prints relevant metrics
Calculates only 2-class confusion matrix (Normal vs Pneumonia)
"""

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
loaded_model = models.resnet18(weights=None)
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = nn.Linear(num_ftrs, 3)
loaded_model.load_state_dict(torch.load("./models/resnet18_pneumonia3.pth", map_location=device, weights_only=True))
loaded_model = loaded_model.to(device)
loaded_model.eval()
print("Model loaded from resnet18_pneumonia3.pth")

# Use the model on the test set, show the accuracy and the confusion matrix
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate binary accuracy for pneumonia detection (Normal vs Pneumonia)
    accuracy = correct / total

    print(accuracy)

    binary_labels = [0 if l == 0 else 1 for l in all_labels]
    binary_preds = [0 if p == 0 else 1 for p in all_preds]
    pneumonia_accuracy = np.mean(np.array(binary_labels) == np.array(binary_preds))

    cm = confusion_matrix(all_labels, all_preds)
    print(f'Accuracy: {pneumonia_accuracy * 100:.2f}%')
    print(f'Pneumonia Detection Accuracy: {pneumonia_accuracy * 100:.2f}%')
    print('Confusion Matrix:\n', cm)
    # Compute and print 2-class confusion matrix (Normal vs Pneumonia)
    binary_cm = confusion_matrix(binary_labels, binary_preds)
    # Calculate metrics for binary classification (Normal vs Pneumonia)
    print('Binary Confusion Matrix:\n', binary_cm)
    TP = binary_cm[1, 1]
    TN = binary_cm[0, 0]
    FP = binary_cm[0, 1]
    FN = binary_cm[1, 0]

    # Avoid division by zero
    epsilon = 1e-7

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    f1score = 2 * precision * recall / (precision + recall + epsilon)
    mcc = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + epsilon)

    print(f'Binary Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall (Sensitivity): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'F1 Score: {f1score:.4f}')
    print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')

    return accuracy, precision, recall, specificity, f1score, mcc, binary_cm

data_dir = './chest_xray_split2' 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset  = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

accuracy, precision, recall, specificity, f1score, mcc, cm = evaluate_model(loaded_model, test_loader)

# Define class names for binary confusion matrix
binary_class_names = ['Normal', 'Pneumonia']

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(binary_class_names)),
    yticks=np.arange(len(binary_class_names)),
    xticklabels=binary_class_names,
    yticklabels=binary_class_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Binary Confusion Matrix'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.show()
