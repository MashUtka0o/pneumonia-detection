import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = nn.Linear(num_ftrs, 3)
loaded_model.load_state_dict(torch.load("./models/resnet18_pneumonia2.pth", map_location=device, weights_only=True))
loaded_model = loaded_model.to(device)
loaded_model.eval()
print("Model loaded from resnet18_pneumonia2.pth")

# use the model on the test set, show the accuracy and the confusion matrix
def evaluate_model(model, test_loader):
    model.eval()
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

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate binary accuracy for pneumonia detection (Normal vs Pneumonia)
    binary_labels = [0 if l == 0 else 1 for l in all_labels]
    binary_preds = [0 if p == 0 else 1 for p in all_preds]
    pneumonia_accuracy = np.mean(np.array(binary_labels) == np.array(binary_preds))

    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, pneumonia_accuracy, cm, f1

data_dir = './chest_xray_split' 
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset  = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
accuracies = []
pneumonia_accuracies = []
cms = []
f1_scores = []

for i in range(1000):
    print(i)
    accuracy, pneumomnia_accuracy, cm, f1 = evaluate_model(loaded_model, test_loader)
    accuracies.append(accuracy)
    pneumonia_accuracies.append(pneumomnia_accuracy)
    cms.append(cm)
    f1_scores.append(f1)

avg_pneumomnia_accuracy = np.mean(pneumonia_accuracies)
avg_accuracy = np.mean(accuracies)
avg_cm = np.mean(cms, axis=0)
avg_f1_score = np.mean(f1_scores)

print(f'Average Accuracy over 1000 runs: {avg_accuracy * 100:.2f}%')
print('Average Confusion Matrix:\n', avg_cm.astype(int))
print(f'Average Pneumonia Detection Accuracy over 1000 runs: {avg_pneumomnia_accuracy * 100:.2f}%')
print(f'Average F1 Score over 1000 runs: {avg_f1_score:.4f}')

# Define class names explicitly
class_names = ['Normal', 'Pneumonia Bacteria', 'Pneumonia Virus']

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(avg_cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix'
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