import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights
from sklearn.metrics import confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = nn.Linear(num_ftrs, 3)
loaded_model.load_state_dict(torch.load("./models/resnet18_pneumonia2.pth", map_location=device))
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
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('Confusion Matrix:' + '\n' + str(cm))
    return accuracy, cm

data_dir = './chest_xray_split' 
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset  = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
accuracy, cm = evaluate_model(loaded_model, test_loader)
import matplotlib.pyplot as plt

# Define class names explicitly
class_names = ['Normal', 'Pneumonia Bacteria', 'Pneumonia Virus']

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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