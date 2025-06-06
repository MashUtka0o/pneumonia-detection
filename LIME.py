import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

'''
Misclassified images (binary):
./chest_xray_split2/test\NORMAL\1229.jpeg
./chest_xray_split2/test\NORMAL\1485.jpeg
./chest_xray_split2/test\NORMAL\1578.jpeg
./chest_xray_split2/test\NORMAL\769.jpeg
./chest_xray_split2/test\NORMAL\844.jpeg
./chest_xray_split2/test\PNEUMONIA_BACTERIA\1165.jpeg
./chest_xray_split2/test\PNEUMONIA_BACTERIA\131.jpeg
./chest_xray_split2/test\PNEUMONIA_BACTERIA\1701.jpeg
./chest_xray_split2/test\PNEUMONIA_BACTERIA\1825.jpeg
./chest_xray_split2/test\PNEUMONIA_BACTERIA\2299.jpeg
./chest_xray_split2/test\PNEUMONIA_BACTERIA\2425.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\1186.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\1197.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\1414.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\165.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\207.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\210.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\488.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\500.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\639.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\776.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\822.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\824.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\827.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\831.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\864.jpeg
./chest_xray_split2/test\PNEUMONIA_VIRUS\872.jpeg'''

# Load the PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust based on number of classes (assumed 2 for pneumonia)
model.load_state_dict(torch.load('models/resnet18_pneumonia3.pth', map_location=device))
model.to(device)
model.eval()

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

# Function to load and preprocess a single image
def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def preprocess_image(img):
    return transform(img).unsqueeze(0).to(device)

# Prediction function for LIME
def batch_predict(images):
    model.eval()
    batch = torch.stack([transform(Image.fromarray(img)).to(device) for img in images], dim=0)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()

# Load and show image
original_img = load_image(r'Data\chest_xray\train\PNEUMONIA\person843_virus_1485.jpeg')
plt.imshow(original_img)
plt.axis('off')
plt.show()

# Convert to numpy for LIME
img_np = np.array(original_img)

# Run LIME
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img_np,
    batch_predict,
    top_labels=2,
    hide_color=0,
    num_samples=1000
)

# Visualize explanation
label_to_explain = explanation.top_labels[0]

temp, mask = explanation.get_image_and_mask(label_to_explain, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp, mask))
plt.title("Positive only - top 5 features")
plt.axis('off')
plt.show()

# Optional: More detailed heatmap
ind = explanation.top_labels[0]
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
plt.colorbar()
plt.title(f"Heatmap for IMG")
plt.axis('off')
plt.show()
plt.savefig("1485.jpeg")
