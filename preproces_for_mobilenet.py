import os
import numpy as np
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class_names = ['NORMAL', 'PNEUMONIA']

def mix_data():
    original_dir = 'chest_xray'
    target_dir = 'mixed'

    for category in class_names:
        os.makedirs(os.path.join(target_dir, category), exist_ok=True)

    splits = ['train', 'test', 'val']
    for split in splits:
        for category in class_names:
            src_folder = os.path.join(original_dir, split, category)
            dst_folder = os.path.join(target_dir, category)
            for filename in os.listdir(src_folder):
                if filename.endswith('.DS_Store'):
                    continue
                src_file = os.path.join(src_folder, filename)
                dst_file = os.path.join(dst_folder, f"{split}_{filename}")
                shutil.copy(src_file, dst_file)  # Copying instead of moving to avoid destroying original dataset

base_dir = 'mixed' # Use the mixed dataset from now on, not the original one

def resize_rgb(): # Resizing the pictures and converting into RGB so they all have the same properties
    data = []
    labels = []
    for category in class_names:
        folder = os.path.join(base_dir, category)
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                image = Image.open(filepath).convert('RGB').resize((224, 224)) # has to be 224 for mobilenet
                data.append(np.array(image))
                labels.append(0 if category == 'NORMAL' else 1)
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")
    data = np.array(data)
    labels = np.array(labels)

    print("Data shape:", data.shape)
    print("Unique labels and counts:", np.unique(labels, return_counts=True))
    os.makedirs('./mobileNetData', exist_ok=True)
    np.savez('./mobileNetData/chest_xray_combined_mobileNet.npz', data=data, labels=labels) # I am saving a dataset in npz, so it is easier to work with it later
    print("Data saved to './mobileNetData/chest_xray_combined_mobileNet.npz'")

mix_data() # mix it up
resize_rgb() # resize and transfer to rgb for better consistency and future training
dataset = np.load('./mobileNetData/chest_xray_combined_mobileNet.npz')

new_data = dataset['data']
new_labels = dataset['labels']
unique_labels, counts = np.unique(new_labels, return_counts=True)

def pneumonia_vs_normal(): # checks how much of pneumonia and normal data are there in og dataset
    for name, count in zip(class_names, counts):
        print(f"{name}: {count} images")

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, counts, color=['skyblue', 'salmon'])
    plt.title('Class Distribution in Chest X-Ray Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def data_split(): # splitting the data
    data_train, data_temp, labels_train, labels_temp = train_test_split(
        new_data, new_labels, test_size=0.3, stratify=new_labels, random_state=21) # I use 70-15-15 split
    data_val, data_test, labels_val, labels_test = train_test_split(
        data_temp, labels_temp, test_size=0.5, stratify=labels_temp, random_state=21)

    print(f"Train: {len(data_train)}")
    print(f"Validation: {len(data_val)}")
    print(f"Test: {len(data_test)}")

    np.savez('./mobileNetData/chest_xray_split_mobileNet.npz', # save the new dataset with train/test/val split
             data_train=data_train, labels_train=labels_train,
             data_val=data_val, labels_val=labels_val,
             data_test=data_test, labels_test=labels_test)

def distribution_check(disease, set_name): # check, so the distribution of normal vs pneumonia pictures is similar for test/train/val
    unique, number = np.unique(disease, return_counts=True)
    total = sum(number)

    print(f"\n{set_name} set:")
    for cls, count in zip(unique, number):
        pct = 100 * count / total
        label_name = 'NORMAL' if cls == 0 else 'PNEUMONIA'
        print(f"  {label_name}: {count} ({pct:.2f}%)")
    print(f"  Total: {total}")

data_split() # split it up again with different train-val-test split size

split = np.load('./mobileNetData/chest_xray_split_mobileNet.npz')
distribution_check(split['labels_train'], "Train") # check the distribution
distribution_check(split['labels_val'], "Validation")
distribution_check(split['labels_test'], "Test")