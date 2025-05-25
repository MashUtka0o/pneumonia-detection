#Combines all files to do our own split

import os
import shutil
import random
from pathlib import Path

def combine_dataset():
    dataset = "./chest_xray"
    splits = ["train", "test", "val"]
    output_dir = "./chest_xray_combined"

    # Create output folders
    os.makedirs(os.path.join(output_dir, "NORMAL"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "PNEUMONIA_VIRUS"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "PNEUMONIA_BACTERIA"), exist_ok=True)

    counters = {
        "NORMAL": 0,
        "PNEUMONIA_VIRUS": 0,
        "PNEUMONIA_BACTERIA": 0
    }

    for split in splits:
        for label in ["NORMAL", "PNEUMONIA"]:
            src_dir = os.path.join(dataset, split, label)
            if not os.path.exists(src_dir):
                continue
            for fname in os.listdir(src_dir):
                src_path = os.path.join(src_dir, fname)
                if label == "NORMAL":
                    dst_label = "NORMAL"
                else:
                    if "virus" in fname.lower():
                        dst_label = "PNEUMONIA_VIRUS"
                    else:
                        dst_label = "PNEUMONIA_BACTERIA"
                dst_dir = os.path.join(output_dir, dst_label)
                ext = os.path.splitext(fname)[1]
                new_fname = f"{counters[dst_label]}{ext}"
                counters[dst_label] += 1
                shutil.move(src_path, os.path.join(dst_dir, new_fname))


def split_train_test():
    # Configuration
    original_dir = './chest_xray_combined'  # Path to the combined dataset
    output_dir = './chest_xray_split'
    classes = ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS']
    splits = ['train', 'val', 'test']
    split_ratios = [0.7, 0.15, 0.15]  # train, val, test

    # Create output directories
    for split in splits:
        for cls in classes:
            Path(f'{output_dir}/{split}/{cls}').mkdir(parents=True, exist_ok=True)

    # Process each class
    for cls in classes:
        img_paths = list(Path(f'{original_dir}/{cls}').glob('*'))
        random.shuffle(img_paths)
        total = len(img_paths)
        
        train_end = int(split_ratios[0] * total)
        val_end = train_end + int(split_ratios[1] * total)

        split_data = {
            'train': img_paths[:train_end],
            'val': img_paths[train_end:val_end],
            'test': img_paths[val_end:]
        }

        for split in splits:
            for img_path in split_data[split]:
                dest = Path(f'{output_dir}/{split}/{cls}/{img_path.name}')
                shutil.copy(img_path, dest)

    print("Dataset successfully split into train/val/test!")

