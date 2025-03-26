import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Paths
FER_CSV_PATH = "FERPlus/fer2013.csv"  # Path to the FER-2013 CSV file
FER_PLUS_LABELS_PATH = "FERPlus/fer2013new.csv"  # Path to FER+ relabeled annotations
OUTPUT_DIR = "data/processed"  # Directory to save processed images

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test", exist_ok=True)

# Load FER-2013 and FER+ labels
fer_data = pd.read_csv(FER_CSV_PATH)
fer_plus_labels = pd.read_csv(FER_PLUS_LABELS_PATH)

# Map FER+ labels to FER-2013 images
fer_data["emotion"] = fer_plus_labels.iloc[:, 2:].idxmax(axis=1)

# Preprocess images
def preprocess_images(data, output_dir, split_name):
    for index, row in data.iterrows():
        pixels = np.array(row["pixels"].split(), dtype="float32").reshape(48, 48)
        label = row["emotion"]
        label_dir = f"{output_dir}/{split_name}/{label}"
        os.makedirs(label_dir, exist_ok=True)
        image_path = f"{label_dir}/{index}.png"
        cv2.imwrite(image_path, pixels)

# Split dataset
train_data, test_data = train_test_split(fer_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Preprocess and save images
preprocess_images(train_data, OUTPUT_DIR, "train")
preprocess_images(val_data, OUTPUT_DIR, "val")
preprocess_images(test_data, OUTPUT_DIR, "test")

print("Preprocessing complete. Processed images saved to:", OUTPUT_DIR)