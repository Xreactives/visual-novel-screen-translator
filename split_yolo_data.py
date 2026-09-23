import os
import shutil
import random

# --- Configurations ---
IMG_DIR = 'dataset/images'
LBL_DIR = './dataset/labels'
OUT_DIR = './yolo_dataset'
TRAIN_RATIO = 0.8  # 80% Train, 20% Validation

# Setup YOLOv5 folder structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels', split), exist_ok=True)

# Find all images that have a corresponding .txt label
all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
valid_pairs = []

for img_file in all_images:
    lbl_file = os.path.splitext(img_file)[0] + '.txt'
    if os.path.exists(os.path.join(LBL_DIR, lbl_file)):
        valid_pairs.append((img_file, lbl_file))

if not valid_pairs:
    print("No paired images and labels found! Check your folder paths.")
    exit()

# Shuffle and calculate split
random.shuffle(valid_pairs)
train_size = int(len(valid_pairs) * TRAIN_RATIO)

print(f"Splitting {len(valid_pairs)} total files...")

# Copy files to new YOLOv5 structure
for i, (img_file, lbl_file) in enumerate(valid_pairs):
    split = 'train' if i < train_size else 'val'

    shutil.copy(os.path.join(IMG_DIR, img_file), os.path.join(OUT_DIR, 'images', split, img_file))
    shutil.copy(os.path.join(LBL_DIR, lbl_file), os.path.join(OUT_DIR, 'labels', split, lbl_file))

print(f"Success! {train_size} Training | {len(valid_pairs) - train_size} Validation.")
print(f"Dataset successfully built at: {OUT_DIR}")