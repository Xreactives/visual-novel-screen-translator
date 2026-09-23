import os
import shutil
import urllib.parse
from pathlib import Path

# Paths
NEW_IMAGES_DIR = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\dataset\new_images"
)
MAIN_IMAGES_DIR = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\dataset\images"
)
EXPORT_TEMP_LABELS = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\dataset\export_temp\labels"
)
DEST_LABELS_DIR = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\dataset\labels"
)

# 1. Consolidate new_images into main images directory
MAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DEST_LABELS_DIR.mkdir(parents=True, exist_ok=True)

moved_img_count = 0
if NEW_IMAGES_DIR.exists():
    for img in NEW_IMAGES_DIR.glob("*.*"):
        dest_img = MAIN_IMAGES_DIR / img.name
        if not dest_img.exists():
            shutil.copy2(img, dest_img)
            moved_img_count += 1

print(f"Copied {moved_img_count} new images into {MAIN_IMAGES_DIR}")

# 2. Process and flatten label files
processed_label_count = 0
if EXPORT_TEMP_LABELS.exists():
    for entry in os.listdir(EXPORT_TEMP_LABELS):
        if not entry.endswith(".txt"):
            continue

        src_path = os.path.join(str(EXPORT_TEMP_LABELS), entry)

        # Strip Label Studio hash prefix ('00bd5bc9__...')
        if "__" in entry:
            clean_encoded_name = entry.split("__", 1)[1]
        else:
            clean_encoded_name = entry

        # Decode URL special characters ('%20' -> ' ', '%28' -> '(', etc.)
        clean_name = urllib.parse.unquote(clean_encoded_name)

        # Extract strictly the filename (e.g. 'Screenshot (1892).txt')
        flat_filename = Path(clean_name).name
        dst_path = os.path.join(str(DEST_LABELS_DIR), flat_filename)

        with open(src_path, "r", encoding="utf-8") as f_in:
            content = f_in.read()

        with open(dst_path, "w", encoding="utf-8") as f_out:
            f_out.write(content)

        processed_label_count += 1

print(f"Successfully processed {processed_label_count} labels to {DEST_LABELS_DIR}")