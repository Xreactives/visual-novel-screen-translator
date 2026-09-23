import os
import shutil
import re

# --- Configuration Paths ---
CROP_DIR = './dataset/crops'
GARBAGE_DIR = './dataset/garbage_crops'
CLEANED_LABEL_FILE = './dataset/labels_cleaned.txt'


def undo_cleaning():
    print("Starting Undo Process...")

    if not os.path.exists(GARBAGE_DIR):
        print(f"Garbage directory '{GARBAGE_DIR}' not found. Nothing to undo.")
        return

    garbage_files = os.listdir(GARBAGE_DIR)
    if not garbage_files:
        print("No files found in the garbage folder.")
        return

    restored_count = 0

    # 1. Move files back and restore their original names
    for filename in garbage_files:
        # Regex to extract the original 'crop_XXXXXX.png' name from 'reason_crop_XXXXXX.png'
        match = re.search(r'(crop_\d+\.(png|jpg|jpeg))', filename)

        if match:
            original_name = match.group(1)
            garbage_path = os.path.join(GARBAGE_DIR, filename)
            restore_path = os.path.join(CROP_DIR, original_name)

            # Move the file back to the crops folder
            shutil.move(garbage_path, restore_path)
            restored_count += 1

    # 2. Delete the cleaned label file so you don't accidentally use outdated clean data
    if os.path.exists(CLEANED_LABEL_FILE):
        os.remove(CLEANED_LABEL_FILE)
        print(f"Deleted outdated '{CLEANED_LABEL_FILE}'.")

    print('==============================================')
    print(f'UNDO COMPLETE! Restored {restored_count} crops back to {CROP_DIR}.')
    print('Your dataset is exactly as it was before running code_cleaning.py.')
    print('==============================================')


if __name__ == '__main__':
    undo_cleaning()