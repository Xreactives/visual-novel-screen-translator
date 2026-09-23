import os
import time
import cv2
import easyocr
import numpy as np
import gc
import torch

# Initialize EasyOCR with GPU acceleration
reader = easyocr.Reader(['ja', 'en'], gpu=True)

IMAGE_DIR = 'dataset/images'
OUTPUT_CROP_DIR = './dataset/crops'
LABEL_FILE = './dataset/labels.txt'

os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)


def preprocess_frame_for_ocr(img):
    """Enhance image contrast for better choice menu detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


all_images = sorted([
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
total_images = len(all_images)

if total_images == 0:
    print(f'No images found in "{IMAGE_DIR}"!')
    exit()

print(f'Starting Sequential (Low-RAM) OCR Pipeline on {total_images} screenshots...\n')

count = 0
start_time = time.time()

with open(LABEL_FILE, 'w', encoding='utf-8') as f_label:
    # Strictly sequential loop: Only one image is processed at a time
    for idx, img_name in enumerate(all_images, start=1):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h_img, w_img, _ = img.shape

        # Preprocess and detect
        enhanced_img = preprocess_frame_for_ocr(img)
        results = reader.readtext(
            enhanced_img,
            text_threshold=0.3,
            low_text=0.3,
            link_threshold=0.2,
            width_ths=0.7,
            canvas_size=1280,
        )

        for bbox, text, prob in results:
            clean_text = text.strip()
            if prob < 0.6 or not clean_text:
                continue

            (tl, tr, br, bl) = bbox
            x_min = max(0, int(tl[0]))
            y_min = max(0, int(tl[1]))
            x_max = min(w_img, int(br[0]))
            y_max = min(h_img, int(br[1]))

            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue

            crop_filename = f'crop_{count:06d}.png'
            crop_path = os.path.join(OUTPUT_CROP_DIR, crop_filename)

            cv2.imwrite(crop_path, crop)

            # Save with 4-part YOLO metadata: crop_filename \t raw_filename \t coordinates \t text
            f_label.write(
                f'{crop_filename}\t{img_name}\t{x_min},{y_min},{x_max},{y_max}\t{clean_text}\n'
            )
            count += 1

        # Destroy image matrices immediately after processing
        del img, enhanced_img, results

        # Run garbage collection strictly every 5 frames
        if idx % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Progress tracking
        elapsed = time.time() - start_time
        percent = (idx / total_images) * 100
        avg_speed = idx / elapsed
        eta_seconds = (total_images - idx) / avg_speed if avg_speed > 0 else 0

        print(
            f'\rProgress: [{idx}/{total_images}] {percent:5.1f}% | '
            f'Crops: {count:4d} | '
            f'Speed: {avg_speed:.2f} img/s | '
            f'ETA: {int(eta_seconds)}s',
            end='',
            flush=True,
        )

print('\n\nSequential auto-labeling complete with coordinate metadata!')