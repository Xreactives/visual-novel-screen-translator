import os
import cv2
import easyocr

# Initialize EasyOCR Japanese reader
reader = easyocr.Reader(['ja', 'en'])

IMAGE_DIR = './raw_screenshots'
OUTPUT_CROP_DIR = './dataset/crops'
LABEL_FILE = './dataset/labels.txt'

os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)
count = 0

with open(LABEL_FILE, 'w', encoding='utf-8') as f_label:
  for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    results = reader.readtext(img_path)

    img = cv2.imread(img_path)
    if img is None:
      continue

    for bbox, text, prob in results:
      # Skip low-confidence predictions to keep dataset clean
      if prob < 0.6 or not text.strip():
        continue

      # Extract Bounding Box coordinates
      (tl, tr, br, bl) = bbox
      x_min, y_min = int(tl[0]), int(tl[1])
      x_max, y_max = int(br[0]), int(br[1])

      crop = img[y_min:y_max, x_min:x_max]
      if crop.size == 0:
        continue

      crop_filename = f'crop_{count:06d}.png'
      crop_path = os.path.join(OUTPUT_CROP_DIR, crop_filename)

      cv2.imwrite(crop_path, crop)
      f_label.write(f'{crop_filename}\t{text.strip()}\n')
      count += 1

print(f'Created dataset with {count} labeled crops!')