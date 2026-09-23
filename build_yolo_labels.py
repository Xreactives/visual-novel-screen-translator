from collections import defaultdict
import os
import cv2

CROP_DIR = './dataset/crops'
LABEL_FILE = './dataset/labels_cleaned.txt'
RAW_SCREENSHOTS_DIR = 'dataset/images'
YOLO_LABEL_DIR = './dataset/labels'

os.makedirs(YOLO_LABEL_DIR, exist_ok=True)

# Keyword triggers for Class 1 (choice_box)
CHOICE_KEYWORDS = ['はい', 'いいえ', '選択', '・']


def determine_class_id(text, crop_img):
  h, w = crop_img.shape[:2]
  aspect_ratio = w / float(h)

  # Centered wide option buttons are treated as Class 1 (choice_box)
  if any(kw in text for kw in CHOICE_KEYWORDS) and aspect_ratio > 3.0:
    return 1

  return 0  # Default to Class 0 (dialogue_box)


screenshot_annotations = defaultdict(list)
converted_crops = 0

if not os.path.exists(LABEL_FILE):
  print(f'Error: "{LABEL_FILE}" not found.')
  exit()

with open(LABEL_FILE, 'r', encoding='utf-8') as f:
  for line in f:
    parts = line.strip().split('\t')
    if len(parts) != 4:
      continue

    crop_name, raw_filename, coords_str, text = (
        parts[0],
        parts[1],
        parts[2],
        parts[3].strip(),
    )
    crop_path = os.path.join(CROP_DIR, crop_name)

    # Skip purged crops (UI icons, furigana, background art)
    if not os.path.exists(crop_path):
      continue

    crop_img = cv2.imread(crop_path)
    if crop_img is None:
      continue

    raw_img_path = os.path.join(RAW_SCREENSHOTS_DIR, raw_filename)
    if not os.path.exists(raw_img_path):
      continue

    raw_img = cv2.imread(raw_img_path)
    if raw_img is None:
      continue

    img_h, img_w = raw_img.shape[:2]

    # Parse coordinates (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = map(int, coords_str.split(','))

    # Convert to normalized YOLO bounding box format (0.0 to 1.0)
    x_center = ((x_min + x_max) / 2.0) / img_w
    y_center = ((y_min + y_max) / 2.0) / img_h
    box_w = (x_max - x_min) / float(img_w)
    box_h = (y_max - y_min) / float(img_h)

    class_id = determine_class_id(text, crop_img)

    annotation_line = (
        f'{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}'
    )
    screenshot_annotations[raw_filename].append(annotation_line)
    converted_crops += 1

# Write individual YOLO label files (.txt) per full screenshot
for raw_filename, lines in screenshot_annotations.items():
  txt_filename = os.path.splitext(raw_filename)[0] + '.txt'
  txt_path = os.path.join(YOLO_LABEL_DIR, txt_filename)

  with open(txt_path, 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(lines) + '\n')

print('==============================================')
print(
    f'Successfully generated YOLO annotations for {converted_crops} valid text'
    ' boxes!'
)
print(f'YOLO Labels written to: {YOLO_LABEL_DIR}')
print('==============================================')