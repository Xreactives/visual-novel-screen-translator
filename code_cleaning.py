import os
import re
import shutil
import time
import cv2
import numpy as np

# --- Start Timer ---
start_time = time.time()

# --- Configuration Paths ---
CROP_DIR = './dataset/crops'
GARBAGE_DIR = './dataset/garbage_crops'  # Inspect purged crops here!
LABEL_FILE = './dataset/labels.txt'
CLEANED_LABEL_FILE = './dataset/labels_cleaned.txt'

os.makedirs(GARBAGE_DIR, exist_ok=True)

# --- Rule Thresholds ---
MIN_WIDTH = 15  # Minimum valid crop width (pixels)
MIN_HEIGHT = 10  # Absolute minimum height (pixels)
FURIGANA_RATIO = 0.35  # Relative ratio threshold (35%)
MAX_FURIGANA_CAP = 26  # SAFETY CAP: Furigana cutoff will NEVER exceed 26px
MAX_ASPECT_RATIO = 20.0  # Max width/height ratio before flagging as anomaly

# Regex for Japanese Kana/Kanji check
JAPANESE_CHAR_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]')

# Regex for punctuation-only text (brackets, dots, symbols)
PUNCTUATION_ONLY_PATTERN = re.compile(
    r'^[\s「」『』【】（）〔〕…・。、！？!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]+$'
)

# Rule H & K: Hallucinated UI text, arrows, standalone Kana, and numeric dots ('0', '°')
UI_HALLUCINATION_PATTERN = re.compile(
    r'^[\s>><|\\/*_\-–—]+$|^[\u3040-\u309F\u30A0-\u30FF]$|^[\s0oO°゜・.●◯◎○]+$'
)

# Rule L: Pure Number-Only Filter (ASCII + Full-Width digits)
NUMBER_ONLY_PATTERN = re.compile(r'^[\s0-9０-９]+$')

# Rule M: Pure Icon / Non-Text Symbol Filter
ICON_ONLY_PATTERN = re.compile(
    r'^[\s!@#$%^&*()_+\-=\[\]{};:\'",.<>/?\\|`~★☆♪♥♦♠♣◈◇◆▲▼△▽→←↑↓»«›‹⚙✕✖✗]+$'
)


# Rule I: Function to detect UI icon strips (Fast-Forward / Settings gear bar)
def is_ui_icon_strip(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_shapes = 0
    for c in contours:
        _, _, w_box, h_box = cv2.boundingRect(c)
        if w_box > 5 and h_box > 5:
            valid_shapes += 1

    h_img, w_img = gray.shape
    aspect_ratio = w_img / float(h_img)

    if aspect_ratio > 2.5 and 2 <= valid_shapes <= 5 and h_img < 45:
        return True

    return False


# Rule J: Function to detect circular UI radio buttons / dots (hallucinated as '0')
def is_circular_ui_icon(img):
    h_img, w_img = img.shape[:2]
    if w_img > 45 or h_img > 45:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=15,
        minRadius=3,
        maxRadius=20,
    )

    return circles is not None


# Rule N: Pure Image-Based Art & Texture Crop Filter (Hair, Grass, CG Artifacts)
def is_background_art_crop(img):
  h, w = img.shape[:2]
  aspect_ratio = w / float(h)

  # 1. Safe Harbor: Real dialogue lines are very horizontally wide.
  # If it is much wider than it is tall, it is valid text. Skip the art check entirely.
  if aspect_ratio > 4.0:
    return False

  # Ignore small crops (these are handled by UI and Furigana filters)
  if h < 45:
    return False

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

  # 2. Detect vertical edge strokes via Sobel X gradient
  sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  abs_sobel_x = np.uint8(np.absolute(sobel_x))
  _, vertical_edges = cv2.threshold(abs_sobel_x, 40, 255, cv2.THRESH_BINARY)

  # 3. Check margin penetration
  # We increased the thresholds to 0.08 so dense Kanji isn't accidentally flagged
  top_row_edges = (vertical_edges[: int(h * 0.15), :] > 0).mean()
  bottom_row_edges = (vertical_edges[int(h * 0.85) :, :] > 0).mean()
  total_edge_ratio = (vertical_edges > 0).mean()

  # Continuous strokes through top and bottom borders indicate blocky artwork
  if top_row_edges > 0.08 and bottom_row_edges > 0.08 and total_edge_ratio > 0.08:
    return True

  return False


valid_crops = {}

# 1. Load initial labels (Updated for 4-part format: crop_name, raw_img, coords, text)
if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                valid_crops[parts[0]] = (parts[1], parts[2], parts[3])
else:
    print(f'Error: Label file "{LABEL_FILE}" not found.')
    exit()

# 2. Robust Height Pass: Calculate 75th percentile height
all_heights = []
for img_name in valid_crops.keys():
    img_path = os.path.join(CROP_DIR, img_name)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            all_heights.append(img.shape[0])

if all_heights:
    typical_dialogue_height = np.percentile(all_heights, 75)
    calculated_cutoff = typical_dialogue_height * FURIGANA_RATIO
    dynamic_furigana_cutoff = min(calculated_cutoff, MAX_FURIGANA_CAP)
else:
    dynamic_furigana_cutoff = 22.0

print(
    f'Typical Line Height: {typical_dialogue_height:.1f}px | Active Furigana'
    f' Cutoff: {dynamic_furigana_cutoff:.2f}px\n'
)

removed_count = 0
reason_breakdown = {
    'corrupted': 0,
    'dimension': 0,
    'furigana_35percent': 0,
    'ui_x_button': 0,
    'punctuation_only': 0,
    'no_japanese': 0,
    'number_only': 0,
    'icon_only': 0,
    'background_art': 0,
    'low_edge_density': 0,
    'blank_image': 0,
    'ui_hallucination': 0,
    'ui_icon_strip': 0,
    'ui_circular_dot': 0,
}


def purge_crop(img_name, img_path, reason):
    """Moves purged crop to garbage folder instead of hard deleting."""
    global removed_count
    garbage_path = os.path.join(GARBAGE_DIR, f'{reason}_{img_name}')
    if os.path.exists(img_path):
        shutil.move(img_path, garbage_path)
    del valid_crops[img_name]
    reason_breakdown[reason] += 1
    removed_count += 1


# 3. Process and filter dataset
for img_name in list(valid_crops.keys()):
    img_path = os.path.join(CROP_DIR, img_name)

    # Unpack metadata (Updated)
    raw_filename, coords, text = valid_crops[img_name]
    text = text.strip()

    if not os.path.exists(img_path):
        del valid_crops[img_name]
        continue

    img = cv2.imread(img_path)

    # Rule A: Corrupted Image
    if img is None:
        purge_crop(img_name, img_path, 'corrupted')
        continue

    h, w, _ = img.shape
    aspect_ratio = w / float(h)

    # Rule N: Purge Background Art / Hair / Texture Hallucinations
    if is_background_art_crop(img):
        purge_crop(img_name, img_path, 'background_art')
        continue

    # Rule B: Extreme Size Anomaly
    if w < MIN_WIDTH or h < MIN_HEIGHT or aspect_ratio > MAX_ASPECT_RATIO:
        purge_crop(img_name, img_path, 'dimension')
        continue

    # Rule C: Furigana Filtering with Safety Cap
    if h <= dynamic_furigana_cutoff:
        purge_crop(img_name, img_path, 'furigana_35percent')
        continue

    # Rule L: Purge Pure Number-Only Crops
    if NUMBER_ONLY_PATTERN.match(text):
        purge_crop(img_name, img_path, 'number_only')
        continue

    # Rule M: Purge Pure Icon Crops
    if ICON_ONLY_PATTERN.match(text):
        purge_crop(img_name, img_path, 'icon_only')
        continue

    # Rule D: UI Close-Button Artifacts
    is_x_text = text.lower() in ['x', '×', '*', 'x']
    is_square = 0.7 <= aspect_ratio <= 1.4
    is_button_size = w < 60 and h < 60
    if (is_x_text and is_square) or (is_square and is_button_size):
        purge_crop(img_name, img_path, 'ui_x_button')
        continue

    # Rule E: Punctuation-Only / Standalone Brackets
    if PUNCTUATION_ONLY_PATTERN.match(text):
        purge_crop(img_name, img_path, 'punctuation_only')
        continue

    # Rule F: Non-Japanese Text
    if not JAPANESE_CHAR_PATTERN.search(text):
        purge_crop(img_name, img_path, 'no_japanese')
        continue

    # Rule H & K: Hallucinated UI Text / Numeric Dots
    if UI_HALLUCINATION_PATTERN.match(text):
        purge_crop(img_name, img_path, 'ui_hallucination')
        continue

    # Rule I: Multi-Icon Control Bars
    if is_ui_icon_strip(img):
        purge_crop(img_name, img_path, 'ui_icon_strip')
        continue

    # Rule J: Circular Radio Buttons
    if is_circular_ui_icon(img):
        purge_crop(img_name, img_path, 'ui_circular_dot')
        continue

    # Rule G: Image Edge Density Check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 10:
        purge_crop(img_name, img_path, 'blank_image')
        continue

    edges = cv2.Canny(gray, 100, 200)
    edge_density = (edges > 0).mean()
    if edge_density < 0.035 and is_square:
        purge_crop(img_name, img_path, 'low_edge_density')
        continue

# 4. Save synchronized clean labels (Updated for 4-part format)
with open(CLEANED_LABEL_FILE, 'w', encoding='utf-8') as f:
    for crop_filename, meta in valid_crops.items():
        raw_filename, coords, text = meta
        f.write(f'{crop_filename}\t{raw_filename}\t{coords}\t{text}\n')

# --- Stop Timer ---
elapsed_time = time.time() - start_time

# 5. Summary Output
print('==============================================')
print(f'DATASET CLEANUP COMPLETE! Moved {removed_count} crops to garbage folder.')
print('==============================================')
print('Reason Breakdown:')
for reason, count in reason_breakdown.items():
    if count > 0:
        print(f' - {reason.replace("_", " ").title()}: {count}')
print('----------------------------------------------')
print(f'Garbage crops stored in: {GARBAGE_DIR}')
print(f'Clean labels saved to: {CLEANED_LABEL_FILE}')
print(f'Remaining valid crops: {len(valid_crops)}')
print(f'Time Elapsed: {elapsed_time:.2f} seconds')
print('==============================================')