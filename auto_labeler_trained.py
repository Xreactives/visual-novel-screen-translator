from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

# 1. Paths Setup
MODEL_PATH = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\runs\detect\vn_translator_yolo26\weights\best.pt"
)
SOURCE_IMAGES_DIR = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\detect_val"
)

# Target structure for label-studio-converter
DATASET_DIR = Path(r"F:\Pycharm Projects\visual-novel-screen-translator\dataset")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load model
print(f"Loading YOLO26 weights from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    0: "character_name",
    1: "choice_box",
    2: "dialogue_box"
}


def merge_horizontal_only(boxes_data, max_x_gap=35, max_y_gap=10):
    """Merges split segments on the exact same line while keeping vertical lines separate."""
    if not boxes_data:
        return []

    grouped = {}
    for box in boxes_data:
        cls_id = box[4]
        grouped.setdefault(cls_id, []).append(box)

    merged_results = []

    for cls_id, box_list in grouped.items():
        box_list.sort(key=lambda b: b[0])
        merged = [list(box_list[0])]

        for current in box_list[1:]:
            prev = merged[-1]

            same_line = abs(current[1] - prev[1]) < max_y_gap
            close_x = (current[0] - prev[2]) < max_x_gap

            if same_line and close_x:
                prev[2] = max(prev[2], current[2])  # Expand x2
                prev[3] = max(prev[3], current[3])  # Expand y2
                prev[1] = min(prev[1], current[1])  # Keep top y1
                prev[5] = max(prev[5], current[5])  # Keep higher confidence
            else:
                merged.append(list(current))

        for b in merged:
            merged_results.append(tuple(b))

    return merged_results


def generate_dataset_for_converter(conf_threshold: float = 0.45):
    image_files = list(SOURCE_IMAGES_DIR.glob("*.png")) + list(
        SOURCE_IMAGES_DIR.glob("*.jpg")
    )

    if not image_files:
        print(f"No images found in {SOURCE_IMAGES_DIR}")
        return

    # Write classes.txt inside the dataset root for reference
    classes_file = DATASET_DIR / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for idx in range(len(CLASS_NAMES)):
            f.write(f"{CLASS_NAMES[idx]}\n")

    print(f"Processing {len(image_files)} images...")

    for idx, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        img_height, img_width = image.shape[:2]

        # Copy image to ./dataset/images/
        target_img_path = IMAGES_DIR / img_path.name
        shutil.copy(img_path, target_img_path)

        # Run inference
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=0.45,
            verbose=False
        )[0]

        # Extract raw box coordinates
        raw_boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_boxes.append((x1, y1, x2, y2, cls_id, conf))

        # Filter and sort single lines top-to-bottom
        clean_boxes = merge_horizontal_only(raw_boxes, max_x_gap=35, max_y_gap=10)
        final_boxes = sorted(clean_boxes, key=lambda b: b[1])

        # Write matching YOLO format .txt label file into ./dataset/labels/
        txt_filename = LABELS_DIR / f"{img_path.stem}.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            for x1, y1, x2, y2, cls_id, conf in final_boxes:
                # Convert to normalized YOLO format (x_center, y_center, width, height)
                x_center = ((x1 + x2) / 2.0) / img_width
                y_center = ((y1 + y2) / 2.0) / img_height
                box_w = (x2 - x1) / img_width
                box_h = (y2 - y1) / img_height

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        print(f"[{idx}/{len(image_files)}] Auto-labeled & copied {img_path.name}")

    print(f"\nDataset successfully prepared at: {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    generate_dataset_for_converter(conf_threshold=0.45)