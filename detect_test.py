from pathlib import Path
import cv2
from ultralytics import YOLO

# 1. Setup paths
MODEL_PATH = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\runs\detect\vn_translator_yolo26\weights\best.pt"
)
VAL_IMAGES_DIR = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\detect_val"
)
OUTPUT_DIR = Path(
    r"F:\Pycharm Projects\visual-novel-screen-translator\output_predictions"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load fine-tuned YOLO26 model
print(f"Loading weights from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    0: "character_name",
    1: "choice_box",
    2: "dialogue_box"
}

CLASS_COLORS = {
    0: (255, 128, 0),  # Blue/Cyan
    1: (0, 0, 255),    # Red
    2: (0, 255, 0),    # Green
}


def merge_horizontal_only(boxes_data, max_x_gap=35, max_y_gap=15):
    """
    Merges horizontally broken line segments ON THE SAME LINE ONLY.
    Does NOT merge vertically stacked lines (keeps lines separate for CRNN).
    """
    if not boxes_data:
        return []

    grouped = {}
    for box in boxes_data:
        cls_id = box[4]
        grouped.setdefault(cls_id, []).append(box)

    merged_results = []

    for cls_id, box_list in grouped.items():
        # Sort left-to-right
        box_list.sort(key=lambda b: b[0])
        merged = [list(box_list[0])]

        for current in box_list[1:]:
            prev = merged[-1]

            # Only merge if on the EXACT same horizontal line
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


def test_eval_images(conf_threshold: float = 0.50, max_images: int = 1000):
    image_files = list(VAL_IMAGES_DIR.glob("*.png")) + list(
        VAL_IMAGES_DIR.glob("*.jpg")
    )

    if not image_files:
        print(f"No validation images found in {VAL_IMAGES_DIR}")
        return

    total_to_process = min(len(image_files), max_images)
    print(f"Found {len(image_files)} validation images. Processing up to {total_to_process}...")

    for idx, img_path in enumerate(image_files[:total_to_process], 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=0.45,
            verbose=False
        )[0]

        # 1. Extract raw detection data
        raw_boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_boxes.append((x1, y1, x2, y2, cls_id, conf))

        # 2. Merge horizontal fragments ON THE SAME LINE ONLY
        single_line_boxes = merge_horizontal_only(raw_boxes, max_x_gap=35, max_y_gap=10)

        # 3. Sort boxes TOP-TO-BOTTOM by Y1 coordinate (prepares correct reading order for CRNN)
        final_boxes = sorted(single_line_boxes, key=lambda b: b[1])

        # 4. Draw individual single-line boxes
        for x1, y1, x2, y2, cls_id, conf in final_boxes:
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

            label_text = f"{cls_name} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            text_y1 = max(y1 - text_height - 6, 0)
            text_y2 = text_y1 + text_height + 6
            text_x2 = x1 + text_width + 6

            cv2.rectangle(image, (x1, text_y1), (text_x2, text_y2), color, cv2.FILLED)
            cv2.putText(
                image,
                label_text,
                (x1 + 3, text_y2 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        save_path = OUTPUT_DIR / f"eval_{img_path.name}"
        cv2.imwrite(str(save_path), image)

        print(
            f"[{idx}/{total_to_process}] {img_path.name} -> "
            f"{len(final_boxes)} single-line crops ready for CRNN -> Saved to {save_path.name}"
        )

    print(f"\nAll prediction images successfully saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    test_eval_images(conf_threshold=0.50, max_images=1000)