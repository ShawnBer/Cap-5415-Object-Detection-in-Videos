import json
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
annotations_path = script_dir / "test_annotations.json"
output_dir = script_dir / "yolo_labels"

#Load COCO annotations
with annotations_path.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annotations = coco["annotations"]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

#convert COCO annotations to YOLO format loop
for ann in annotations:
    img = images[ann["image_id"]]
    img_w, img_h = img["width"], img["height"]
    x, y, w, h = ann["bbox"]

    #Calculate normalized center coordinates and dimensions
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    #COCO category IDs start from 1, YOLO from 0
    class_id = ann["category_id"] - 1

    label_file = os.path.join(output_dir, f"{img['file_name'].split('.')[0]}.txt")
    with open(label_file, "a") as lf:
        lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("Conversion complete! YOLO labels saved in:", output_dir)