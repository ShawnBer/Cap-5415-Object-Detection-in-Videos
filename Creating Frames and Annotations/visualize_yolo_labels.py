#python visualize_yolo_labels.py --images "frames_output" --labels "annotations_for_frames" --out "visualized"

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def list_images(img_dir: str, max_count: Optional[int] = None) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    paths = sorted(paths)
    if max_count:
        paths = paths[:max_count]
    return paths


def draw_yolo_boxes(
    img: np.ndarray,
    label_path: str,
    class_names: Optional[List[str]] = None,
    color=(0, 255, 0),
    thickness=2
) -> Tuple[np.ndarray, dict]:

    #Draw YOLO-format labels on the image.
    #Returns (image_with_boxes, stats_dict).
    stats = {
        "boxes": 0,
        "invalid_lines": 0,
        "out_of_bounds": 0,
        "file_missing": False,
        "file_empty": False,
    }

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        stats["file_missing"] = True
        return img, stats

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        stats["file_empty"] = True
        return img, stats

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            stats["invalid_lines"] += 1
            continue

        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            conf = None
            if len(parts) >= 6:
                try:
                    conf = float(parts[5])
                except Exception:
                    conf = None
        except Exception:
            stats["invalid_lines"] += 1
            continue

        #Convert normalized center format to pixel xyxy
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        #Check out-of-bounds and clamp
        oob = any([x1 < 0, y1 < 0, x2 >= w, y2 >= h])
        if oob:
            stats["out_of_bounds"] += 1
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        #Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        #Label text
        if class_names and 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = f"id:{cls_id}"
        if conf is not None:
            label = f"{label} {conf:.2f}"

        #Background box behind text
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 6, th + 2)
        cv2.rectangle(img, (x1, y_text - th - 4), (x1 + tw + 4, y_text + baseline - 2), color, -1)
        cv2.putText(img, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        stats["boxes"] += 1

    return img, stats


def resolve_from_data_yaml(data_yaml: str, split: str) -> Tuple[str, str, List[str]]:

    #Resolve images and labels directories and class names from data.yaml.
    #Supports paths like:
      #train: images/train or a list of folders
      #val: images/val

    d = load_yaml(data_yaml)
    names = d.get("names")
    if isinstance(names, dict):
        #Convert {0:"person",1:"car"} to list by index
        names = [names[i] for i in sorted(names.keys())]

    split_key = split.lower()
    if split_key not in d:
        #Some YAMLs use 'train', 'val', 'test'; others 'path' + 'train/val' relative.
        #Try standard keys.
        if split_key == "val" and "validation" in d:
            split_key = "validation"
        elif split_key == "test" and "test" not in d:
            raise ValueError(f"Split '{split}' not found in {data_yaml}.")
    split_entry = d[split_key]
    if isinstance(split_entry, list):
        img_dir = split_entry[0]
    else:
        img_dir = split_entry

    #Infer labels directory:
    #If images/... -> labels/... with same split name
    if "images" in img_dir:
        lbl_dir = img_dir.replace("images", "labels")
    else:
        #Fallback: sibling 'labels' folder next to images
        parent = os.path.dirname(img_dir)
        split_name = os.path.basename(img_dir.rstrip("/"))
        lbl_dir = os.path.join(parent, "labels", split_name)

    return img_dir, lbl_dir, names


def main():

    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Visualize YOLO TXT labels on images (no inference).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", help="Path to images folder.")
    g.add_argument("--data", help="Path to data.yaml to auto-resolve paths.")
    ap.add_argument("--labels", help="Path to labels folder (required if using --images without --data).")
    ap.add_argument("--split", default="train", help="Split to use when resolving from data.yaml (train/val/test).")
    ap.add_argument("--out", required=True, help="Output folder for annotated images.")
    ap.add_argument("--names", nargs="*", default=None, help="Optional class names (overrides data.yaml).")
    ap.add_argument("--max", type=int, default=None, help="Max number of images to visualize.")
    ap.add_argument("--warn-only", action="store_true",
                    help="Continue on errors; otherwise, raise for critical mismatches.")
    args = ap.parse_args()

    

    script_dir = Path(__file__).resolve().parent

    def resolve_rel(p):
        # If p is None, leave it; if it is absolute, leave it; else make it script-dir relative
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else (script_dir / p)

    #Resolve --out always (it is required)
    args.out = str(resolve_rel(args.out))

    if args.data:
        # Resolve data.yaml path
        args.data = str(resolve_rel(args.data))
    else:
        # Resolve images/labels if provided
        if args.images:
            args.images = str(resolve_rel(args.images))
        if args.labels:
            args.labels = str(resolve_rel(args.labels))


    if args.data:
        img_dir, lbl_dir, yaml_names = resolve_from_data_yaml(args.data, args.split)
        class_names = args.names if args.names else yaml_names
    else:
        if not args.labels:
            ap.error("--labels is required when using --images.")
        img_dir, lbl_dir = args.images, args.labels
        class_names = args.names

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images folder not found: {img_dir}")
    if not os.path.isdir(lbl_dir):
        raise FileNotFoundError(f"Labels folder not found: {lbl_dir}")

    os.makedirs(args.out, exist_ok=True)

    image_paths = list_images(img_dir, max_count=args.max)
    if not image_paths:
        print("[WARN] No images found. Check your path and extensions.")
        sys.exit(0)

    #Main processing loop with stats tracking
    total = len(image_paths)
    counts = {
        "processed": 0,
        "missing_label": 0,
        "empty_label": 0,
        "invalid_lines": 0,
        "out_of_bounds": 0,
        "no_boxes": 0,
    }

    print(f"[INFO] Visualizing {total} images from: {img_dir}")
    print(f"[INFO] Labels directory: {lbl_dir}")
    if class_names:
        print(f"[INFO] Class names ({len(class_names)}): {class_names}")

    for i, img_path in enumerate(image_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(lbl_dir, base + ".txt")

        vis, stats = draw_yolo_boxes(img, label_path, class_names=class_names)

        #Update counts based on stats
        if stats["file_missing"]:
            counts["missing_label"] += 1
        if stats["file_empty"]:
            counts["empty_label"] += 1
        if stats["boxes"] == 0:
            counts["no_boxes"] += 1
        counts["invalid_lines"] += stats["invalid_lines"]
        counts["out_of_bounds"] += stats["out_of_bounds"]

        out_path = os.path.join(args.out, f"{base}_viz.jpg")
        ok = cv2.imwrite(out_path, vis)
        if not ok:
            print(f"[WARN] Failed to write: {out_path}")

        counts["processed"] += 1
        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] Wrote: {out_path}")

    #Summary
    print("\n====== SUMMARY ======")
    print(f"Processed images : {counts['processed']}")
    print(f"Missing labels   : {counts['missing_label']}")
    print(f"Empty label files: {counts['empty_label']}")
    print(f"No boxes drawn   : {counts['no_boxes']}")
    print(f"Invalid lines    : {counts['invalid_lines']}")
    print(f"Out-of-bounds bb : {counts['out_of_bounds']}")
    print(f"Output previews  : {args.out}")

    #sanity check
    if not args.warn_only and (counts["missing_label"] > 0 or counts["invalid_lines"] > 0):
        print("\n[ERROR] Found issues. Re-run with --warn-only to ignore.")
        sys.exit(2)


if __name__ == "__main__":
    main()
