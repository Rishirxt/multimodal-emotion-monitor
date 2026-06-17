"""
Step 1: Look directly at the data before writing any model code.

Loads a handful of (image, label) pairs from a YOLO-format dataset
    images/<name>.jpg
    labels/<name>.txt
draws the ground-truth face boxes on each sampled image, and prints basic
dataset stats. If the boxes don't land on faces here, nothing built on top
of this data will work either -- so this step earns its place even though
it doesn't touch a model.

YOLO label format (one line per face, space-separated):
    class_id  x_center  y_center  width  height
All four numbers are normalized to [0, 1] relative to the image's own
width/height. That's what makes the same label file valid no matter what
resolution the image gets resized to during training.

Usage:
    python explore_data.py --images-dir /path/to/images --labels-dir /path/to/labels
"""

import argparse
import os
import random
from collections import Counter

import cv2


def load_yolo_label(label_path):
    """Read one .txt label file. Returns a list of (class_id, xc, yc, w, h),
    all four box values normalized to [0, 1]."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            boxes.append((class_id, x, y, w, h))
    return boxes


def yolo_to_pixel(box, img_w, img_h):
    """Convert one normalized (center, width, height) box to pixel (x1,y1,x2,y2).
    This is the conversion every later step will reuse."""
    _, xc, yc, w, h = box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def draw_boxes(img, boxes):
    out = img.copy()
    h, w = img.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def dataset_stats(images_dir, labels_dir):
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    n_boxes_per_image = []
    missing_labels = 0
    widths, heights = [], []
    class_counts = Counter()

    for fname in image_files:
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(label_path):
            missing_labels += 1
            continue
        boxes = load_yolo_label(label_path)
        n_boxes_per_image.append(len(boxes))
        for (cls, _, _, w, h) in boxes:
            widths.append(w)
            heights.append(h)
            class_counts[cls] += 1

    print(f"Images found:        {len(image_files)}")
    print(f"Missing label files: {missing_labels}")
    if n_boxes_per_image:
        avg = sum(n_boxes_per_image) / len(n_boxes_per_image)
        print(f"Faces per image -> min/avg/max: "
              f"{min(n_boxes_per_image)}/{avg:.2f}/{max(n_boxes_per_image)}")
        print(f"Images with 0 faces labeled: {n_boxes_per_image.count(0)}")
    if widths:
        print(f"Box width  (normalized) -> min/avg/max: "
              f"{min(widths):.3f}/{sum(widths)/len(widths):.3f}/{max(widths):.3f}")
        print(f"Box height (normalized) -> min/avg/max: "
              f"{min(heights):.3f}/{sum(heights)/len(heights):.3f}/{max(heights):.3f}")
    if class_counts:
        print(f"Class ids found: {dict(class_counts)} "
              f"(expect just one class, e.g. {{0: N}}, if this is a face-only dataset)")

    return image_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir", default="explore_output")
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image_files = dataset_stats(args.images_dir, args.labels_dir)
    if not image_files:
        print("No images found -- check --images-dir.")
        return

    sample = random.sample(image_files, min(args.n_samples, len(image_files)))
    print()
    for fname in sample:
        img_path = os.path.join(args.images_dir, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(args.labels_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}, skipping")
            continue

        boxes = load_yolo_label(label_path)
        vis = draw_boxes(img, boxes)
        out_path = os.path.join(args.out_dir, fname)
        cv2.imwrite(out_path, vis)
        print(f"{fname}: {len(boxes)} face(s) -> saved to {out_path}")

    print(f"\nDone. Open the images in '{args.out_dir}/' and check the boxes actually sit on faces.")


if __name__ == "__main__":
    main()