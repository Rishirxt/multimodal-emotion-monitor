"""
Emotion Phase 1: Look at the data before writing any model code.

FER2013 layout:
    train/
        angry/    Training_XXXXX.jpg ...
        disgust/  ...
        fear/     ...
        happy/    ...
        neutral/  ...
        sad/      ...
        surprise/ ...
    test/         (same structure)

The folder name IS the label — no separate label files needed.

Usage:
    python emotion_explore.py --data-dir C:/path/to/EmotionData
"""

import argparse
import os
import random
from collections import Counter

import cv2
import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def scan_split(split_dir):
    """Returns list of (filepath, emotion_label) for every image found."""
    samples = []
    for emotion in EMOTIONS:
        folder = os.path.join(split_dir, emotion)
        if not os.path.isdir(folder):
            print(f"  WARNING: folder not found — {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append((os.path.join(folder, fname), emotion))
    return samples


def dataset_stats(split_dir, split_name):
    samples = scan_split(split_dir)
    counts  = Counter(label for _, label in samples)

    print(f"\n{'='*50}")
    print(f"  {split_name}  —  {len(samples)} total images")
    print(f"{'='*50}")

    widths, heights = [], []
    corrupted = 0
    for fpath, _ in random.sample(samples, min(200, len(samples))):
        img = cv2.imread(fpath)
        if img is None:
            corrupted += 1
            continue
        h, w = img.shape[:2]
        widths.append(w); heights.append(h)

    print(f"\n  Per-class counts:")
    max_count = max(counts.values())
    for emotion in EMOTIONS:
        n    = counts.get(emotion, 0)
        bar  = '█' * int(30 * n / max_count)
        print(f"    {emotion:10s}  {n:5d}  {bar}")

    if widths:
        print(f"\n  Image size (sampled 200):")
        print(f"    width  → min {min(widths)}  avg {sum(widths)//len(widths)}  max {max(widths)}")
        print(f"    height → min {min(heights)}  avg {sum(heights)//len(heights)}  max {max(heights)}")
        channels = set()
        for fpath, _ in random.sample(samples, min(20, len(samples))):
            img = cv2.imread(fpath)
            if img is not None:
                channels.add(img.shape[2] if len(img.shape)==3 else 1)
        print(f"    channels: {channels}  (3=BGR colour, 1=grayscale)")

    if corrupted:
        print(f"\n  WARNING: {corrupted} images could not be read — may be corrupted")

    return samples, counts


def save_sample_grid(samples, out_path, n=21):
    """Save a 3×7 grid — one random sample per emotion."""
    grid_imgs = []
    for emotion in EMOTIONS:
        emotion_samples = [(f,l) for f,l in samples if l == emotion]
        picked = random.sample(emotion_samples, min(3, len(emotion_samples)))
        for fpath, label in picked:
            img = cv2.imread(fpath)
            if img is None:
                img = np.zeros((48,48,3), dtype=np.uint8)
            img = cv2.resize(img, (96, 96))
            # label bar at the bottom
            cv2.rectangle(img, (0, 78), (96, 96), (0,0,0), -1)
            cv2.putText(img, label, (3, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            grid_imgs.append(img)

    # pad to fill 3 rows of 7
    while len(grid_imgs) < 21:
        grid_imgs.append(np.zeros((96,96,3), dtype=np.uint8))

    rows = [np.hstack(grid_imgs[i*7:(i+1)*7]) for i in range(3)]
    grid = np.vstack(rows)
    cv2.imwrite(out_path, grid)
    print(f"\n  Sample grid saved → {out_path}")
    print("  Open it to visually confirm images look like face crops.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True,
                        help='Root folder containing train/ and test/ subfolders')
    parser.add_argument('--out-dir', default='emotion_explore_output')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, 'train')
    test_dir  = os.path.join(args.data_dir, 'test')

    train_samples, train_counts = dataset_stats(train_dir, 'TRAIN')
    test_samples,  test_counts  = dataset_stats(test_dir,  'TEST')

    # class imbalance check
    counts = list(train_counts.values())
    ratio  = max(counts) / max(min(counts), 1)
    print(f"\n  Class imbalance ratio (train): {ratio:.1f}x")
    if ratio > 5:
        print("  NOTE: significant imbalance — we'll use weighted loss during training")
    else:
        print("  Imbalance is manageable — standard cross-entropy loss is fine")

    # save grids
    save_sample_grid(train_samples,
                     os.path.join(args.out_dir, 'train_samples.jpg'))
    save_sample_grid(test_samples,
                     os.path.join(args.out_dir, 'test_samples.jpg'))

    print(f"\n  Next step: run emotion_train.py once you've confirmed the grid looks correct.")


if __name__ == '__main__':
    main()