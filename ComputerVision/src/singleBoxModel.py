"""
Step 2: The smallest possible trainable face detector.

This model does NOT use a grid or anchors. It looks at a whole image and
directly regresses ONE bounding box: (center_x, center_y, width, height),
all normalized to [0, 1] -- the exact same units your label files already
use, so no coordinate conversion happens anywhere in this script.

Because it only ever outputs one box, we deliberately train it on the
single-face subset of your data and skip every image with 2+ faces.
Watching this model run on a multi-face image afterwards is the point of
this step: it physically cannot output more than one box, and *that*
limitation is exactly the gap that grids + anchors (the next step) exist
to close.

Usage:
    python step2_single_box_model.py --images-dir ... --labels-dir ... --epochs 15
"""

import argparse
import os
import random

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_yolo_label(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append(list(map(float, parts[1:5])))  # xc, yc, w, h
    return boxes


class SingleFaceDataset(Dataset):
    """Only keeps images that have EXACTLY one labeled face."""

    def __init__(self, images_dir, labels_dir, img_size=128):
        self.images_dir = images_dir
        self.img_size = img_size
        self.samples = []

        all_images = sorted(f for f in os.listdir(images_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
        skipped_multi = 0
        for fname in all_images:
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.exists(label_path):
                continue
            boxes = load_yolo_label(label_path)
            if len(boxes) == 1:
                self.samples.append((fname, boxes[0]))
            elif len(boxes) > 1:
                skipped_multi += 1

        print(f"SingleFaceDataset: kept {len(self.samples)} single-face images, "
              f"skipped {skipped_multi} multi-face images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, box = self.samples[idx]
        img = cv2.imread(os.path.join(self.images_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resizing (even non-uniformly) doesn't break the label: cx/w are
        # already fractions of width, cy/h are already fractions of height,
        # and those fractions don't change when the image is resized.
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        target = torch.tensor(box, dtype=torch.float32)
        return img_t, target


class TinySingleBoxNet(nn.Module):
    """Whole-image -> one box. No grid, no anchors, no classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 128->64
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4), nn.Sigmoid(),   # squash to [0,1] -- same units as the labels
        )

    def forward(self, x):
        return self.head(self.features(x))


def yolo_to_pixel(box, w, h):
    xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return int(x1), int(y1), int(x2), int(y2)


def visualize(model, dataset, images_dir, out_dir, device, n=6):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    with torch.no_grad():
        for i in idxs:
            fname, gt_box = dataset.samples[i]
            img = cv2.imread(os.path.join(images_dir, fname))
            h, w = img.shape[:2]
            img_t, _ = dataset[i]
            pred = model(img_t.unsqueeze(0).to(device))[0].cpu().tolist()

            vis = img.copy()
            gx1, gy1, gx2, gy2 = yolo_to_pixel(gt_box, w, h)
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)   # ground truth = green
            px1, py1, px2, py2 = yolo_to_pixel(pred, w, h)
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)  # prediction = red
            cv2.imwrite(os.path.join(out_dir, fname), vis)
    print(f"Saved {len(idxs)} comparison images (green=ground truth, red=prediction) to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--out-dir", default="step2_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_ds = SingleFaceDataset(args.images_dir, args.labels_dir, img_size=args.img_size)
    if len(full_ds) < 10:
        raise RuntimeError("Too few single-face images found -- check --images-dir/--labels-dir.")

    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TinySingleBoxNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, targets).item() * imgs.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1}/{args.epochs}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

    visualize(model, full_ds, args.images_dir, args.out_dir, device)

    ckpt_path = os.path.join(args.out_dir, "tiny_single_box_net.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()