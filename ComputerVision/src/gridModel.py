"""
Step 3: A grid of small detectors instead of one big one.

Step 2's model looked at the whole image and guessed ONE box. This script
divides the image into an S x S grid of cells. Each cell only looks at its
own patch of the feature map and makes its own independent prediction:
  - objectness: "is the CENTER of a face located inside me?" (yes/no)
  - if yes: a box, with its position expressed as an offset from MY OWN
    cell (not the whole image) -- a much smaller, easier question.

This fixes both problems Step 2 had:
  1. Multiple faces -> multiple cells can each claim a face. One global
     head physically cannot do this; a grid of S*S independent heads can.
  2. Imprecise localization -> each cell only answers "is a face centered
     near ME, and exactly where" instead of "where in this whole image is
     THE face."

Ground-truth assignment: for every labeled face, the cell containing its
center point is the one cell trained to predict it; every other cell is
trained to predict "no object here." This is the core idea behind YOLOv1
(2015) -- we're rebuilding it from scratch, one piece at a time.

Known limitation (intentionally not fixed yet): if two faces' centers land
in the SAME cell, this design can only keep one of them. That's exactly
what Step 4 (anchors -- multiple boxes per cell) exists to fix.

Usage:
    python step3_grid_model.py --images-dir ... --labels-dir ... --epochs 20
"""

import argparse
import os
import random

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


GRID_SIZE = 7   # try 5 or 9 later and see how results change
IMG_SIZE = 224  # must stay divisible by 32 given the backbone below (224/32 = 7 = GRID_SIZE)


def load_yolo_label(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append(list(map(float, parts[1:5])))  # xc, yc, w, h (normalized)
    return boxes


class GridFaceDataset(Dataset):
    """Unlike Step 2, this keeps EVERY image, including multi-face ones --
    a grid has room for more than one face per image."""

    def __init__(self, images_dir, labels_dir, img_size=IMG_SIZE, grid_size=GRID_SIZE):
        self.images_dir = images_dir
        self.img_size = img_size
        self.grid_size = grid_size

        all_images = sorted(f for f in os.listdir(images_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.samples = []
        for fname in all_images:
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            if not os.path.exists(label_path):
                continue
            boxes = load_yolo_label(label_path)
            if boxes:
                self.samples.append((fname, boxes))

        n_faces = sum(len(b) for _, b in self.samples)
        print(f"GridFaceDataset: {len(self.samples)} images, {n_faces} total faces.")

        collisions = 0
        for _, boxes in self.samples:
            seen = set()
            for xc, yc, w, h in boxes:
                cell = (min(int(xc * grid_size), grid_size - 1),
                        min(int(yc * grid_size), grid_size - 1))
                if cell in seen:
                    collisions += 1
                seen.add(cell)
        if collisions:
            print(f"Note: {collisions} face(s) share a grid cell with another face in the "
                  f"same image -- one-box-per-cell can only keep one of them. This is the "
                  f"gap Step 4 (anchors) closes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, boxes = self.samples[idx]
        img = cv2.imread(os.path.join(self.images_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        target = self.encode(boxes)
        return img_t, target

    def encode(self, boxes):
        """boxes: list of (xc, yc, w, h), all normalized [0,1].
        Returns target tensor (S, S, 5): [obj, tx, ty, w, h].
        tx, ty are the face center's position WITHIN its cell (so in [0,1)).
        w, h pass straight through as fractions of the whole image -- same
        units your label files already use, no extra encoding needed."""
        S = self.grid_size
        target = torch.zeros((S, S, 5), dtype=torch.float32)
        for xc, yc, w, h in boxes:
            gi = min(int(xc * S), S - 1)
            gj = min(int(yc * S), S - 1)
            tx = xc * S - gi
            ty = yc * S - gj
            target[gj, gi, 0] = 1.0
            target[gj, gi, 1:] = torch.tensor([tx, ty, w, h])
        return target


class TinyGridFaceNet(nn.Module):
    """Same backbone family as Step 2, but the head now produces a full
    grid of predictions instead of a single vector."""

    def __init__(self, grid_size=GRID_SIZE):
        super().__init__()
        self.grid_size = grid_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),     # 224->112
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 112->56
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),    # 56->28
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 14->7
        )
        self.head = nn.Conv2d(128, 5, kernel_size=1)  # per-cell: [obj, tx, ty, w, h]

    def forward(self, x):
        feat = self.backbone(x)         # (B, 128, S, S)
        out = self.head(feat)           # (B, 5, S, S)
        return out.permute(0, 2, 3, 1)  # (B, S, S, 5) -- matches target layout


class GridLoss(nn.Module):
    """obj/no-obj classification (BCE) + box regression (MSE, object cells only).
    no-object cells vastly outnumber object cells in any one image, so their
    loss contribution is down-weighted (lambda_noobj) -- otherwise the easiest
    way to minimize loss is to just always predict 'no object' everywhere."""

    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        obj_logit = pred[..., 0]
        box_pred = torch.sigmoid(pred[..., 1:])  # constrain tx,ty,w,h to [0,1], same as targets

        obj_mask = target[..., 0] == 1
        noobj_mask = ~obj_mask
        batch_size = pred.shape[0]

        obj_loss = self.bce(obj_logit[obj_mask], target[..., 0][obj_mask]) if obj_mask.any() else pred.new_tensor(0.0)
        noobj_loss = self.bce(obj_logit[noobj_mask], target[..., 0][noobj_mask])
        coord_loss = self.mse(box_pred[obj_mask], target[..., 1:][obj_mask]) if obj_mask.any() else pred.new_tensor(0.0)

        total = (obj_loss + self.lambda_noobj * noobj_loss + self.lambda_coord * coord_loss) / batch_size
        return total, {
            "obj": obj_loss.item() / batch_size,
            "noobj": noobj_loss.item() / batch_size,
            "coord": coord_loss.item() / batch_size,
        }


def decode(pred, grid_size=GRID_SIZE, conf_thresh=0.3):
    """pred: (S,S,5) raw model output for ONE image.
    Returns a list of (score, xc, yc, w, h) in normalized [0,1] coords."""
    obj = torch.sigmoid(pred[..., 0])
    box = torch.sigmoid(pred[..., 1:])
    detections = []
    for gj in range(grid_size):
        for gi in range(grid_size):
            score = obj[gj, gi].item()
            if score < conf_thresh:
                continue
            tx, ty, w, h = box[gj, gi].tolist()
            xc = (gi + tx) / grid_size
            yc = (gj + ty) / grid_size
            detections.append((score, xc, yc, w, h))
    return detections


def to_xyxy(xc, yc, w, h):
    return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def box_iou_xyxy(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms(detections, iou_thresh=0.4):
    """Greedy NMS: keep the highest-scoring box, drop anything that overlaps
    it too much, repeat. Needed because nearby cells can both fire weakly
    for the same face."""
    detections = sorted(detections, key=lambda d: -d[0])
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        best_box = to_xyxy(*best[1:])
        detections = [d for d in detections if box_iou_xyxy(best_box, to_xyxy(*d[1:])) <= iou_thresh]
    return keep


def visualize(model, dataset, images_dir, out_dir, device, n=8, conf_thresh=0.3, nms_thresh=0.4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    with torch.no_grad():
        for i in idxs:
            fname, gt_boxes = dataset.samples[i]
            img = cv2.imread(os.path.join(images_dir, fname))
            h, w = img.shape[:2]
            img_t, _ = dataset[i]
            pred = model(img_t.unsqueeze(0).to(device))[0].cpu()
            dets = nms(decode(pred, conf_thresh=conf_thresh), iou_thresh=nms_thresh)

            vis = img.copy()
            for (xc, yc, bw, bh) in gt_boxes:
                x1, y1, x2, y2 = to_xyxy(xc, yc, bw, bh)
                cv2.rectangle(vis, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)
            for score, xc, yc, bw, bh in dets:
                x1, y1, x2, y2 = to_xyxy(xc, yc, bw, bh)
                p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
                cv2.rectangle(vis, p1, p2, (0, 0, 255), 2)
                cv2.putText(vis, f"{score:.2f}", (p1[0], max(0, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out_dir, fname), vis)
    print(f"Saved {len(idxs)} comparison images (green=ground truth, red=prediction) to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--out-dir", default="step3_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_ds = GridFaceDataset(args.images_dir, args.labels_dir)
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TinyGridFaceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = GridLoss()

    for epoch in range(args.epochs):
        model.train()
        running = {"total": 0.0, "obj": 0.0, "noobj": 0.0, "coord": 0.0}
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss, parts = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running["total"] += loss.item()
            for k in ("obj", "noobj", "coord"):
                running[k] += parts[k]
        n = len(train_loader)
        train_msg = " ".join(f"{k}={v/n:.4f}" for k, v in running.items())

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                val_total += loss.item()
        val_loss = val_total / max(len(val_loader), 1)

        print(f"Epoch {epoch+1}/{args.epochs}  train: {train_msg}  val_total={val_loss:.4f}")

    visualize(model, full_ds, args.images_dir, args.out_dir, device, conf_thresh=args.conf_thresh)

    ckpt_path = os.path.join(args.out_dir, "tiny_grid_face_net.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()