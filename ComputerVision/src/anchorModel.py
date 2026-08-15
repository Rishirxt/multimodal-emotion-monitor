"""
Step 4: Anchors -- giving each cell more than one shape to work with.

Step 3 gave each grid cell exactly one box to predict. That broke down
exactly where you'd expect: on the crowd photo, two real faces close
together ended up collapsed into a single surviving detection, because
each cell (and then NMS) only had one shape to offer.

This step gives each cell a small set of pre-defined box "shape templates"
(anchors) -- e.g. one shaped like a typical small/crowd face, one shaped
like a typical large/portrait face. Each anchor gets its own independent
[objectness, box] prediction, so one cell can now represent up to
NUM_ANCHORS different faces, distinguished by which template shape best
matches each one. This is exactly how YOLOv2 (2017) extended YOLOv1 --
same core idea, one more degree of freedom.

How a ground-truth face is assigned during training:
  1. Find the cell containing its center (same rule as Step 3).
  2. Within that cell, find whichever anchor template's SHAPE (just width
     and height, ignoring position) overlaps the face's true shape the
     most -- that anchor is the one trained to predict this face.
Width/height are now predicted as a multiplier on the anchor's own size,
in log-space (tw = log(w / anchor_w)) rather than as a raw fraction of the
image -- the network only has to learn "how do I adjust THIS template,"
not invent a size from nothing.

Usage:
    python step4_anchor_model.py --images-dir ... --labels-dir ... --epochs 25
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


GRID_SIZE = 7
IMG_SIZE = 224

# Two shape templates, as (width, height) fractions of the whole image.
# Picked from this dataset's own stats: most faces are large/centered
# (avg ~0.44 x 0.60), but a meaningful tail are small crowd faces.
# A more rigorous version would run k-means on your actual box sizes
# instead of hand-picking these -- ask if you want that script.
ANCHORS = [(0.15, 0.20), (0.45, 0.60)]
NUM_ANCHORS = len(ANCHORS)


def load_yolo_label(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append(list(map(float, parts[1:5])))
    return boxes


def shape_iou(wh, anchor_wh):
    """IoU between two boxes of the given (w,h), as if both were centered
    at the same point -- compares SHAPE only, ignoring position."""
    w, h = wh
    aw, ah = anchor_wh
    inter = min(w, aw) * min(h, ah)
    union = w * h + aw * ah - inter
    return inter / union if union > 0 else 0.0


class AnchorFaceDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=IMG_SIZE, grid_size=GRID_SIZE, anchors=ANCHORS):
        self.images_dir = images_dir
        self.img_size = img_size
        self.grid_size = grid_size
        self.anchors = anchors

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
        print(f"AnchorFaceDataset: {len(self.samples)} images, {n_faces} total faces.")

        collisions = 0
        for _, boxes in self.samples:
            seen = set()
            for xc, yc, w, h in boxes:
                gi = min(int(xc * grid_size), grid_size - 1)
                gj = min(int(yc * grid_size), grid_size - 1)
                a = int(np.argmax([shape_iou((w, h), anc) for anc in anchors]))
                key = (gi, gj, a)
                if key in seen:
                    collisions += 1
                seen.add(key)
        if collisions:
            print(f"Note: {collisions} face(s) still share the same (cell, anchor) slot as "
                  f"another face -- with only {NUM_ANCHORS} anchors there's a limit to how much "
                  f"this helps. Better-tuned anchors or a finer grid narrows this further.")
        else:
            print("No (cell, anchor) collisions in this dataset -- every face got its own slot.")

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
        """Returns target tensor (S, S, A, 5): [obj, tx, ty, tw, th]."""
        S, A = self.grid_size, len(self.anchors)
        target = torch.zeros((S, S, A, 5), dtype=torch.float32)
        for xc, yc, w, h in boxes:
            gi = min(int(xc * S), S - 1)
            gj = min(int(yc * S), S - 1)
            a = int(np.argmax([shape_iou((w, h), anc) for anc in self.anchors]))
            aw, ah = self.anchors[a]

            tx = xc * S - gi
            ty = yc * S - gj
            tw = np.log(max(w, 1e-6) / aw)
            th = np.log(max(h, 1e-6) / ah)

            target[gj, gi, a, 0] = 1.0
            target[gj, gi, a, 1:] = torch.tensor([tx, ty, tw, th])
        return target


class TinyAnchorFaceNet(nn.Module):
    def __init__(self, grid_size=GRID_SIZE, num_anchors=NUM_ANCHORS):
        super().__init__()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Conv2d(128, num_anchors * 5, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)             # (B, 128, S, S)
        out = self.head(feat)               # (B, A*5, S, S)
        B, _, H, W = out.shape
        out = out.view(B, self.num_anchors, 5, H, W)
        return out.permute(0, 3, 4, 1, 2)   # (B, S, S, A, 5)


class AnchorLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        obj_logit = pred[..., 0]
        # tx,ty are positions within a cell -> squash to [0,1). tw,th are
        # log-ratios against the anchor -> can be any sign, left raw.
        box_pred = torch.cat([torch.sigmoid(pred[..., 1:3]), pred[..., 3:5]], dim=-1)

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


def decode(pred, anchors=ANCHORS, grid_size=GRID_SIZE, conf_thresh=0.3):
    """pred: (S,S,A,5) raw model output for ONE image.
    Returns a list of (score, xc, yc, w, h) in normalized [0,1] coords."""
    S, A = grid_size, len(anchors)
    obj = torch.sigmoid(pred[..., 0])
    tx_ty = torch.sigmoid(pred[..., 1:3])
    tw_th = pred[..., 3:5]

    detections = []
    for gj in range(S):
        for gi in range(S):
            for a in range(A):
                score = obj[gj, gi, a].item()
                if score < conf_thresh:
                    continue
                tx, ty = tx_ty[gj, gi, a].tolist()
                tw, th = tw_th[gj, gi, a].tolist()
                aw, ah = anchors[a]
                xc = (gi + tx) / S
                yc = (gj + ty) / S
                w = float(np.exp(tw)) * aw
                h = float(np.exp(th)) * ah
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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--out-dir", default="step4_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_ds = AnchorFaceDataset(args.images_dir, args.labels_dir)
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TinyAnchorFaceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = AnchorLoss()

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

    ckpt_path = os.path.join(args.out_dir, "tiny_anchor_face_net.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()