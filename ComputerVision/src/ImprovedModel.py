"""
Step 5: Stronger detector — pretrained backbone + aggressive augmentation.

Two changes that fix the "looks slightly down and disappears" problem:

1. DATA AUGMENTATION
   Every training image is randomly transformed:
   - Horizontal flip (faces from either side)
   - Rotation up to ±25° (tilted heads, looking up/down)
   - Brightness / contrast / saturation jitter (lighting changes)
   - Small perspective warp (camera angle variation)
   All box coordinates are transformed along with the image so labels
   stay correct.

2. PRETRAINED MOBILENETV2 BACKBONE
   Instead of learning features from scratch on 1154 images, we start
   from MobileNetV2 weights trained on ImageNet (1.2M images). The
   network already knows how to detect edges, textures, and structures;
   we only teach it "here's what face features look like" on top of that.
   This is called transfer learning and it's why modern detectors train
   well on relatively small datasets.

   MobileNetV2 is specifically designed to be fast enough for real-time
   use on mobile/CPU — important once this runs on a live webcam.

Usage:
    python step5_strong_model.py --images-dir ... --labels-dir ... --epochs 40
"""

import argparse
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# ── constants ─────────────────────────────────────────────────────────────────
GRID_SIZE   = 7
IMG_SIZE    = 224
ANCHORS     = [(0.15, 0.20), (0.45, 0.60)]
NUM_ANCHORS = len(ANCHORS)


# ── label helpers ─────────────────────────────────────────────────────────────
def load_yolo_label(path):
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(list(map(float, parts[1:5])))
    return boxes


def shape_iou(wh, anchor_wh):
    w, h = wh;  aw, ah = anchor_wh
    inter = min(w, aw) * min(h, ah)
    union = w*h + aw*ah - inter
    return inter/union if union else 0.0


# ── augmentation ──────────────────────────────────────────────────────────────
def augment(img_bgr, boxes):
    """
    img_bgr : HxWx3 uint8
    boxes   : list of [xc, yc, w, h] normalized [0,1]
    Returns augmented (img_bgr, boxes) — boxes stay in [0,1] coords.
    """
    h, w = img_bgr.shape[:2]

    # 1. Horizontal flip
    if random.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
        boxes   = [[1.0 - xc, yc, bw, bh] for xc, yc, bw, bh in boxes]

    # 2. Rotation (up to ±25°) — the key fix for looking up/down
    angle = random.uniform(-25, 25)
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    new_boxes = []
    for xc, yc, bw, bh in boxes:
        # rotate all four corners, then refit a new axis-aligned box
        cx, cy = xc*w, yc*h
        half_w, half_h = bw*w/2, bh*h/2
        corners = np.array([
            [cx-half_w, cy-half_h],
            [cx+half_w, cy-half_h],
            [cx+half_w, cy+half_h],
            [cx-half_w, cy+half_h],
        ])
        ones = np.ones((4,1))
        rotated = (M @ np.hstack([corners, ones]).T).T
        x_min, y_min = rotated[:,0].min(), rotated[:,1].min()
        x_max, y_max = rotated[:,0].max(), rotated[:,1].max()
        # clip to image boundaries
        x_min, x_max = np.clip([x_min, x_max], 0, w)
        y_min, y_max = np.clip([y_min, y_max], 0, h)
        if x_max - x_min < 4 or y_max - y_min < 4:
            continue   # box rotated entirely outside frame
        new_boxes.append([
            (x_min+x_max)/(2*w), (y_min+y_max)/(2*h),
            (x_max-x_min)/w,     (y_max-y_min)/h
        ])
    boxes = new_boxes if new_boxes else boxes

    # 3. Color jitter (brightness / contrast / saturation)
    img_bgr = img_bgr.astype(np.float32)
    alpha = random.uniform(0.7, 1.3)   # contrast
    beta  = random.uniform(-30, 30)    # brightness
    img_bgr = np.clip(img_bgr * alpha + beta, 0, 255)
    # saturation jitter in HSV
    img_hsv = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:,:,1] = np.clip(img_hsv[:,:,1] * random.uniform(0.7, 1.3), 0, 255)
    img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return img_bgr.astype(np.uint8), boxes


# ── dataset ───────────────────────────────────────────────────────────────────
class StrongFaceDataset(Dataset):
    def __init__(self, images_dir, labels_dir, augment_data=True,
                 img_size=IMG_SIZE, grid_size=GRID_SIZE):
        self.images_dir   = images_dir
        self.augment_data = augment_data
        self.img_size     = img_size
        self.grid_size    = grid_size

        all_imgs = sorted(f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg','.jpeg','.png')))
        self.samples = []
        for fname in all_imgs:
            stem  = os.path.splitext(fname)[0]
            lpath = os.path.join(labels_dir, stem + '.txt')
            if not os.path.exists(lpath):
                continue
            boxes = load_yolo_label(lpath)
            if boxes:
                self.samples.append((fname, boxes))

        n_faces = sum(len(b) for _,b in self.samples)
        split   = 'train (with augmentation)' if augment_data else 'val'
        print(f"StrongFaceDataset [{split}]: {len(self.samples)} images, {n_faces} faces")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, boxes = self.samples[idx]
        img = cv2.imread(os.path.join(self.images_dir, fname))

        if self.augment_data:
            img, boxes = augment(img, [list(b) for b in boxes])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = torch.from_numpy(img / 255.0).permute(2,0,1).float()
        return img_t, self._encode(boxes)

    def _encode(self, boxes):
        S, A = self.grid_size, NUM_ANCHORS
        target = torch.zeros((S, S, A, 5), dtype=torch.float32)
        for xc, yc, w, h in boxes:
            gi = min(int(xc * S), S-1)
            gj = min(int(yc * S), S-1)
            a  = int(np.argmax([shape_iou((w,h), anc) for anc in ANCHORS]))
            aw, ah = ANCHORS[a]
            tx = xc*S - gi
            ty = yc*S - gj
            tw = np.log(max(w, 1e-6) / aw)
            th = np.log(max(h, 1e-6) / ah)
            target[gj, gi, a, 0]  = 1.0
            target[gj, gi, a, 1:] = torch.tensor([tx, ty, tw, th])
        return target


# ── model: MobileNetV2 backbone + custom detection head ──────────────────────
class StrongFaceNet(nn.Module):
    """
    MobileNetV2 features (pretrained on ImageNet) feed into a lightweight
    detection head. We freeze the early layers and only train the later
    ones + the head — this avoids destroying the useful low-level features
    while still adapting to faces.
    """
    def __init__(self, grid_size=GRID_SIZE, num_anchors=NUM_ANCHORS, pretrained=True):
        super().__init__()
        self.grid_size   = grid_size
        self.num_anchors = num_anchors

        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        # MobileNetV2 features: (B, 1280, 7, 7) for a 224×224 input
        self.backbone = base.features

        # Freeze the first 10 of 19 feature blocks — keep strong low-level
        # features locked, fine-tune the deeper ones
        for i, layer in enumerate(self.backbone):
            if i < 10:
                for p in layer.parameters():
                    p.requires_grad = False

        # Lightweight detection head
        self.head = nn.Sequential(
            nn.Conv2d(1280, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,  num_anchors * 5, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)                  # (B, 1280, 7, 7)
        out  = self.head(feat)                   # (B, A*5, 7, 7)
        B, _, H, W = out.shape
        out  = out.view(B, self.num_anchors, 5, H, W)
        return out.permute(0, 3, 4, 1, 2)        # (B, S, S, A, 5)


# ── loss (same as Step 4) ─────────────────────────────────────────────────────
class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        obj_logit = pred[..., 0]
        box_pred  = torch.cat([torch.sigmoid(pred[..., 1:3]),
                                pred[..., 3:5]], dim=-1)
        obj_mask   = target[..., 0] == 1
        noobj_mask = ~obj_mask
        B = pred.shape[0]

        obj_l   = self.bce(obj_logit[obj_mask],   target[...,0][obj_mask])   if obj_mask.any()   else pred.new_tensor(0.)
        noobj_l = self.bce(obj_logit[noobj_mask], target[...,0][noobj_mask])
        coord_l = self.mse(box_pred[obj_mask],    target[...,1:][obj_mask])  if obj_mask.any()   else pred.new_tensor(0.)

        total = (obj_l + self.lambda_noobj * noobj_l + self.lambda_coord * coord_l) / B
        return total, {
            'obj':   obj_l.item()/B,
            'noobj': noobj_l.item()/B,
            'coord': coord_l.item()/B,
        }


# ── decode + NMS (same logic, self-contained) ─────────────────────────────────
def decode(pred, conf_thresh=0.35):
    S = GRID_SIZE
    obj   = torch.sigmoid(pred[..., 0])
    tx_ty = torch.sigmoid(pred[..., 1:3])
    tw_th = pred[..., 3:5]
    dets  = []
    for gj in range(S):
        for gi in range(S):
            for a in range(NUM_ANCHORS):
                score = obj[gj, gi, a].item()
                if score < conf_thresh:
                    continue
                tx, ty = tx_ty[gj, gi, a].tolist()
                tw, th = tw_th[gj, gi, a].tolist()
                aw, ah = ANCHORS[a]
                xc = (gi + tx) / S
                yc = (gj + ty) / S
                w  = float(np.exp(np.clip(tw, -4, 4))) * aw
                h  = float(np.exp(np.clip(th, -4, 4))) * ah
                dets.append((score, xc, yc, w, h))
    return dets


def iou_xyxy(b1, b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter  = max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1+a2-inter
    return inter/union if union else 0.


def nms(dets, iou_thresh=0.4):
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    while dets:
        best = dets.pop(0); keep.append(best)
        bx1=best[1]-best[3]/2; by1=best[2]-best[4]/2
        bx2=best[1]+best[3]/2; by2=best[2]+best[4]/2
        dets = [d for d in dets
                if iou_xyxy((bx1,by1,bx2,by2),
                            (d[1]-d[3]/2,d[2]-d[4]/2,
                             d[1]+d[3]/2,d[2]+d[4]/2)) <= iou_thresh]
    return keep


# ── visualise ─────────────────────────────────────────────────────────────────
def visualize(model, dataset, images_dir, out_dir, device, n=8, conf_thresh=0.35):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    with torch.no_grad():
        for i in idxs:
            fname, gt_boxes = dataset.samples[i]
            img   = cv2.imread(os.path.join(images_dir, fname))
            h, w  = img.shape[:2]
            img_t, _ = dataset[i]
            pred  = model(img_t.unsqueeze(0).to(device))[0].cpu()
            dets  = nms(decode(pred, conf_thresh=conf_thresh))

            vis = img.copy()
            for (xc,yc,bw,bh) in gt_boxes:
                x1,y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
                x2,y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            for score,xc,yc,bw,bh in dets:
                x1,y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
                x2,y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(vis,f'{score:.2f}',(x1,max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            cv2.imwrite(os.path.join(out_dir, fname), vis)
    print(f"Saved {len(idxs)} comparison images to {out_dir}/  (green=GT, red=pred)")


# ── training loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--labels-dir', required=True)
    parser.add_argument('--epochs',     type=int,   default=40)
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--val-frac',   type=float, default=0.1)
    parser.add_argument('--conf-thresh',type=float, default=0.35)
    parser.add_argument('--out-dir',    default='step5_output')
    parser.add_argument('--no-pretrain',action='store_true',
                        help='Skip ImageNet weights (slower convergence, for reference)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # datasets — augmentation ON for train, OFF for val
    full_ds = StrongFaceDataset(args.images_dir, args.labels_dir, augment_data=False)
    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(full_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # build separate dataset objects so train gets augmentation, val doesn't
    train_ds = StrongFaceDataset(args.images_dir, args.labels_dir, augment_data=True)
    val_ds   = StrongFaceDataset(args.images_dir, args.labels_dir, augment_data=False)
    # use the same split indices
    train_ds = torch.utils.data.Subset(train_ds, list(train_idx.indices))
    val_ds   = torch.utils.data.Subset(val_ds,   list(val_idx.indices))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = StrongFaceNet(pretrained=not args.no_pretrain).to(device)
    criterion = DetectionLoss()

    # two learning-rate groups: lower LR for backbone (don't destroy pretrained weights),
    # higher LR for the fresh detection head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params,     'lr': args.lr},
    ])
    # cosine annealing: smoothly reduces LR so the model doesn't overshoot at the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val  = float('inf')
    ckpt_path = os.path.join(args.out_dir, 'strong_face_net.pt')
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = {'total':0., 'obj':0., 'noobj':0., 'coord':0.}
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss, parts = criterion(preds, targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running['total'] += loss.item()
            for k in ('obj','noobj','coord'): running[k] += parts[k]
        n = len(train_loader)

        model.eval(); val_total = 0.
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                loss, _ = criterion(model(imgs), targets)
                val_total += loss.item()
        val_loss = val_total / max(len(val_loader),1)

        lr_now = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train={running['total']/n:.3f}  "
              f"obj={running['obj']/n:.3f}  coord={running['coord']/n:.3f}  "
              f"val={val_loss:.3f}  lr={lr_now:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)

        scheduler.step()

    print(f"\nBest val loss: {best_val:.4f}  — checkpoint: {ckpt_path}")

    # load best checkpoint for visualisation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # use the underlying dataset for visualisation (not Subset)
    vis_ds = StrongFaceDataset(args.images_dir, args.labels_dir, augment_data=False)
    visualize(model, vis_ds, args.images_dir, args.out_dir,
              device, conf_thresh=args.conf_thresh)


if __name__ == '__main__':
    main()