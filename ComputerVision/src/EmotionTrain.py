"""
Emotion Phase 2: Train a 7-class emotion classifier on FER2013.

Architecture: MobileNetV2 pretrained backbone + 7-class head.
Same transfer-learning idea as the face detector (Step 5) — the backbone
already knows face textures from ImageNet; we just teach it to distinguish
emotional expressions on top of that.

Key design decisions driven by the data exploration output:
  - 16.5x class imbalance → weighted cross-entropy loss
  - 48×48 input → resized to 96×96 (better than 224 for tiny faces,
    still fast enough for CPU inference)
  - Augmentation: horizontal flip + small rotation + brightness jitter.
    No vertical flip — an upside-down face is not a face expression.

Usage:
    python emotion_train.py --data-dir C:/path/to/EmotionData
    python emotion_train.py --data-dir ... --epochs 30 --batch-size 64
"""

import argparse
import os
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ── constants ─────────────────────────────────────────────────────────────────
EMOTIONS   = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)
LABEL2IDX  = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL  = {i: e for i, e in enumerate(EMOTIONS)}
IMG_SIZE   = 96   # upsample from 48 — gives the model more spatial signal


# ── augmentation ──────────────────────────────────────────────────────────────
def augment(img_bgr):
    # horizontal flip — happy face flipped is still happy
    if random.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)

    # small rotation — up to ±15° (emotions are robust to slight tilts)
    angle = random.uniform(-15, 15)
    M     = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (IMG_SIZE, IMG_SIZE),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # brightness + contrast jitter
    alpha = random.uniform(0.75, 1.25)
    beta  = random.uniform(-20, 20)
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img_bgr


# ── dataset ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, split_dir, augment_data=False):
        self.augment_data = augment_data
        self.samples = []

        for emotion in EMOTIONS:
            folder = os.path.join(split_dir, emotion)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(
                        (os.path.join(folder, fname), LABEL2IDX[emotion])
                    )

        counts = Counter(label for _, label in self.samples)
        split  = 'train+aug' if augment_data else 'val/test'
        print(f"EmotionDataset [{split}]: {len(self.samples)} images")
        for emotion in EMOTIONS:
            print(f"  {emotion:10s}: {counts[LABEL2IDX[emotion]]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = cv2.imread(fpath)
        if img is None:
            img = np.zeros((48, 48, 3), dtype=np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.augment_data:
            img = augment(img)

        # BGR → RGB, HWC → CHW, [0,255] → [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()

        # ImageNet normalisation — matches what MobileNetV2 was pretrained with
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        return img_t, label


# ── model ─────────────────────────────────────────────────────────────────────
class EmotionClassifier(nn.Module):
    """
    MobileNetV2 backbone → 7-class emotion head.

    We keep the backbone frozen for the first few epochs (warmup) so the
    random head doesn't destroy the pretrained weights, then unfreeze the
    deeper layers for fine-tuning. Same strategy that worked for the face
    detector in Step 5.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.backbone = base.features  # outputs (B, 1280, 3, 3) for 96×96 input

        # freeze everything initially — we unfreeze after warmup
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, 1280, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def unfreeze_top(self, n_blocks=5):
        """Unfreeze the last n_blocks of the backbone for fine-tuning."""
        blocks = list(self.backbone.children())
        for block in blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x)    # (B, 1280, H, W)
        feat = self.pool(feat)     # (B, 1280, 1, 1)
        return self.head(feat)     # (B, 7)


# ── class weights for imbalanced data ─────────────────────────────────────────
def compute_class_weights(dataset, device):
    counts = Counter(label for _, label in dataset.samples)
    total  = len(dataset)
    # weight = total / (num_classes * count) — rare classes get higher weight
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(device)
    print("\nClass weights (higher = rarer class gets more attention):")
    for i, e in enumerate(EMOTIONS):
        print(f"  {e:10s}: {weights[i].item():.3f}")
    return weights


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    per_class_correct = Counter()
    per_class_total   = Counter()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            for p, l in zip(preds.cpu(), labels.cpu()):
                per_class_total[l.item()]   += 1
                per_class_correct[l.item()] += (p == l).item()

    acc = 100.0 * correct / total
    return total_loss / total, acc, per_class_correct, per_class_total


# ── training loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   required=True)
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--warmup',     type=int,   default=5,
                        help='Epochs to train head only before unfreezing backbone')
    parser.add_argument('--out-dir',    default='emotion_output')
    parser.add_argument('--no-pretrain', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # datasets
    train_ds = EmotionDataset(os.path.join(args.data_dir, 'train'), augment_data=True)
    test_ds  = EmotionDataset(os.path.join(args.data_dir, 'test'),  augment_data=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # model + weighted loss
    model   = EmotionClassifier(pretrained=not args.no_pretrain).to(device)
    weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # only the head trains during warmup
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc  = 0.0
    ckpt_path = os.path.join(args.out_dir, 'emotion_model.pt')

    print(f"\nWarmup ({args.warmup} epochs): training head only, backbone frozen")

    for epoch in range(args.epochs):

        # after warmup: unfreeze top backbone layers + add them to optimizer
        if epoch == args.warmup:
            print(f"\nEpoch {epoch+1}: unfreezing top 5 backbone blocks for fine-tuning")
            model.unfreeze_top(n_blocks=5)
            optimizer.add_param_group(
                {'params': [p for p in model.backbone.parameters()
                            if p.requires_grad],
                 'lr': args.lr * 0.1}   # lower LR for pretrained weights
            )

        # train
        model.train()
        running_loss, correct, total = 0., 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct      += (logits.argmax(1) == labels).sum().item()
            total        += imgs.size(0)

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total

        # evaluate
        val_loss, val_acc, pc_correct, pc_total = evaluate(
            model, test_loader, criterion, device
        )

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.3f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.1f}%  "
              f"lr={lr_now:.5f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':      epoch + 1,
                'state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'emotions':   EMOTIONS,
            }, ckpt_path)
            print(f"  ✓ new best val_acc={val_acc:.1f}%  → saved {ckpt_path}")

        scheduler.step()

    # final per-class breakdown
    print(f"\n{'='*50}")
    print(f"Best val accuracy: {best_acc:.1f}%")
    print(f"Checkpoint: {ckpt_path}")
    print(f"\nPer-class accuracy on test set (final epoch):")
    for i, emo in enumerate(EMOTIONS):
        n   = pc_total[i]
        acc = 100.0 * pc_correct[i] / n if n else 0.0
        bar = '█' * int(acc / 5)
        print(f"  {emo:10s}: {acc:5.1f}%  {bar}")
    print(f"{'='*50}")
    print("\nNext: run webcam_emotion.py to see detection + emotion live.")


if __name__ == '__main__':
    main()