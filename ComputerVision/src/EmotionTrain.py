"""
Emotion Phase 3: Retrain 7-class emotion classifier with stronger class-balancing
to fix sad bias and improve neutral prediction.

Key changes vs Phase 2:
  - Oversample BOTH disgust AND neutral (not just disgust) to reduce sad dominance
  - Linear inverse-frequency class weights (instead of sqrt) for stronger correction
  - Extra 1.4x weight multiplier on neutral class
  - Slightly higher label smoothing (0.12) to prevent overconfidence on sad
  - neutral_bias field saved in checkpoint for inference-time calibration

Usage:
    python EmotionTrain.py --data-dir C:/Users/Admin/OneDrive/Desktop/CV/ComputerVision/EmotionData --epochs 20 --batch-size 64
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
EMOTIONS    = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)
LABEL2IDX   = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL   = {i: e for i, e in enumerate(EMOTIONS)}
IMG_SIZE    = 96   # upsample from 48 for spatial fidelity


# ── enhanced augmentation ─────────────────────────────────────────────────────
def augment(img_bgr):
    # 1. Random horizontal flip
    if random.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)

    # 2. Random rotation (±20°)
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (IMG_SIZE, IMG_SIZE),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # 3. Brightness + Contrast jitter
    alpha = random.uniform(0.70, 1.30)
    beta  = random.uniform(-25, 25)
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 4. Color jitter (HSV shifts)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. Random grayscale conversion (simulates lighting / gray webcams)
    if random.random() < 0.2:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 6. Random Gaussian blur (simulates soft/blurry webcam crops)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img_bgr = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    # 7. Random small erasing / occlusion patch (p=0.3)
    if random.random() < 0.3:
        pw, ph = random.randint(10, 24), random.randint(10, 24)
        px, py = random.randint(0, IMG_SIZE - pw), random.randint(0, IMG_SIZE - ph)
        color  = random.randint(0, 255)
        img_bgr[py:py+ph, px:px+pw] = color

    return img_bgr


# ── dataset ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, split_dir, augment_data=False,
                 min_disgust_count=1000, min_neutral_count=2000):
        self.augment_data = augment_data
        self.samples = []

        raw_samples = []
        for emotion in EMOTIONS:
            folder = os.path.join(split_dir, emotion)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    raw_samples.append((os.path.join(folder, fname), LABEL2IDX[emotion]))

        # ── Oversample disgust (was collapsing) ──────────────────────────────
        disgust_idx     = LABEL2IDX['disgust']
        disgust_samples = [s for s in raw_samples if s[1] == disgust_idx]
        other_samples   = [s for s in raw_samples if s[1] != disgust_idx]

        if augment_data and len(disgust_samples) > 0 and len(disgust_samples) < min_disgust_count:
            multiplier = (min_disgust_count // len(disgust_samples)) + 1
            disgust_samples = (disgust_samples * multiplier)[:min_disgust_count]

        # ── Oversample neutral to fight sad-dominance bias ────────────────────
        # Neutral is typically underrepresented relative to sad in FER datasets;
        # boosting its count forces the model to see more "nothing happening" faces.
        neutral_idx     = LABEL2IDX['neutral']
        neutral_samples = [s for s in other_samples if s[1] == neutral_idx]
        rest_samples    = [s for s in other_samples if s[1] != neutral_idx]

        if augment_data and len(neutral_samples) > 0 and len(neutral_samples) < min_neutral_count:
            multiplier = (min_neutral_count // len(neutral_samples)) + 1
            neutral_samples = (neutral_samples * multiplier)[:min_neutral_count]

        self.samples = disgust_samples + neutral_samples + rest_samples
        random.shuffle(self.samples)

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

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        return img_t, label


# ── model architecture ────────────────────────────────────────────────────────
class EmotionClassifier(nn.Module):
    """
    MobileNetV2 backbone → 1280 → 512 → 128 → 7-class head.
    Includes BatchNorm and Layered Dropout for regularized feature learning.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.backbone = base.features

        # Freeze backbone parameters initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes)
        )

    def unfreeze_all(self):
        """Unfreeze full backbone for end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat)
        return self.head(feat)


# ── class weights ─────────────────────────────────────────────────────────────
NEUTRAL_WEIGHT_BOOST = 1.4   # extra multiplier on neutral to push model away from sad

def compute_smoothed_class_weights(dataset, device):
    counts = Counter(label for _, label in dataset.samples)
    max_c  = max(counts.values())

    # Linear inverse-frequency weighting (stronger correction than sqrt):
    # w_i = max_count / count_i — directly proportional to under-representation
    raw_weights = [max_c / max(counts[i], 1) for i in range(NUM_CLASSES)]

    # Apply an extra explicit boost to neutral to counteract sad bias
    neutral_idx = LABEL2IDX['neutral']
    raw_weights[neutral_idx] *= NEUTRAL_WEIGHT_BOOST

    mean_w  = np.mean(raw_weights)
    weights = torch.tensor([w / mean_w for w in raw_weights], dtype=torch.float32).to(device)

    print("\nClass Weights (linear inverse-freq, neutral boosted):")
    for i, e in enumerate(EMOTIONS):
        print(f"  {e:10s}: {weights[i].item():.3f}")
    return weights


# ── evaluation & confusion matrix ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    all_logits  = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_logits.append(logits.cpu())

            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                conf_matrix[l, p] += 1

    acc = 100.0 * correct / total
    avg_logits = torch.cat(all_logits, dim=0).mean(dim=0)
    return total_loss / total, acc, conf_matrix, avg_logits


# ── training loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    default_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument('--data-dir',   default=default_data_dir)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--warmup',     type=int,   default=3, help='Warmup epochs for head only')
    parser.add_argument('--out-dir',    default='Models')
    parser.add_argument('--no-pretrain', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_ds = EmotionDataset(os.path.join(args.data_dir, 'train'), augment_data=True)
    test_ds  = EmotionDataset(os.path.join(args.data_dir, 'test'),  augment_data=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Model + Smoothed Class Weights + Label Smoothing
    model   = EmotionClassifier(pretrained=not args.no_pretrain).to(device)
    weights = compute_smoothed_class_weights(train_ds, device)
    # label_smoothing=0.12 softens overconfidence; helps stop model from
    # collapsing hard onto 'sad' when it's uncertain
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.12)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc    = 0.0
    best_priors = None
    ckpt_path   = os.path.join(args.out_dir, 'emotion_model.pt')

    print(f"\nWarmup ({args.warmup} epochs): Head training only...")

    for epoch in range(args.epochs):

        if epoch == args.warmup:
            print(f"\nEpoch {epoch+1}: Unfreezing entire backbone for fine-tuning...")
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                [
                    {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
                    {'params': model.head.parameters(),     'lr': args.lr}
                ],
                weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup)

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

        val_loss, val_acc, conf_mat, avg_logits = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{args.epochs}  "
              f"train_loss={train_loss:.3f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.1f}%  "
              f"lr={lr_now:.5f}")

        if val_acc > best_acc:
            best_acc    = val_acc
            best_priors = avg_logits.tolist()

            # neutral_bias: a small positive offset added to the neutral logit
            # at inference time to further offset any residual sad-bias.
            # Set to 0.0 here — adjust in WebcamEmotion.py NEUTRAL_LOGIT_BIAS
            # if live predictions still skew sad after retraining.
            neutral_bias = 0.0

            torch.save({
                'epoch':        epoch + 1,
                'state_dict':   model.state_dict(),
                'val_acc':      val_acc,
                'emotions':     EMOTIONS,
                'emo_prior':    best_priors,
                'neutral_bias': neutral_bias,
            }, ckpt_path)
            print(f"  ✓ new best val_acc={val_acc:.1f}% → saved {ckpt_path}")

    # Final Summary & Confusion Matrix
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Validation Accuracy: {best_acc:.1f}%")
    print(f"Checkpoint saved to: {ckpt_path}")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    header = "          " + "".join([f"{e[:5]:>7}" for e in EMOTIONS])
    print(header)

    final_val_loss, final_val_acc, conf_mat, _ = evaluate(model, test_loader, criterion, device)
    for i, emo in enumerate(EMOTIONS):
        row_str = f"{emo:10s}" + "".join([f"{conf_mat[i, j]:7d}" for j in range(NUM_CLASSES)])
        total_class = conf_mat[i].sum()
        acc = 100.0 * conf_mat[i, i] / total_class if total_class > 0 else 0.0
        print(f"{row_str}  (Acc: {acc:5.1f}%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()