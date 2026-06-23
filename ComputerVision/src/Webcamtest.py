"""
webcam_test.py  —  live face detection using the Step 5 stronger model.

Self-contained: defines the model architecture inline.
Works with the checkpoint saved by step5_strong_model.py (strong_face_net.pt).

Usage:
    python webcam_test.py --checkpoint strong_face_net.pt

Controls:
    Q  /  ESC   →  quit
    +  /  -     →  raise / lower confidence threshold live
"""

import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ── constants (must match training) ──────────────────────────────────────────
IMG_SIZE    = 224
GRID_SIZE   = 7
ANCHORS     = [(0.15, 0.20), (0.45, 0.60)]
NUM_ANCHORS = len(ANCHORS)


# ── model ─────────────────────────────────────────────────────────────────────
class StrongFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        base           = models.mobilenet_v2(weights=None)
        self.backbone  = base.features
        self.head      = nn.Sequential(
            nn.Conv2d(1280, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, NUM_ANCHORS * 5, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        out  = self.head(feat)
        B, _, H, W = out.shape
        out  = out.view(B, NUM_ANCHORS, 5, H, W)
        return out.permute(0, 3, 4, 1, 2)


# ── decode + NMS ──────────────────────────────────────────────────────────────
def decode(pred, conf_thresh=0.35):
    S     = GRID_SIZE
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
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter  = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter/union if union else 0.


def nms(dets, iou_thresh=0.4):
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    while dets:
        best = dets.pop(0); keep.append(best)
        bx1 = best[1]-best[3]/2; by1 = best[2]-best[4]/2
        bx2 = best[1]+best[3]/2; by2 = best[2]+best[4]/2
        dets = [d for d in dets
                if iou_xyxy((bx1,by1,bx2,by2),
                            (d[1]-d[3]/2, d[2]-d[4]/2,
                             d[1]+d[3]/2, d[2]+d[4]/2)) <= iou_thresh]
    return keep


# ── draw ─────────────────────────────────────────────────────────────────────
def draw(frame, dets, fps, conf_thresh):
    h, w = frame.shape[:2]
    for score, xc, yc, bw, bh in dets:
        x1 = int(max(0, (xc-bw/2)*w)); y1 = int(max(0, (yc-bh/2)*h))
        x2 = int(min(w-1, (xc+bw/2)*w)); y2 = int(min(h-1, (yc+bh/2)*h))
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0,y1-lh-6)), (x1+lw+4, y1), (0,255,0), -1)
        cv2.putText(frame, label, (x1+2, max(lh, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:4.1f}",
                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {len(dets)}",
                (10,56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Thresh: {conf_thresh:.2f}  (+/-)",
                (10,84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q / ESC to quit",
                (10, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to strong_face_net.pt')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--conf',   type=float, default=0.35,
                        help='Starting confidence threshold (default 0.35 — lower than before)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = StrongFaceNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded  →  {args.checkpoint}  on {device}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}. Try --camera 1")
    print("Camera open. Q / ESC to quit  |  +/- to adjust threshold")

    conf_thresh = args.conf
    prev_time   = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        tensor  = torch.from_numpy(resized/255.0).permute(2,0,1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor)[0].cpu()

        dets = nms(decode(pred, conf_thresh=conf_thresh))

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw(frame, dets, fps, conf_thresh)
        cv2.imshow("AffectFusion — Strong Face Detection  (Q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
            print(f"Threshold → {conf_thresh:.2f}")
        elif key == ord('-'):
            conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
            print(f"Threshold → {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()