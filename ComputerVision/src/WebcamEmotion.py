"""
run_camera.py — opens camera, detects face, classifies emotion, prints JSON.

Run with:
    python run_camera.py --face Models/strong_face_net.pt --emotion Models/emotion_model.pt

Press Q to quit.
"""

import argparse, json, math, os, time
import cv2, numpy as np
import torch, torch.nn as nn
from torchvision import models

EMOTIONS     = ['angry','disgust','fear','happy','neutral','sad','surprise']
FACE_ANCHORS = [(0.15,0.20),(0.45,0.60)]
EMO_MEAN     = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
EMO_STD      = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
EMO_COLORS   = {'angry':(0,0,220),'disgust':(0,140,0),'fear':(140,0,140),
                'happy':(0,210,0),'neutral':(180,180,180),'sad':(200,90,0),
                'surprise':(0,200,220)}
ENERGY       = {'happy':1.0,'surprise':0.95,'angry':0.85,'fear':0.75,
                'disgust':0.65,'sad':0.40,'neutral':0.10}

EMO_TEMP           = 1.2   # temperature scaling softener
# Extra logit offset applied to the neutral class at inference time.
# Increase this (e.g. 0.3–0.8) if the model still skews sad after retraining.
NEUTRAL_LOGIT_BIAS = 0.0

# ── models ────────────────────────────────────────────────────────────────────
class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None).features
        self.head = nn.Sequential(
            nn.Conv2d(1280,256,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, len(FACE_ANCHORS)*5, 1))
    def forward(self, x):
        o = self.head(self.backbone(x)); B,_,H,W = o.shape
        return o.view(B, len(FACE_ANCHORS), 5, H, W).permute(0,3,4,1,2)

class EmoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None).features
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
            nn.Linear(128, len(EMOTIONS))
        )
    def forward(self, x): return self.head(self.pool(self.backbone(x)))

# ── detection helpers ─────────────────────────────────────────────────────────
def decode(pred, thresh=0.35):
    obj=torch.sigmoid(pred[...,0]); txy=torch.sigmoid(pred[...,1:3]); twh=pred[...,3:5]
    dets=[]
    for gj in range(7):
        for gi in range(7):
            for a,(aw,ah) in enumerate(FACE_ANCHORS):
                s=obj[gj,gi,a].item()
                if s<thresh: continue
                tx,ty=txy[gj,gi,a].tolist(); tw,th=twh[gj,gi,a].tolist()
                dets.append((s,(gi+tx)/7,(gj+ty)/7,
                             float(np.exp(np.clip(tw,-4,4)))*aw,
                             float(np.exp(np.clip(th,-4,4)))*ah))
    return dets

def nms(dets, t=0.4):
    def b(d): return d[1]-d[3]/2,d[2]-d[4]/2,d[1]+d[3]/2,d[2]+d[4]/2
    def iou(a,bb):
        x1,y1=max(a[0],bb[0]),max(a[1],bb[1])
        x2,y2=min(a[2],bb[2]),min(a[3],bb[3])
        i=max(0,x2-x1)*max(0,y2-y1)
        u=(a[2]-a[0])*(a[3]-a[1])+(bb[2]-bb[0])*(bb[3]-bb[1])-i
        return i/u if u else 0
    dets=sorted(dets,key=lambda d:-d[0]); keep=[]
    while dets:
        best=dets.pop(0); keep.append(best)
        dets=[d for d in dets if iou(b(best),b(d))<=t]
    return keep

# ── affect state ──────────────────────────────────────────────────────────────
def get_state(x1n,y1n,x2n,y2n,emotion,conf):
    cx=(x1n+x2n)/2; cy=(y1n+y2n)/2
    # head pose
    vy=cy-0.5; hx=cx-0.5
    v=('up' if vy<-0.20 else 'slightly_up' if vy<-0.08 else
       'down' if vy>0.20 else 'slightly_down' if vy>0.08 else 'center')
    h=('left' if hx<-0.20 else 'slightly_left' if hx<-0.08 else
       'right' if hx>0.20 else 'slightly_right' if hx>0.08 else 'center')
    pose = 'center' if v=='center' and h=='center' else (h if v=='center' else (v if h=='center' else f'{v}_{h}'))
    # attention
    area=min(1.0,max(0.0,((x2n-x1n)*(y2n-y1n)-0.04)/0.36))
    cent=max(0.0,1.0-math.sqrt((cx-0.5)**2+(cy-0.5)**2)/0.5)
    att=round(0.5*area+0.5*cent,2)
    # engagement
    eng=round(min(1.0,0.6*att+0.4*(ENERGY.get(emotion,0.5)*conf)),2)
    # eye contact
    ec=not(set(pose.split('_'))&{'left','right','up','down'}) and abs(cx-0.5)<=0.30
    return {'emotion':emotion,'confidence':round(float(conf),2),
            'attention_score':att,'engagement_score':eng,
            'eye_contact':ec,'head_pose':pose}

# ── draw ──────────────────────────────────────────────────────────────────────
def draw(frame, x1,y1,x2,y2, state):
    color = EMO_COLORS.get(state['emotion'],(0,255,0))
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    label = f"{state['emotion']}  {state['confidence']*100:.0f}%"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_ = cv2.getTextSize(label,font,0.6,1)
    cv2.rectangle(frame,(x1,max(0,y1-th-8)),(x1+tw+8,y1),color,-1)
    cv2.putText(frame,label,(x1+4,y1-5),font,0.6,(255,255,255),1,cv2.LINE_AA)
    info=[f"attn  {state['attention_score']}",
          f"engag {state['engagement_score']}",
          f"eye   {'yes' if state['eye_contact'] else 'no'}",
          f"pose  {state['head_pose']}"]
    for i,ln in enumerate(info):
        cv2.putText(frame,ln,(x2+6,y1+16+i*18),font,0.42,color,1,cv2.LINE_AA)

def resolve_model_path(path):
    if os.path.exists(path):
        return path
    alt1 = os.path.join("src", path)
    if os.path.exists(alt1):
        return alt1
    alt2 = os.path.join("src", "Models", os.path.basename(path))
    if os.path.exists(alt2):
        return alt2
    alt3 = os.path.join("Models", os.path.basename(path))
    if os.path.exists(alt3):
        return alt3
    return path

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--face',    default='src/Models/strong_face_net.pt', help='Path to strong_face_net.pt')
    ap.add_argument('--emotion', default='src/Models/emotion_model.pt',     help='Path to emotion_model.pt')
    ap.add_argument('--camera',  type=int, default=0)
    ap.add_argument('--conf',    type=float, default=0.35)
    args = ap.parse_args()

    face_path = resolve_model_path(args.face)
    emo_path  = resolve_model_path(args.emotion)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load face model
    face_net = FaceNet().to(device)
    face_net.load_state_dict(torch.load(face_path, map_location=device))
    face_net.eval()
    print(f"Face model loaded   <- {face_path}")

    # load emotion model
    emo_net = EmoNet().to(device)
    ckpt = torch.load(emo_path, map_location=device)
    emo_net.load_state_dict(ckpt.get('state_dict', ckpt))
    emo_net.eval()
    print(f"Emotion model loaded <- {emo_path}")

    # Dynamic prior offset from model training (subtracts average logits to
    # correct for the model's inherent class-preference bias)
    raw_prior = ckpt.get('emo_prior', [0.0]*len(EMOTIONS))
    emo_prior = torch.tensor(raw_prior, dtype=torch.float32)
    print(f"Loaded logit prior offset: {emo_prior.tolist()}")

    # neutral_bias: positive offset on the neutral logit to further reduce
    # sad dominance; comes from checkpoint or falls back to NEUTRAL_LOGIT_BIAS
    neutral_idx  = EMOTIONS.index('neutral')
    ckpt_nb      = ckpt.get('neutral_bias', 0.0)
    neutral_bias = max(ckpt_nb, NEUTRAL_LOGIT_BIAS)   # take the larger of the two
    print(f"Neutral logit bias applied: {neutral_bias:.3f}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: could not open camera. Try --camera 1")
        return
    print("\nCamera open — press Q to quit\n")

    conf   = args.conf
    prev   = time.time()
    last_print = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ih, iw = frame.shape[:2]

        # detect face
        rgb = cv2.cvtColor(cv2.resize(frame,(224,224)), cv2.COLOR_BGR2RGB)
        ft  = torch.from_numpy(rgb/255.0).permute(2,0,1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = face_net(ft)[0].cpu()
        dets = nms(decode(pred, conf))

        state = None
        if dets:
            _,xc,yc,bw,bh = dets[0]
            pad = 0.08
            x1n=max(0.0,xc-bw/2-pad*bw); y1n=max(0.0,yc-bh/2-pad*bh)
            x2n=min(1.0,xc+bw/2+pad*bw); y2n=min(1.0,yc+bh/2+pad*bh)
            x1,y1=int(x1n*iw),int(y1n*ih)
            x2,y2=int(x2n*iw),int(y2n*ih)

            # classify emotion
            crop = frame[y1:y2,x1:x2]
            if crop.size > 0:
                rc = cv2.cvtColor(cv2.resize(crop,(96,96)), cv2.COLOR_BGR2RGB)
                et = ((torch.from_numpy(rc/255.0).permute(2,0,1).float()-EMO_MEAN)/EMO_STD)
                with torch.no_grad():
                    logits = emo_net(et.unsqueeze(0).to(device))[0].cpu()
                    # 1. subtract learned prior offset (corrects average class bias)
                    cal_logits = logits - emo_prior
                    # 2. apply neutral boost to counteract residual sad dominance
                    cal_logits[neutral_idx] += neutral_bias
                    # 3. temperature scaling to soften over-confident predictions
                    cal_logits = cal_logits / EMO_TEMP
                    probs = torch.softmax(cal_logits, 0)
                    idx   = probs.argmax().item()
                state = get_state(x1n,y1n,x2n,y2n, EMOTIONS[idx], probs[idx].item())
                draw(frame,x1,y1,x2,y2,state)

        # print JSON every second
        if state and time.time()-last_print >= 1.0:
            print(json.dumps(state, indent=2))
            last_print = time.time()

        # fps
        now=time.time(); fps=1/max(now-prev,1e-6); prev=now
        cv2.putText(frame,f"FPS {fps:.1f} | Q to quit",(10,26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2,cv2.LINE_AA)

        cv2.imshow("AffectFusion",frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'),ord('Q'),27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()