# AffectFusion
### Multimodal Conversational AI for Adaptive Depression Screening and Early Intervention

> **Disclaimer:** AffectFusion is a research and screening support tool only. It is not a diagnostic system and does not replace licensed mental health professionals. Clinical validation is required before any real-world deployment.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [CV Track — Face Detection & Emotion Recognition](#cv-track)
6. [NLP Track — Conversational Depression Screening](#nlp-track)
7. [Fusion Layer — Combining CV + NLP](#fusion-layer)
8. [Running the Full System](#running-the-full-system)
9. [Output Reference](#output-reference)
10. [Model Performance](#model-performance)
11. [Dataset Credits](#dataset-credits)
12. [Ethical Considerations](#ethical-considerations)
13. [Team](#team)

---

## Project Overview

AffectFusion is an intelligent multimodal system that conducts adaptive depression screening by combining two independent signals:

- **Visual signal (CV Track):** Real-time facial expression recognition via webcam — detects emotion, attention, engagement, eye contact, and head pose.
- **Verbal signal (NLP Track):** Conversational AI that analyses text responses for sentiment, depression risk, suicidal ideation, and dynamically selects the next clinical question.

By fusing both signals, the system produces a richer picture of a user's mental state than either modality alone could provide. A user who says they feel "fine" but displays a consistently sad facial expression and low engagement score is flagged differently than one whose verbal and facial signals agree.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER                                  │
│              speaks + types + faces camera                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────┐
│   CV PIPELINE   │       │    NLP PIPELINE       │
│                 │       │                        │
│ 1. Face Detect  │       │ 1. Intent Analysis     │
│ 2. Face Crop    │       │ 2. Sentiment Analysis  │
│ 3. Emotion      │       │ 3. Depression Scoring  │
│    Classify     │       │ 4. Risk Assessment     │
│ 4. Affect State │       │ 5. Question Selection  │
└────────┬────────┘       └───────────┬────────────┘
         │                            │
         │     cv_output.json         │
         └──────────┬─────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  FUSION LAYER    │
         │  fusion.py       │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  UNIFIED OUTPUT  │
         │  (see below)     │
         └──────────────────┘
```

---

## Repository Structure

```
AffectFusion/
│
├── Models/
│   ├── strong_face_net.pt          # Trained face detector (Step 5)
│   └── emotion_model.pt            # Trained emotion classifier
│
├── FaceData/
│   ├── images/                     # WIDER FACE dataset images
│   └── labels/                     # YOLO format bounding box labels
│
├── EmotionData/
│   ├── train/                      # FER2013 train split
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                       # FER2013 test split (same structure)
│
├── cv/                             # CV Track scripts
│   ├── explore_data.py             # Phase 1 — data sanity check
│   ├── step2_single_box_model.py   # Phase 2 — single box baseline
│   ├── step3_grid_model.py         # Phase 3 — 7×7 grid detector
│   ├── step4_anchor_model.py       # Phase 4 — anchor-based detector
│   ├── step5_strong_model.py       # Phase 5 — MobileNetV2 + augmentation
│   ├── emotion_explore.py          # Emotion Phase 1 — data exploration
│   ├── emotion_train.py            # Emotion Phase 2 — classifier training
│   └── run_camera.py               # LIVE — camera + detection + emotion
│
├── nlp/                            # NLP Track scripts
│   ├── intent_classifier.py        # Intent and sentiment analysis
│   ├── depression_scorer.py        # PHQ-9 based risk scoring
│   ├── question_engine.py          # Adaptive question selection
│   ├── safety_layer.py             # Suicidal ideation detection
│   └── conversation_manager.py     # Conversation state management
│
├── fusion.py                       # Merges CV + NLP outputs
├── affect_state.py                 # Computes 6-field affect JSON from CV
├── cv_output.json                  # Written by run_camera.py every frame
└── README.md
```

---

## Installation

**Requirements:** Python 3.8+, a webcam

```bash
pip install torch torchvision opencv-python numpy
```

For NLP dependencies:
```bash
pip install transformers spacy torch sentencepiece
python -m spacy download en_core_web_sm
```

---

## CV Track

**Owner:** Person A — Computer Vision  
**Dataset:** WIDER FACE (face detection) + FER2013 (emotion)

The CV track was built from scratch in five progressive steps, each fixing a specific limitation of the previous one.

### Step-by-step build

#### Phase 1 — Data Exploration
```bash
python cv/explore_data.py --images-dir FaceData/images --labels-dir FaceData/labels
```
Reads the YOLO-format label files, draws ground-truth boxes on sample images, and prints dataset statistics. Run this first to confirm the data is being read correctly before training anything.

#### Phase 2 — Single Box Baseline (`step2_single_box_model.py`)
```bash
python cv/step2_single_box_model.py --images-dir FaceData/images --labels-dir FaceData/labels --epochs 15
```
The simplest possible detector — one CNN, one output vector, one predicted box per image. Intentionally limited: cannot detect more than one face. Establishes a baseline IoU of **0.67** and makes the structural ceiling concrete.

#### Phase 3 — Grid Detector (`step3_grid_model.py`)
```bash
python cv/step3_grid_model.py --images-dir FaceData/images --labels-dir FaceData/labels --epochs 20
```
Divides the image into a **7×7 grid**. Each cell independently predicts whether a face center falls inside it. Enables multi-face detection for the first time. Portrait IoU improves to **0.93**. Two faces close together can still merge — fixed in Phase 4.

#### Phase 4 — Anchor-Based Detector (`step4_anchor_model.py`)
```bash
python cv/step4_anchor_model.py --images-dir FaceData/images --labels-dir FaceData/labels --epochs 25
```
Gives each grid cell **two shape templates** (anchors) — one for small crowd faces, one for large portrait faces. Each face now lands in its own anchor slot. Crowd photo: **8/8 faces detected, 0 false positives**.

#### Phase 5 — Strong Model (`step5_strong_model.py`) ← Final face model
```bash
python cv/step5_strong_model.py --images-dir FaceData/images --labels-dir FaceData/labels --epochs 40
```
Two major upgrades:
- **MobileNetV2 pretrained backbone** — starts from ImageNet weights instead of learning from scratch
- **Data augmentation** — random ±25° rotation, horizontal flip, brightness/contrast/saturation jitter

The rotation augmentation is specifically what fixed detection when the user looks slightly down. Saves best checkpoint to `step5_output/strong_face_net.pt`.

### Emotion Classifier

#### Emotion Phase 1 — Data Exploration
```bash
python cv/emotion_explore.py --data-dir EmotionData
```
Prints per-class counts, image sizes, and class imbalance ratio. FER2013 has a **16.5× imbalance** (happy: 7215 vs disgust: 436) — the training script handles this with weighted loss.

#### Emotion Phase 2 — Training (`emotion_train.py`) ← Final emotion model
```bash
python cv/emotion_train.py --data-dir EmotionData --epochs 30 --batch-size 64
```

Key design decisions:
- **Weighted cross-entropy loss** — rare classes (disgust, fear) get proportionally higher weight so the model doesn't just learn to predict "happy"
- **Warmup phase** — first 5 epochs train the classification head only, backbone frozen, then top layers unfreeze for fine-tuning
- **ImageNet normalisation** — matches MobileNetV2 pretraining expectations
- **96×96 input** — upsampled from 48×48 for better spatial signal

Saves best checkpoint to `emotion_output/emotion_model.pt`.

### Running the Live CV Pipeline
```bash
# From Command Prompt (not VS Code terminal — OpenCV windows require a real terminal on Windows)
python cv/run_camera.py --face Models/strong_face_net.pt --emotion Models/emotion_model.pt
```

What happens:
1. Camera opens
2. Every frame: face detected → cropped → emotion classified
3. Affect state computed from box geometry + emotion
4. Green box + emotion label drawn on screen
5. JSON printed to terminal every second
6. `cv_output.json` written every frame for the NLP side to read

---

## NLP Track

**Owner:** Person B — Natural Language Processing  
**Models:** Fine-tuned transformer (BERT/RoBERTa base)

The NLP track conducts an adaptive clinical interview based on PHQ-9 domains, dynamically selecting follow-up questions based on the user's responses.

### Components

#### Intent & Sentiment Analysis (`intent_classifier.py`)
Classifies each user message into one of the PHQ-9 domains (sleep, energy, interest, appetite, concentration, self-worth, psychomotor, mood) and extracts sentiment polarity.

```python
from nlp.intent_classifier import classify
result = classify("I haven't been sleeping well for weeks")
# → { "intent": "sleep_issue", "confidence": 0.91 }
```

#### Depression Scorer (`depression_scorer.py`)
Maps conversation history to a PHQ-9 risk score (0–1). Tracks which domains have been covered and estimates severity.

```python
from nlp.depression_scorer import score
result = score(conversation_history)
# → { "risk_score": 0.78, "severity": "moderate" }
```

#### Question Engine (`question_engine.py`)
Selects the next clinical question based on:
- Which PHQ-9 domains are already covered
- Current risk score
- User's emotional state (from CV output)

```python
from nlp.question_engine import next_question
q = next_question(covered_domains, risk_score, cv_state)
# → "Have you lost interest in activities you usually enjoy?"
```

#### Safety Layer (`safety_layer.py`)
Screens every message for suicidal ideation or self-harm risk. Escalates immediately if threshold is crossed regardless of conversation stage.

```python
from nlp.safety_layer import check
flags = check(user_message)
# → { "risk_score": 2, "risk_level": "moderate", "emergency": False }
```

#### Conversation Manager (`conversation_manager.py`)
Maintains full conversation state across turns — stage, covered domains, history length, and the next question to ask.

---

## Fusion Layer

`fusion.py` is called by the NLP side after every conversation turn. It reads `cv_output.json` (written by the CV pipeline) and merges the visual signal with the NLP analysis.

```python
from fusion import merge_outputs

final = merge_outputs(nlp_output_dict)
```

The CV state is considered **stale** if the timestamp in `cv_output.json` is more than 3 seconds old — in that case `face_detected` is set to `false` and facial emotion is omitted from the merged output. This prevents old CV data from influencing NLP decisions when the camera is paused.

**Emotion fusion rule:**
- Both agree → use that emotion
- CV confidence < 0.50 → trust text (more reliable for depression screening)
- Disagreement with confident CV → still defer to text, log the disagreement for research

---

## Running the Full System

### Terminal 1 — CV Pipeline (keep running)
```bash
python cv/run_camera.py --face Models/strong_face_net.pt --emotion Models/emotion_model.pt
```

### Terminal 2 — NLP + Fusion (your conversation loop)
```python
from fusion import merge_outputs
from nlp.conversation_manager import ConversationManager

cm = ConversationManager()

while True:
    user_input = input("User: ")
    nlp_output = cm.process(user_input)
    final      = merge_outputs(nlp_output)

    print(final["response"]["question"])

    if final["flags"]["emergency"]:
        trigger_escalation()
        break
```

---

## Output Reference

### CV Output (`cv_output.json`) — written every frame
```json
{
  "emotion":          "sad",
  "confidence":       0.82,
  "attention_score":  0.74,
  "engagement_score": 0.63,
  "eye_contact":      true,
  "head_pose":        "slightly_down",
  "face_detected":    true,
  "timestamp":        1719123456.78
}
```

| Field | Range | Description |
|---|---|---|
| `emotion` | 7 classes | angry, disgust, fear, happy, neutral, sad, surprise |
| `confidence` | 0–1 | Softmax probability of predicted emotion |
| `attention_score` | 0–1 | Face size + centrality in frame |
| `engagement_score` | 0–1 | Attention × emotion energy |
| `eye_contact` | bool | Face centered + no strong head turn |
| `head_pose` | string | center / slightly_down / left / etc. |

### Fused Output — produced after each NLP turn
```json
{
  "analysis": {
    "text_emotion":   { "emotion": "sadness",  "confidence": 0.98 },
    "facial_emotion": { "emotion": "sadness",  "confidence": 0.82 },
    "combined_emotion": "sadness",
    "visual_context": {
      "attention_score": 0.74, "engagement_score": 0.63,
      "eye_contact": true, "head_pose": "slightly_down",
      "face_detected": true
    },
    "intent":      { "intent": "sleep_issue", "confidence": 0.91 },
    "depression":  { "risk_score": 0.78 },
    "suicide":     { "risk_level": "moderate" },
    "severity":    "moderate"
  },
  "conversation": {
    "covered_domains":   ["sleep", "energy"],
    "remaining_domains": ["interest", "appetite", "self_worth"]
  },
  "response": {
    "question": "Have you noticed changes in your appetite recently?"
  },
  "flags": {
    "requires_attention": false,
    "high_risk": false,
    "emergency": false
  }
}
```

---

## Model Performance

### Face Detector (Step 5 — Strong Model)
| Metric | Result |
|---|---|
| Architecture | MobileNetV2 + 2-anchor YOLO head |
| Training data | 1,154 images (WIDER FACE subset) |
| Portrait avg IoU | 0.93 |
| Crowd detection | 8/8 faces, 0 false positives |
| Pose robustness | Handles ±25° head tilt |

### Emotion Classifier
| Metric | Result |
|---|---|
| Architecture | MobileNetV2 + FC head |
| Training data | FER2013 (28,709 train / 7,178 test) |
| Overall accuracy | 56.8% |
| Best class | happy 75.8% / surprise 74.0% |
| Hardest class | fear 28.1% (visually similar to surprise) |

> FER2013 human-level agreement is ~65%. Our 56.8% is within the expected range for this dataset.

---

## Dataset Credits

| Dataset | Use | License |
|---|---|---|
| WIDER FACE | Face detection training | Research only |
| FER2013 | Emotion classification training | Research only |
| PHQ-9 | Clinical question framework | Public domain |

---

## Ethical Considerations

- **Not a diagnostic tool.** Depression severity estimates are indicators for clinical review, not diagnoses.
- **Privacy.** Camera frames are processed locally in real time. No images are stored or transmitted.
- **Safety escalation.** The safety layer monitors every turn for suicidal ideation. If `emergency: true` is flagged, the system must route to a human professional immediately.
- **Bias.** FER2013 has known demographic imbalances. The model may perform differently across skin tones, ages, and genders. Clinical validation across diverse populations is required before deployment.
- **Consent.** Users must be informed that their facial expressions are being analysed before a session begins.

---

## Team

| Track | Responsibility |
|---|---|
| **Rishi Ratheesh — CV** | Face detection · Emotion recognition · Affect state · Camera pipeline |
| **Abhishek Gajendran — NLP** | Intent analysis · Depression scoring · Question engine · Safety layer · Conversation management |
| **Shared** | Fusion layer · Evaluation · Ethics review · Clinical safety validation |
