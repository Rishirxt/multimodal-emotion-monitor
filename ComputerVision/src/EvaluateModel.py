import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from EmotionTrain import EmotionClassifier, EmotionDataset, EMOTIONS, NUM_CLASSES

def main():
    parser = argparse.ArgumentParser()
    default_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument('--data-dir',   default=default_data_dir)
    parser.add_argument('--model-path', default=os.path.join(os.path.dirname(__file__), 'Models', 'emotion_model.pt'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--neutral-bias', type=float, default=0.0, help="Neutral logit boost offset")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading checkpoint from: {args.model_path}")

    if not os.path.exists(args.model_path):
        print(f"ERROR: Checkpoint file not found at {args.model_path}")
        return

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    
    # Initialize model
    model = EmotionClassifier(pretrained=False).to(device)
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    model.eval()

    # Load logit prior offset
    raw_prior = ckpt.get('emo_prior', [0.0]*NUM_CLASSES)
    emo_prior = torch.tensor(raw_prior, dtype=torch.float32).to(device)
    print(f"Loaded logit prior offset: {emo_prior.tolist()}")

    # Load test dataset
    test_dir = os.path.join(args.data_dir, 'test')
    print(f"Loading test dataset from: {test_dir}")
    test_ds = EmotionDataset(test_dir, augment_data=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # We evaluate both raw outputs and calibrated outputs
    raw_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    cal_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    neutral_idx = EMOTIONS.index('neutral')
    neutral_bias = args.neutral_bias

    correct_raw = 0
    correct_cal = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            
            # Raw predictions
            preds_raw = logits.argmax(dim=1)
            correct_raw += (preds_raw == labels).sum().item()
            for p, l in zip(preds_raw.cpu().numpy(), labels.cpu().numpy()):
                raw_conf_matrix[l, p] += 1

            # Calibrated predictions (subtract prior offset + add neutral bias)
            cal_logits = logits - emo_prior
            cal_logits[:, neutral_idx] += neutral_bias
            preds_cal = cal_logits.argmax(dim=1)
            correct_cal += (preds_cal == labels).sum().item()
            for p, l in zip(preds_cal.cpu().numpy(), labels.cpu().numpy()):
                cal_conf_matrix[l, p] += 1

            total += imgs.size(0)

    acc_raw = 100.0 * correct_raw / total if total > 0 else 0
    acc_cal = 100.0 * correct_cal / total if total > 0 else 0

    def print_metrics(title, conf_matrix, overall_acc):
        print(f"\n{'='*60}")
        print(f" {title} (Accuracy: {overall_acc:.2f}%)")
        print(f"{'='*60}")
        header = "          " + "".join([f"{e[:5]:>7}" for e in EMOTIONS])
        print(header)

        for i, emo in enumerate(EMOTIONS):
            row_str = f"{emo:10s}" + "".join([f"{conf_matrix[i, j]:7d}" for j in range(NUM_CLASSES)])
            total_class = conf_matrix[i].sum()
            acc = 100.0 * conf_matrix[i, i] / total_class if total_class > 0 else 0.0
            print(f"{row_str}  (Acc: {acc:5.1f}%)")
        print(f"{'='*60}")

    print_metrics("RAW MODEL PERFORMANCE", raw_conf_matrix, acc_raw)
    print_metrics(f"CALIBRATED MODEL PERFORMANCE (Neutral Bias: {neutral_bias:.2f})", cal_conf_matrix, acc_cal)

if __name__ == '__main__':
    main()
