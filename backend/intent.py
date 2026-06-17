from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import pickle

MODEL_PATH = "models/intent_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

with open(
    f"{MODEL_PATH}/label_encoder.pkl",
    "rb"
) as f:

    label_encoder = pickle.load(f)


def detect_intent(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

    confidence, pred = torch.max(
        probs,
        dim=1
    )

    intent = label_encoder.inverse_transform(
        [pred.item()]
    )[0]

    return {
        "intent": intent,
        "confidence": round(
            confidence.item(),
            4
        )
    }


if __name__ == "__main__":

    samples = [

        "I can't sleep at night.",

        "I am worried about my exams.",

        "Nobody talks to me anymore.",

        "I feel hopeless."
    ]

    for text in samples:

        print("\nText:", text)

        print(
            detect_intent(text)
        )