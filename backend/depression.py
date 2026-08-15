from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch

MODEL_PATH = "models/depression_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


def detect_depression(text):

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

    depression_risk = probs[0][1].item()

    return {
        "depression_risk": round(
            depression_risk,
            4
        )
    }


if __name__ == "__main__":

    samples = [

        "I have lost interest in everything.",

        "I feel empty and hopeless.",

        "I enjoyed spending time with my friends today.",

        "I don't see any purpose in life anymore."
    ]

    for text in samples:

        print("\nText:", text)

        print(
            detect_depression(text)
        )