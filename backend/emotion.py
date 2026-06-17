from transformers import pipeline

# Load emotion detection model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def detect_emotion(text):
    """
    Detects the primary emotion in the input text.
    """

    results = emotion_classifier(text)[0]

    best_emotion = max(results, key=lambda x: x["score"])

    return {
        "emotion": best_emotion["label"],
        "confidence": round(best_emotion["score"], 4)
    }


if __name__ == "__main__":

    sample_text = "I feel sad and lonely."

    result = detect_emotion(sample_text)

    print(result)