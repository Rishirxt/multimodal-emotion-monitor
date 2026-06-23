import random

class QuestionEngine:

    def __init__(self):

        self.question_bank = {

            "sleep_issue": [
                "How long have you been experiencing sleep difficulties?",
                "Do you find it difficult to fall asleep or stay asleep?",
                "How does poor sleep affect your daily activities?"
            ],

            "academic_stress": [
                "What aspect of your studies is causing the most stress?",
                "How long have you been feeling overwhelmed academically?",
                "Do you feel the stress is affecting your sleep or mood?"
            ],

            "loneliness": [
                "Do you feel isolated from friends or family?",
                "How often do you feel alone during the week?",
                "Has this feeling changed recently?"
            ],

            "hopelessness": [
                "Can you tell me more about what makes you feel hopeless?",
                "How long have these feelings been present?",
                "Do these feelings affect your motivation?"
            ],

            "general_depression": [
                "Have you noticed changes in your energy levels?",
                "Have you lost interest in activities you once enjoyed?",
                "How often do you experience these feelings?"
            ]
        }

    def get_next_question(self, state):

        intent = state["intent"]

        if intent in self.question_bank:

            return random.choice(
                self.question_bank[intent]
            )

        return (
            "Can you tell me more about how you've been feeling lately?"
        )


if __name__ == "__main__":

    state = {

        "emotion": "sadness",

        "intent": "sleep_issue",

        "depression_risk": 0.92,

        "severity": "high",

        "risk_level": "medium",

        "conversation_stage": 1
    }

    engine = QuestionEngine()

    print(
        engine.get_next_question(state)
    )