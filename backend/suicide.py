class SuicideDetector:

    def __init__(self):

        self.high_risk_keywords = [

            "kill myself",

            "end my life",

            "suicide",

            "want to die",

            "don't want to live",

            "better off dead",

            "hurt myself",

            "self harm",

            "self-harm",

            "take my own life"
        ]

    def assess_risk(

        self,

        text,

        emotion,

        intent,

        depression_risk
    ):

        text = text.lower()

        score = 0

        # -----------------
        # Keyword Matching
        # -----------------

        for keyword in self.high_risk_keywords:

            if keyword in text:

                score += 5

        # -----------------
        # Intent
        # -----------------

        if intent == "self_harm":

            score += 3

        # -----------------
        # Depression Risk
        # -----------------

        if depression_risk >= 0.9:

            score += 2

        elif depression_risk >= 0.7:

            score += 1

        # -----------------
        # Emotion
        # -----------------

        if emotion == "sadness":

            score += 1

        # -----------------
        # Final Assessment
        # -----------------

        if score >= 7:

            level = "emergency"

        elif score >= 4:

            level = "high"

        elif score >= 2:

            level = "moderate"

        else:

            level = "low"

        return {

            "suicide_risk_score": score,

            "suicide_risk_level": level
        }


if __name__ == "__main__":

    detector = SuicideDetector()

    samples = [

        (
            "I want to kill myself.",

            "sadness",

            "self_harm",

            0.95
        ),

        (
            "Life feels meaningless.",

            "sadness",

            "general_depression",

            0.92
        ),

        (
            "I am stressed about exams.",

            "fear",

            "academic_stress",

            0.02
        )
    ]

    for sample in samples:

        result = detector.assess_risk(

            text=sample[0],

            emotion=sample[1],

            intent=sample[2],

            depression_risk=sample[3]
        )

        print("\n", sample[0])

        print(result)