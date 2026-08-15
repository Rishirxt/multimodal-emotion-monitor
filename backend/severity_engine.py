class SeverityEngine:

    """
    Estimates overall symptom severity.

    This is NOT a diagnostic model.

    It combines:
    - Depression model confidence
    - Emotion
    - Intent
    - Conversation history
    """

    NEGATIVE_INTENTS = {

        "sleep_issue",

        "loneliness",

        "hopelessness",

        "general_depression",

        "self_harm"

    }

    NEGATIVE_EMOTIONS = {

        "sadness",

        "fear"

    }

    def calculate_severity(

        self,

        emotion,

        intent,

        depression_risk,

        history

    ):

        score = 0

        # ----------------------------------
        # Depression Risk
        # ----------------------------------

        if depression_risk >= 0.90:

            score += 4

        elif depression_risk >= 0.75:

            score += 3

        elif depression_risk >= 0.50:

            score += 2

        elif depression_risk >= 0.20:

            score += 1

        # ----------------------------------
        # Current Emotion
        # ----------------------------------

        if emotion in self.NEGATIVE_EMOTIONS:

            score += 1

        # ----------------------------------
        # Current Intent
        # ----------------------------------

        if intent in self.NEGATIVE_INTENTS:

            score += 1

        if intent == "self_harm":

            score += 2

        # ----------------------------------
        # Conversation History
        # ----------------------------------

        negative_history = 0

        for item in history:

            analysis = item.get("analysis", {})

            hist_intent = analysis.get("intent")

            if hist_intent in self.NEGATIVE_INTENTS:

                negative_history += 1

        if negative_history >= 3:

            score += 1

        elif negative_history >= 6:

            score += 2

        # ----------------------------------
        # Final Severity
        # ----------------------------------

        if score >= 7:

            return "high"

        elif score >= 3:

            return "moderate"

        return "low"


if __name__ == "__main__":

    engine = SeverityEngine()

    history = [

        {

            "analysis": {

                "intent": "sleep_issue"

            }

        },

        {

            "analysis": {

                "intent": "loneliness"

            }

        }

    ]

    severity = engine.calculate_severity(

        emotion="sadness",

        intent="sleep_issue",

        depression_risk=0.72,

        history=history

    )

    print(severity)