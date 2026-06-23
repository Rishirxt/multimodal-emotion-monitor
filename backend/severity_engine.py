class SeverityEngine:

    def calculate_severity(

        self,

        emotion,

        intent,

        depression_risk,

        history
    ):

        score = 0

        # --------------------
        # Depression Risk Score
        # --------------------

        if depression_risk >= 0.8:

            score += 3

        elif depression_risk >= 0.5:

            score += 2

        elif depression_risk >= 0.2:

            score += 1

        # --------------------
        # Emotion Score
        # --------------------

        if emotion in [

            "sadness",

            "fear"
        ]:

            score += 1

        # --------------------
        # Intent Score
        # --------------------

        if intent in [

            "sleep_issue",

            "loneliness",

            "hopelessness",

            "general_depression"
        ]:

            score += 1

        # --------------------
        # Conversation History
        # --------------------

        negative_count = 0

        for item in history:

            if item["intent"] in [

                "sleep_issue",

                "loneliness",

                "hopelessness",

                "general_depression"
            ]:

                negative_count += 1

        if negative_count >= 3:

            score += 1

        # --------------------
        # Final Severity
        # --------------------

        if score >= 5:

            return "high"

        elif score >= 2:

            return "moderate"

        else:

            return "low"


if __name__ == "__main__":

    engine = SeverityEngine()

    severity = engine.calculate_severity(

        emotion="sadness",

        intent="sleep_issue",

        depression_risk=0.02,

        history=[
            {
                "intent":
                "sleep_issue"
            }
        ]
    )

    print(severity)