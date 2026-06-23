from severity_engine import SeverityEngine


class StateTracker:

    def __init__(self):

        self.conversation_stage = 1

        self.history = []

        self.severity_engine = SeverityEngine()

    def update_state(

        self,

        text,

        emotion,

        intent,

        depression_risk
    ):

        self.history.append({

            "text": text,

            "emotion": emotion,

            "intent": intent,

            "depression_risk":
                round(depression_risk, 4)
        })

        if len(self.history) > 10:

            self.history.pop(0)

        severity = self.severity_engine.calculate_severity(

            emotion,

            intent,

            depression_risk,

            self.history
        )

        if intent == "self_harm":

            risk_level = "high"

        elif depression_risk >= 0.9:

            risk_level = "medium"

        else:

            risk_level = "low"

        return {

            "emotion": emotion,

            "intent": intent,

            "depression_risk":
                round(depression_risk, 4),

            "severity": severity,

            "risk_level": risk_level,

            "conversation_stage":
                self.conversation_stage,

            "history":
                self.history
        }


if __name__ == "__main__":

    tracker = StateTracker()

    messages = [

        (
            "I can't sleep at night",
            "sadness",
            "sleep_issue",
            0.91
        ),

        (
            "I feel exhausted all the time",
            "sadness",
            "general_depression",
            0.94
        )
    ]

    for msg in messages:

        state = tracker.update_state(

            text=msg[0],

            emotion=msg[1],

            intent=msg[2],

            depression_risk=msg[3]
        )

    print(state)