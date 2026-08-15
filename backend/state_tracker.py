from severity_engine import SeverityEngine


class StateTracker:

    """
    Maintains the conversation history and analysis state.

    This class DOES NOT make interview decisions.
    It simply stores information collected so far.
    """

    def __init__(self):

        self.turn = 0

        self.history = []

        self.severity_engine = SeverityEngine()

    def update_state(

        self,

        text,

        emotion,

        intent,

        depression_risk,

        suicide_risk_level="low"

    ):

        self.turn += 1

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

        analysis = {

            "emotion": emotion,

            "intent": intent,

            "depression_risk": round(
                depression_risk,
                4
            ),

            "severity": severity,

            "risk_level": risk_level,

            "suicide_risk_level": suicide_risk_level
        }

        history_item = {

            "turn": self.turn,

            "user_message": text,

            "analysis": analysis.copy(),

            "assistant_question": None

        }

        self.history.append(

            history_item

        )

        if len(self.history) > 20:

            self.history.pop(0)

        state = {

            "analysis": analysis,

            "conversation": {

                "turn": self.turn,

                "history": self.history

            }

        }

        return state

    def save_assistant_question(

        self,

        question

    ):

        if self.history:

            self.history[-1][

                "assistant_question"

            ] = question


if __name__ == "__main__":

    tracker = StateTracker()

    state = tracker.update_state(

        text="I can't sleep.",

        emotion="sadness",

        intent="sleep_issue",

        depression_risk=0.82

    )

    tracker.save_assistant_question(

        "How long have you been having trouble sleeping?"

    )

    from pprint import pprint

    pprint(state)