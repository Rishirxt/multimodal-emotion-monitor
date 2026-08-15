

"""
clinical_reasoning_engine.py

Determines what the interview should do next.

This module never generates natural language.

It returns structured reasoning decisions based on
clinical objectives rather than question count.
"""

from clinical_knowledge import get_domain
from objective_extractor import ObjectiveExtractor
from information_estimator import InformationEstimator


class ClinicalReasoningEngine:

    def __init__(self):

        self.extractor = ObjectiveExtractor()

        self.estimator = InformationEstimator()

    def decide(self, context):

        analysis = context["analysis"]

        conversation = context["conversation"]

        assessment = context["assessment"]

        # -------------------------------------------------
        # Emergency Override
        # -------------------------------------------------

        if analysis["suicide_risk_level"] == "emergency":

            return {

                "decision": "EMERGENCY_PROTOCOL",

                "target_domain": "suicidal_thoughts",

                "clinical_goal":
                    "Assess immediate safety.",

                "reason":
                    "Emergency suicide risk detected.",

                "priority":
                    "critical",

                "llm_focus":
                    "Assess whether the user is currently safe.",

                "completion": {

                    "completion": 1.0,

                    "completed_objectives": [],

                    "remaining_objectives": []

                }

            }

        current = assessment["current_domain"]

        remaining = assessment["remaining_domains"]

        history = conversation["history"]

        # -------------------------------------------------
        # First domain
        # -------------------------------------------------

        if current is None:

            first = remaining[0]

            info = get_domain(first)

            return {

                "decision":
                    "SWITCH_DOMAIN",

                "target_domain":
                    first,

                "clinical_goal":
                    info["goal"],

                "reason":
                    "Beginning interview.",

                "priority":
                    "normal",

                "llm_focus":
                    info["llm_focus"],

                "completion": {

                    "completion": 0,

                    "completed_objectives": [],

                    "remaining_objectives":
                        list(info["objectives"].keys())

                }

            }

        # -------------------------------------------------
        # Estimate information completeness
        # -------------------------------------------------

        extracted = self.extractor.extract(

            current,

            history

        )

        completion = self.estimator.estimate(

            extracted

        )

        # -------------------------------------------------
        # Continue if not enough information
        # -------------------------------------------------

        if completion["completion"] < 0.75:

            info = get_domain(current)

            return {

                "decision":
                    "CONTINUE_CURRENT_DOMAIN",

                "target_domain":
                    current,

                "clinical_goal":
                    info["goal"],

                "reason":
                    f"Only {completion['completion']*100:.0f}% of clinical objectives collected.",

                "priority":
                    "normal",

                "llm_focus":
                    info["llm_focus"],

                "completion":
                    completion

            }

        # -------------------------------------------------
        # Move to preferred transition domain
        # -------------------------------------------------

        info = get_domain(current)

        for candidate in info["transition_candidates"]:

            if candidate in remaining:

                next_info = get_domain(candidate)

                return {

                    "decision":
                        "SWITCH_DOMAIN",

                    "target_domain":
                        candidate,

                    "clinical_goal":
                        next_info["goal"],

                    "reason":
                        f"{current} sufficiently explored ({completion['completion']*100:.0f}% complete).",

                    "priority":
                        "normal",

                    "llm_focus":
                        next_info["llm_focus"],

                    "completion":
                        completion

                }

        # -------------------------------------------------
        # Otherwise move to first remaining domain
        # -------------------------------------------------

        if remaining:

            next_domain = remaining[0]

            next_info = get_domain(next_domain)

            return {

                "decision":
                    "SWITCH_DOMAIN",

                "target_domain":
                    next_domain,

                "clinical_goal":
                    next_info["goal"],

                "reason":
                    "Current domain completed.",

                "priority":
                    "normal",

                "llm_focus":
                    next_info["llm_focus"],

                "completion":
                    completion

            }

        # -------------------------------------------------
        # Interview Complete
        # -------------------------------------------------

        return {

            "decision":
                "FINISH_ASSESSMENT",

            "target_domain":
                None,

            "clinical_goal":
                "Summarize findings.",

            "reason":
                "All PHQ domains completed.",

            "priority":
                "normal",

            "llm_focus":
                "Generate assessment summary.",

            "completion":
                completion

        }


if __name__ == "__main__":

    engine = ClinicalReasoningEngine()

    context = {

        "analysis": {

            "emotion": "sadness",

            "intent": "sleep_issue",

            "depression_risk": 0.82,

            "severity": "moderate",

            "risk_level": "low",

            "suicide_risk_level": "low"

        },

        "conversation": {

            "turn": 2,

            "current_domain": "sleep",

            "history": [

                {

                    "user_message":
                        "I sleep only three hours every night.",

                    "assistant_question":
                        "How long has this been happening?"

                },

                {

                    "user_message":
                        "Almost every day for two months.",

                    "assistant_question":
                        "How does it affect your day?"

                },

                {

                    "user_message":
                        "I feel exhausted all day."

                }

            ]

        },

        "assessment": {

            "current_domain": "sleep",

            "covered_domains": [

                "sleep"

            ],

            "remaining_domains": [

                "energy",

                "interest",

                "mood"

            ]

        }

    }

    from pprint import pprint

    pprint(

        engine.decide(

            context

        )

    )