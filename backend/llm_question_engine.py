import ollama


class LLMQuestionEngine:

    def generate_question(

        self,

        reasoning,

        analysis,

        history

    ):

        # ------------------------------------------
        # Emergency Override
        # ------------------------------------------

        if reasoning["decision"] == "EMERGENCY_PROTOCOL":

            return (

                "Are you currently safe, or do you feel "

                "you may act on these thoughts?"

            )

        conversation = ""

        for turn in history:

            conversation += (

                f"User: {turn['user_message']}\n"

            )

            if turn.get("assistant_question"):

                conversation += (

                    f"Assistant: "

                    f"{turn['assistant_question']}\n"

                )

        completion = reasoning.get(

            "completion",

            {}

        )

        prompt = f"""
You are an experienced clinical interviewer.

You are conducting a structured depression screening.

You are NOT diagnosing.

You are ONLY collecting missing information.

------------------------------------------------

Current Domain

{reasoning["target_domain"]}

Clinical Goal

{reasoning["clinical_goal"]}

LLM Focus

{reasoning["llm_focus"]}

------------------------------------------------

Already Collected

{completion.get("completed_objectives", [])}

Still Missing

{completion.get("remaining_objectives", [])}

------------------------------------------------

Latest Analysis

Emotion:
{analysis["emotion"]}

Intent:
{analysis["intent"]}

Severity:
{analysis["severity"]}

------------------------------------------------

Conversation

{conversation}

------------------------------------------------

Instructions

1. Ask ONLY ONE follow-up question.

2. Focus ONLY on the remaining objectives.

3. Never ask about completed objectives.

4. Build naturally from the latest user response.

5. Be empathetic.

6. Do not diagnose.

7. Do not provide advice.

8. Keep the question under 20 words.

9. Return ONLY the question.
"""

        response = ollama.chat(

            model="qwen2.5:3b",

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ]

        )

        return response["message"]["content"].strip()


if __name__ == "__main__":

    engine = LLMQuestionEngine()

    reasoning = {

        "decision": "CONTINUE_CURRENT_DOMAIN",

        "target_domain": "sleep",

        "clinical_goal": "Assess sleep disturbance.",

        "llm_focus": "Understand sleep quality.",

        "completion": {

            "completed_objectives": [

                "duration",

                "frequency"

            ],

            "remaining_objectives": [

                "quality",

                "daytime_impact"

            ]

        }

    }

    analysis = {

        "emotion": "sadness",

        "intent": "sleep_issue",

        "severity": "moderate"

    }

    history = [

        {

            "user_message":

            "I sleep only three hours every night.",

            "assistant_question":

            "How long has this been happening?"

        },

        {

            "user_message":

            "For two months."

        }

    ]

    print(

        engine.generate_question(

            reasoning,

            analysis,

            history

        )

    )