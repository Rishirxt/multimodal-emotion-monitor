import json
import ollama

from clinical_knowledge import get_domain


class ObjectiveExtractor:

    """
    Uses the LLM to determine which objectives
    of the current PHQ domain have already been covered.
    """

    def extract(

        self,

        current_domain,

        history

    ):

        if current_domain is None:

            return {}

        domain = get_domain(current_domain)

        objectives = list(domain["objectives"].keys())

        conversation = ""

        for turn in history:

            conversation += f"User: {turn['user_message']}\n"

            if turn.get("assistant_question"):

                conversation += (

                    f"Assistant: "

                    f"{turn['assistant_question']}\n"

                )

        prompt = f"""
You are assisting a clinical interview system.

Your task is to determine which clinical objectives
have already been answered.

Current PHQ Domain:

{current_domain}

Objectives:

{json.dumps(objectives, indent=4)}

Conversation:

{conversation}

Return ONLY valid JSON.

Example:

{{
    "duration": true,
    "frequency": false,
    "quality": true
}}

Rules:

1. Mark TRUE only if the conversation clearly provides
information for that objective.

2. Otherwise FALSE.

3. Do NOT explain.

4. Return JSON ONLY.
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

        text = response["message"]["content"]

        try:

            return json.loads(text)

        except Exception:

            return {

                objective: False

                for objective in objectives

            }


if __name__ == "__main__":

    extractor = ObjectiveExtractor()

    history = [

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

    from pprint import pprint

    pprint(

        extractor.extract(

            "sleep",

            history

        )

    )