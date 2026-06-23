from langchain_ollama import ChatOllama


llm = ChatOllama(

    model="qwen2.5:3b",

    temperature=0.3
)


def generate_question(state):

    prompt = f"""

You are an empathetic mental health screening assistant.

Current conversation state:

Emotion:
{state['emotion']}

Intent:
{state['intent']}

Depression Risk:
{state['depression_risk']}

Severity:
{state['severity']}

Suicide Risk:
{state['suicide_risk_level']}

Conversation History:
{state['history']}

Generate ONE short follow-up question.

Do not give advice.

Do not diagnose.

Ask only one question.

"""

    response = llm.invoke(
        prompt
    )

    return response.content


if __name__ == "__main__":

    state = {

        "emotion": "sadness",

        "intent": "sleep_issue",

        "depression_risk": 0.91,

        "severity": "high",

        "suicide_risk_level": "low",

        "history": [
            {
                "text":
                "I can't sleep at night."
            }
        ]
    }

    question = generate_question(
        state
    )

    print(question)