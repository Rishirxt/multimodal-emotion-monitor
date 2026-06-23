from emotion import detect_emotion

from intent import detect_intent

from depression import detect_depression

from state_tracker import StateTracker

from question_engine import QuestionEngine

from suicide import SuicideDetector

tracker = StateTracker()

engine = QuestionEngine()

suicide_detector = SuicideDetector()

def process_message(text):

    emotion_result = detect_emotion(
        text
    )

    intent_result = detect_intent(
        text
    )

    depression_result = detect_depression(
        text
    )
    
    suicide_result = suicide_detector.assess_risk(

    text=text,

    emotion=
        emotion_result["emotion"],

    intent=
        intent_result["intent"],

    depression_risk=
        depression_result["depression_risk"]
    )

    state = tracker.update_state(

        text=text,

        emotion=
            emotion_result["emotion"],

        intent=
            intent_result["intent"],

        depression_risk=
            depression_result["depression_risk"]
    )

    next_question = engine.get_next_question(
        state
    )

    return {

        "emotion":
            emotion_result,

        "intent":
            intent_result,

        "depression":
            depression_result,
            
        "suicide":
            suicide_result,

        "state":
            state,

        "next_question":
            next_question
    }


if __name__ == "__main__":

    samples = [

        "I can't sleep at night and I feel exhausted all the time.",

        "Nobody talks to me anymore and I feel lonely.",

        "I am worried about my exams and future.",

        "I don't see any purpose in life anymore.",
        
        "I want to kill myself."
    ]

    for text in samples:

        print("\n" + "=" * 80)

        print("\nUSER:")

        print(text)

        result = process_message(
            text
        )

        print("\nPIPELINE OUTPUT:")

        print(result)