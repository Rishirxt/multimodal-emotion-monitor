from emotion import detect_emotion
from intent import detect_intent
from depression import detect_depression
from suicide import SuicideDetector

from state_tracker import StateTracker
from phq_tracker import PHQTracker
from conversation_manager import ConversationManager
from clinical_reasoning_engine import ClinicalReasoningEngine
from llm_question_engine import LLMQuestionEngine


# ==========================================================
# Initialize Components
# ==========================================================

state_tracker = StateTracker()

phq_tracker = PHQTracker()

conversation_manager = ConversationManager()

reasoning_engine = ClinicalReasoningEngine()

llm_engine = LLMQuestionEngine()

suicide_detector = SuicideDetector()


# ==========================================================
# Main Pipeline
# ==========================================================

def process_message(text):

    # =====================================================
    # NLP Analysis
    # =====================================================

    emotion = detect_emotion(text)

    intent = detect_intent(text)

    depression = detect_depression(text)

    suicide = suicide_detector.assess_risk(

        text=text,

        emotion=emotion["emotion"],

        intent=intent["intent"],

        depression_risk=depression["depression_risk"]

    )

    # =====================================================
    # Conversation State
    # =====================================================

    state = state_tracker.update_state(

        text=text,

        emotion=emotion["emotion"],

        intent=intent["intent"],

        depression_risk=depression["depression_risk"],

        suicide_risk_level=suicide["suicide_risk_level"]

    )

    analysis = state["analysis"]

    history = state["conversation"]["history"]

    turn = state["conversation"]["turn"]

    # =====================================================
    # Conversation Manager
    # =====================================================

    conversation_manager.start_new_turn()

    conversation_manager.update_stage(

        phq_tracker.get_covered_domains()

    )

    conversation = {

    **conversation_manager.get_state(),

    "history": history

}

    # =====================================================
    # Temporary Assessment
    # =====================================================

    assessment = {

        "covered_domains":

            phq_tracker.get_covered_domains(),

        "remaining_domains":

            phq_tracker.get_remaining_domains(),

        "current_domain":

            conversation["current_domain"]

    }

    # =====================================================
    # Clinical Reasoning
    # =====================================================

    context = {

        "analysis": analysis,

        "conversation": conversation,

        "assessment": assessment

    }
    print(conversation)

    reasoning = reasoning_engine.decide(

        context

    )

    # =====================================================
    # Apply Reasoning
    # =====================================================

    if reasoning["target_domain"] is not None:

        conversation_manager.set_current_domain(

            reasoning["target_domain"]

        )

        phq_tracker.update(

            domain=reasoning["target_domain"],

            turn=turn

        )

    #conversation = conversation_manager.get_state()
    
    conversation = {

    **conversation_manager.get_state(),

    "history": history

    }

    covered_domains = phq_tracker.get_covered_domains()

    remaining_domains = phq_tracker.get_remaining_domains()

    # update stage again after PHQ changes

    conversation_manager.update_stage(

        covered_domains

    )

    conversation = {

    **conversation_manager.get_state(),

    "history": history

}

    assessment = {

        "covered_domains": covered_domains,

        "remaining_domains": remaining_domains,

        "current_domain": conversation["current_domain"]

    }

    context["conversation"] = conversation

    context["assessment"] = assessment

    context["reasoning"] = reasoning

    # =====================================================
    # Generate Question
    # =====================================================

    question = llm_engine.generate_question(

        reasoning,

        analysis,

        history

    )

    conversation_manager.question_asked()

    state_tracker.save_assistant_question(

        question

    )

    # =====================================================
    # Final Response
    # =====================================================

    return {

        "analysis": {

            "emotion": emotion,

            "intent": intent,

            "depression": depression,

            "suicide": suicide,

            "severity": analysis["severity"]

        },

        "assessment": assessment,

        "conversation": conversation_manager.get_state(),

        "reasoning": reasoning,

        "assistant": {

            "question": question

        }

    }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    samples = [

        "I can't sleep at night and I feel exhausted all the time.",

        "Nobody talks to me anymore and I feel lonely.",

        "I am worried about my exams and future.",

        "I don't see any purpose in life anymore.",

        "I want to kill myself."

    ]

    from pprint import pprint

    for text in samples:

        print()

        print("=" * 80)

        print()

        print("USER:")

        print(text)

        print()

        result = process_message(text)

        print("PIPELINE OUTPUT:")

        pprint(result)