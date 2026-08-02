"""
conversation_manager.py

Maintains interview state.

This class DOES NOT perform clinical reasoning.

Responsibilities:

- Track interview stage
- Track current domain
- Track question count
- Track interview turns

The ClinicalReasoningEngine decides
WHEN to switch domains.
"""


class ConversationManager:

    def __init__(self):

        self.turn = 0

        self.stage = "initial_assessment"

        self.current_domain = None

        self.questions_on_current_domain = 0

    def update_stage(self, covered_domains):

        covered = len(covered_domains)

        if covered < 2:

            self.stage = "initial_assessment"

        elif covered < 6:

            self.stage = "symptom_exploration"

        else:

            self.stage = "summary"

    def start_new_turn(self):

        self.turn += 1

    def set_current_domain(self, domain):

        if domain != self.current_domain:

            self.current_domain = domain

            self.questions_on_current_domain = 0

    def question_asked(self):

        self.questions_on_current_domain += 1

    def get_state(self):

        return {

            "stage": self.stage,

            "current_domain": self.current_domain,

            "questions_on_current_domain":
                self.questions_on_current_domain,

            "turn": self.turn
        }


if __name__ == "__main__":

    manager = ConversationManager()

    manager.start_new_turn()

    manager.update_stage(["sleep"])

    manager.set_current_domain("sleep")

    print(manager.get_state())

    manager.question_asked()

    print(manager.get_state())

    manager.question_asked()

    print(manager.get_state())

    manager.set_current_domain("energy")

    print(manager.get_state())