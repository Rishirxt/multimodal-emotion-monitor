from clinical_knowledge import get_all_domains


class PHQTracker:
    """
    Tracks which PHQ-9 domains have been explored.

    This class contains NO reasoning.
    It simply records interview progress.
    """

    def __init__(self):

        knowledge = get_all_domains()

        self.domains = {

            domain: {

                "covered": False,

                "times_discussed": 0,

                "last_turn": None

            }

            for domain in knowledge

        }

    def update(self, domain, turn):

        """
        Mark a PHQ domain as discussed.
        """

        if domain is None:
            return

        if domain not in self.domains:
            return

        self.domains[domain]["covered"] = True
        self.domains[domain]["times_discussed"] += 1
        self.domains[domain]["last_turn"] = turn

    def get_covered_domains(self):

        return [

            domain

            for domain, info in self.domains.items()

            if info["covered"]

        ]

    def get_remaining_domains(self):

        return [

            domain

            for domain, info in self.domains.items()

            if not info["covered"]

        ]

    def get_domain_info(self, domain):

        return self.domains.get(domain)

    def get_state(self):

        return self.domains


if __name__ == "__main__":

    tracker = PHQTracker()

    tracker.update(

        domain="sleep",

        turn=1

    )

    tracker.update(

        domain="sleep",

        turn=2

    )

    tracker.update(

        domain="energy",

        turn=3

    )

    from pprint import pprint

    pprint(tracker.get_state())

    print("\nCovered:")

    print(tracker.get_covered_domains())

    print("\nRemaining:")

    print(tracker.get_remaining_domains())