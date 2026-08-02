class InformationEstimator:

    """
    Calculates interview completeness from
    extracted objectives.
    """

    def estimate(self, extracted_objectives):

        total = len(extracted_objectives)

        if total == 0:

            return {

                "completion": 0,

                "completed_objectives": [],

                "remaining_objectives": []

            }

        completed = [

            key

            for key, value

            in extracted_objectives.items()

            if value

        ]

        remaining = [

            key

            for key, value

            in extracted_objectives.items()

            if not value

        ]

        completion = len(completed) / total

        return {

            "completion": round(completion, 2),

            "completed_objectives": completed,

            "remaining_objectives": remaining

        }


if __name__ == "__main__":

    estimator = InformationEstimator()

    result = estimator.estimate(

        {

            "duration": True,

            "frequency": False,

            "quality": True,

            "daytime_impact": False

        }

    )

    from pprint import pprint

    pprint(result)