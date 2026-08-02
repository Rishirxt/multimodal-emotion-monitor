"""
clinical_knowledge.py

Central knowledge base for the adaptive depression screening system.

The reasoning engine should NEVER hardcode PHQ logic.

Everything about the interview should come from this file.
"""

from copy import deepcopy


PHQ_KNOWLEDGE = {

    "sleep": {

        "priority": 1,

        "goal":
            "Assess sleep disturbance and its impact.",

        "transition_candidates": [

            "energy",

            "concentration"

        ],

        "llm_focus":
            "Understand sleep duration, quality and daytime impact.",

        "completion_rule": "ALL",

        "objectives": {

            "duration": {

                "completed": False,

                "importance": "high"
            },

            "frequency": {

                "completed": False,

                "importance": "medium"
            },

            "quality": {

                "completed": False,

                "importance": "high"
            },

            "daytime_impact": {

                "completed": False,

                "importance": "high"
            }
        },

        "keywords": [

            "sleep",

            "insomnia",

            "night",

            "awake",

            "rest",

            "bed"

        ]
    },

    "energy": {

        "priority": 2,

        "goal":
            "Assess fatigue and motivation.",

        "transition_candidates": [

            "interest",

            "mood"

        ],

        "llm_focus":
            "Understand physical and mental exhaustion.",

        "completion_rule": "ALL",

        "objectives": {

            "fatigue": {

                "completed": False,

                "importance": "high"
            },

            "motivation": {

                "completed": False,

                "importance": "high"
            },

            "daily_function": {

                "completed": False,

                "importance": "medium"
            }
        },

        "keywords": [

            "tired",

            "fatigue",

            "energy",

            "exhausted"

        ]
    },

    "interest": {

        "priority": 3,

        "goal":
            "Assess loss of interest and pleasure.",

        "transition_candidates": [

            "mood",

            "self_worth"

        ],

        "llm_focus":
            "Understand enjoyment and hobbies.",

        "completion_rule": "ALL",

        "objectives": {

            "loss_of_interest": {

                "completed": False,

                "importance": "high"
            },

            "social_interest": {

                "completed": False,

                "importance": "medium"
            },

            "hobbies": {

                "completed": False,

                "importance": "medium"
            }
        },

        "keywords": [

            "interest",

            "enjoy",

            "hobby",

            "motivation"

        ]
    },

    "mood": {

        "priority": 4,

        "goal":
            "Assess persistent low mood.",

        "transition_candidates": [

            "self_worth",

            "appetite"

        ],

        "llm_focus":
            "Understand sadness and emotional state.",

        "completion_rule": "ALL",

        "objectives": {

            "sadness": {

                "completed": False,

                "importance": "high"
            },

            "hopelessness": {

                "completed": False,

                "importance": "high"
            },

            "duration": {

                "completed": False,

                "importance": "medium"
            }
        },

        "keywords": [

            "sad",

            "hopeless",

            "lonely",

            "empty"

        ]
    },

    "appetite": {

        "priority": 5,

        "goal":
            "Assess appetite changes.",

        "transition_candidates": [

            "movement"

        ],

        "llm_focus":
            "Understand eating behaviour.",

        "completion_rule": "ALL",

        "objectives": {

            "loss_of_appetite": {

                "completed": False,

                "importance": "high"
            },

            "overeating": {

                "completed": False,

                "importance": "medium"
            }
        },

        "keywords": [

            "eat",

            "hungry",

            "food",

            "appetite"

        ]
    },

    "concentration": {

        "priority": 6,

        "goal":
            "Assess concentration problems.",

        "transition_candidates": [

            "self_worth"

        ],

        "llm_focus":
            "Understand focus and attention.",

        "completion_rule": "ALL",

        "objectives": {

            "focus": {

                "completed": False,

                "importance": "high"
            },

            "memory": {

                "completed": False,

                "importance": "medium"
            }
        },

        "keywords": [

            "focus",

            "study",

            "memory",

            "concentrate"

        ]
    },

    "movement": {

        "priority": 7,

        "goal":
            "Assess psychomotor changes.",

        "transition_candidates": [

            "self_worth"

        ],

        "llm_focus":
            "Understand restlessness or slowing.",

        "completion_rule": "ALL",

        "objectives": {

            "restlessness": {

                "completed": False,

                "importance": "high"
            },

            "slowing": {

                "completed": False,

                "importance": "high"
            }
        },

        "keywords": [

            "slow",

            "restless",

            "pace"

        ]
    },

    "self_worth": {

        "priority": 8,

        "goal":
            "Assess guilt and self-worth.",

        "transition_candidates": [

            "suicidal_thoughts"

        ],

        "llm_focus":
            "Understand guilt and self-esteem.",

        "completion_rule": "ALL",

        "objectives": {

            "worthlessness": {

                "completed": False,

                "importance": "high"
            },

            "guilt": {

                "completed": False,

                "importance": "high"
            }
        },

        "keywords": [

            "worthless",

            "failure",

            "guilt"

        ]
    },

    "suicidal_thoughts": {

        "priority": 9,

        "goal":
            "Assess immediate safety.",

        "transition_candidates": [],

        "llm_focus":
            "Assess current safety only.",

        "completion_rule": "ALL",

        "objectives": {

            "thoughts": {

                "completed": False,

                "importance": "high"
            },

            "plan": {

                "completed": False,

                "importance": "high"
            },

            "intent": {

                "completed": False,

                "importance": "critical"
            }
        },

        "keywords": [

            "kill",

            "suicide",

            "die",

            "end my life"

        ]
    }

}


def get_domain(domain):

    return deepcopy(PHQ_KNOWLEDGE.get(domain))


def get_all_domains():

    return deepcopy(PHQ_KNOWLEDGE)


if __name__ == "__main__":

    from pprint import pprint

    pprint(get_domain("sleep"))