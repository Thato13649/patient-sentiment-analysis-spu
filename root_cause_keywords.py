import pandas as pd
import re
from aspect_analysis import (
    find_aspects,
    determine_aspect_sentiment
)


# Possible root causes for each hospital aspect
ROOT_CAUSES = {

    "Waiting Time": {
        "Long waiting": [
            "wait",
            "waiting",
            "waited",
            "long wait",
            "hours",
            "queue"
        ],

        "Delays": [
            "delay",
            "delayed",
            "late",
            "slow"
        ]
    },

    "Staff": {
        "Rude staff": [
            "rude",
            "unfriendly",
            "unprofessional",
            "disrespectful"
        ],

        "Unhelpful staff": [
            "unhelpful",
            "couldn't help",
            "could not help",
            "not helpful"
        ]
    },

    "Cleanliness": {
        "Poor hygiene": [
            "hygiene",
            "dirty",
            "filthy",
            "not clean",
            "unclean"
        ],

        "Unclean rooms": [
            "room was not cleaned",
            "room not cleaned",
            "not cleaned",
            "dirty room"
        ],

        "Pests": [
            "rat",
            "rats",
            "cockroach",
            "cockroaches",
            "insects",
            "pest"
        ],

        "Dirty toilets": [
            "toilet",
            "toilets",
            "bathroom",
            "bathrooms"
        ]
    },

    "Treatment": {
        "Poor treatment": [
            "poor treatment",
            "bad treatment",
            "worst treatment",
            "no proper treatment",
            "inadequate treatment"
        ],

        "Unnecessary tests": [
            "unwanted test",
            "unwanted tests",
            "unnecessary test",
            "unnecessary tests"
        ]
    },

    "Facilities": {
        "Poor facilities": [
            "worst facility",
            "poor facility",
            "poor facilities",
            "bad facility",
            "bad facilities"
        ],

        "Equipment problems": [
            "equipment",
            "machine",
            "machines"
        ]
    },

    "Emergency": {
        "Emergency service problems": [
            "emergency room",
            "emergency department",
            "emergency service"
        ],

        "Emergency delays": [
            "emergency delay",
            "waited in emergency",
            "waiting in emergency"
        ]
    },

    "Communication": {
        "Poor communication": [
            "poor communication",
            "bad communication",
            "no communication"
        ],

        "Poor explanation": [
            "not explained",
            "wasn't explained",
            "was not explained",
            "no explanation",
            "not informed"
        ]
    }
}


# Load the dataset
data = pd.read_csv("hospital.csv")

data["Feedback"] = (
    data["Feedback"]
    .fillna("")
    .astype(str)
)


# Store root causes
root_cause_results = {}


# Analyse every review
for review in data["Feedback"]:

    aspects = find_aspects(review)

    for aspect in aspects:

        sentiment = determine_aspect_sentiment(
            review,
            aspect
        )

        # We only want negative reviews
        if sentiment != "Negative":
            continue

        review_lower = review.lower()

        # Check possible root causes
        if aspect in ROOT_CAUSES:

            for cause, keywords in ROOT_CAUSES[aspect].items():

                for keyword in keywords:

                    if keyword in review_lower:

                        if cause not in root_cause_results:

                            root_cause_results[cause] = {
                                "aspect": aspect,
                                "count": 0,
                                "examples": []
                            }

                        root_cause_results[cause]["count"] += 1

                        # Keep up to 3 examples
                        if (
                            len(
                                root_cause_results[cause]["examples"]
                            ) < 3
                        ):

                            if review not in root_cause_results[cause]["examples"]:

                                root_cause_results[cause]["examples"].append(
                                    review
                                )

                        break


# Display results
print("\n===================================")
print("       ROOT CAUSE FINDINGS")
print("===================================\n")


if not root_cause_results:

    print("No specific root causes were identified.")

else:

    for cause, information in root_cause_results.items():

        print("-----------------------------------")

        print(
            f"Aspect: {information['aspect']}"
        )

        print(
            f"Root Cause: {cause}"
        )

        print(
            f"Occurrences: {information['count']}"
        )

        print("\nExample patient feedback:")

        for example in information["examples"]:

            print(f"- {example}")

        print()