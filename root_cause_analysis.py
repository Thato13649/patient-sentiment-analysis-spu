import pandas as pd
import re

from aspect_analysis import (
    find_aspects,
    determine_aspect_sentiment
)


# ============================================================
# ROOT CAUSE DEFINITIONS
# ============================================================

ROOT_CAUSES = {

    "Waiting Time": {

        "Long waiting": [
            "waited for",
            "waited",
            "waiting for",
            "waiting time",
            "long wait",
            "long waiting",
            "hours to",
            "hours before",
            "hours n hours",
            "kept waiting",
            "made to wait",
            "had to wait",
            "still waiting",
            "wait for one hour",
            "wait for 1 hour",
            "wait for 2 hrs",
            "wait for 3 hrs",
            "waited for one hour",
            "waited for 1 hour",
            "waited for 2 hours",
            "waited for 3 hours",
            "waited for more than",
            "waited for hours"
        ],

        "Delays": [
            "delay",
            "delayed",
            "delays",
            "late",
            "slow service",
            "slow process",
            "slow billing",
            "slow procedure",
            "things delay",
            "everything delay"
        ]
    },


    "Staff": {

        "Rude staff": [
            "rude staff",
            "rude nurse",
            "rude nurses",
            "rude receptionist",
            "rude reception",
            "rude security",
            "rude behaviour",
            "rude behavior",
            "unfriendly staff",
            "unprofessional staff",
            "disrespectful staff",
            "disrespectful nurse",
            "disrespectful receptionist",
            "rude doctors",
            "rude doctor",
            "rude physiotherapists"
        ],

        "Unhelpful staff": [
            "unhelpful staff",
            "unhelpful nurse",
            "unhelpful receptionist",
            "not helpful",
            "couldn't help",
            "could not help",
            "refused to help",
            "staff refused"
        ]
    },


    "Cleanliness": {

        "Poor hygiene": [
            "poor hygiene",
            "poor hygiene conditions",
            "hygiene conditions are poor",
            "hygiene was poor",
            "poor cleanliness",
            "unclean hospital",
            "unclean environment",
            "dirty hospital",
            "hospital was dirty",
            "filthy hospital"
        ],

        "Unclean rooms": [
            "dirty room",
            "dirty rooms",
            "room was dirty",
            "rooms were dirty",
            "room was not cleaned",
            "rooms were not cleaned",
            "room not cleaned",
            "rooms not cleaned",
            "unclean room",
            "unclean rooms"
        ],

        "Pests": [
            "rats",
            "rat infestation",
            "rat problem",
            "cockroaches",
            "cockroach infestation",
            "cockroach problem",
            "insect infestation",
            "insects infestation",
            "pest infestation",
            "pest problem",
            "pests"
        ],

        "Dirty toilets": [
            "dirty toilet",
            "dirty toilets",
            "toilet was dirty",
            "toilets were dirty",
            "blocked toilet",
            "blocked toilets",
            "toilet was blocked",
            "toilets were blocked",
            "dirty bathroom",
            "dirty bathrooms",
            "bathroom was dirty",
            "bathrooms were dirty",
            "blocked bathroom",
            "blocked bathrooms"
        ]
    },


    "Treatment": {

        "Poor treatment": [
            "poor treatment",
            "bad treatment",
            "worst treatment",
            "poor medical treatment",
            "bad medical treatment",
            "no proper treatment",
            "proper treatment was not",
            "inadequate treatment"
        ],

        "Unnecessary tests": [
            "unwanted test",
            "unwanted tests",
            "unnecessary test",
            "unnecessary tests",
            "unwanted tests and procedures",
            "unnecessary tests and procedures"
        ]
    },


    "Facilities": {

        "Poor facilities": [
            "poor facility",
            "poor facilities",
            "bad facility",
            "bad facilities",
            "worst facility",
            "worst facilities",
            "poor hospital facilities",
            "bad hospital facilities"
        ],

        "Equipment problems": [
            "broken equipment",
            "broken machine",
            "equipment was broken",
            "equipment was not working",
            "equipment not working",
            "equipment doesn't work",
            "equipment does not work",
            "machine was broken",
            "machine was not working",
            "machine not working",
            "faulty equipment",
            "faulty machine"
        ]
    },


    "Emergency": {

        "Emergency service problems": [
            "worst emergency service",
            "poor emergency service",
            "bad emergency service",
            "emergency service was poor",
            "emergency service was bad",
            "worst emergency department",
            "poor emergency department",
            "bad emergency department",
            "worst emergency room",
            "poor emergency room",
            "bad emergency room"
        ],

        "Emergency delays": [
            "emergency delay",
            "emergency delays",
            "waited in emergency",
            "waiting in emergency",
            "kept waiting in emergency",
            "waited for hours in emergency",
            "waiting for hours in emergency"
        ]
    },


    "Communication": {

        "Poor communication": [
            "poor communication",
            "bad communication",
            "no communication",
            "lack of communication",
            "communication was poor",
            "communication was bad"
        ],

        "Poor explanation": [
            "not explained",
            "wasn't explained",
            "was not explained",
            "no explanation",
            "poor explanation",
            "bad explanation",
            "not informed",
            "wasn't informed",
            "was not informed"
        ]
    }
}


# ============================================================
# ASPECT KEYWORDS
# ============================================================

ASPECT_KEYWORDS = {

    "Doctor": [
        "doctor",
        "dr",
        "physician"
    ],

    "Staff": [
        "staff",
        "nurse",
        "nurses",
        "receptionist",
        "reception",
        "security"
    ],

    "Waiting Time": [
        "wait",
        "waiting",
        "waited",
        "delay",
        "delayed",
        "queue"
    ],

    "Cleanliness": [
        "clean",
        "dirty",
        "toilet",
        "toilets",
        "bathroom",
        "hygiene",
        "filthy",
        "unclean",
        "rats",
        "rat",
        "cockroach",
        "cockroaches"
    ],

    "Facilities": [
        "facility",
        "facilities",
        "room",
        "bed",
        "building",
        "equipment",
        "machine"
    ],

    "Communication": [
        "communication",
        "communicate",
        "explained",
        "explanation",
        "information",
        "informed"
    ],

    "Treatment": [
        "treatment",
        "care",
        "medicine",
        "medication",
        "surgery",
        "operation"
    ],

    "Emergency": [
        "emergency",
        "ambulance",
        "emergency department",
        "emergency room"
    ]
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def phrase_exists(text, phrase):

    pattern = (
        r"(?<!\w)"
        + re.escape(phrase.lower())
        + r"(?!\w)"
    )

    return re.search(
        pattern,
        text.lower()
    ) is not None


def find_aspect_position(review, aspect):

    review_lower = review.lower()

    for keyword in ASPECT_KEYWORDS.get(
        aspect,
        []
    ):

        pattern = (
            r"(?<!\w)"
            + re.escape(keyword)
            + r"(?!\w)"
        )

        match = re.search(
            pattern,
            review_lower
        )

        if match:
            return match.start()

    return -1


def get_local_context(review, position):

    if position == -1:
        return review.lower()

    start = max(
        0,
        position - 120
    )

    end = min(
        len(review),
        position + 220
    )

    return review[
        start:end
    ].lower()


def find_root_causes(review, aspect):

    position = find_aspect_position(
        review,
        aspect
    )

    context = get_local_context(
        review,
        position
    )

    results = []

    if aspect not in ROOT_CAUSES:
        return results

    for cause, keywords in ROOT_CAUSES[
        aspect
    ].items():

        for keyword in keywords:

            if phrase_exists(
                context,
                keyword
            ):

                results.append({
                    "cause": cause,
                    "evidence": keyword
                })

                break

    return results


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv(
    "hospital.csv"
)


# Make sure feedback is text
data["Feedback"] = (
    data["Feedback"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

data = data[
    data["Feedback"] != ""
]

data = data.drop_duplicates(
    subset=["Feedback"],
    keep="first"
)

data = data.reset_index(
    drop=True
)


print(
    f"\nAnalysing {len(data)} unique patient reviews..."
)


# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

root_cause_results = {}


for review in data["Feedback"]:

    aspects = find_aspects(
        review
    )

    for aspect in aspects:

        sentiment = determine_aspect_sentiment(
            review,
            aspect
        )

        # Only analyse negative aspects
        if sentiment != "Negative":
            continue

        causes = find_root_causes(
            review,
            aspect
        )

        # Prevent the same review from
        # being counted twice for the
        # same root cause.
        causes_already_found = set()

        for result in causes:

            cause = result["cause"]
            evidence = result["evidence"]

            if cause in causes_already_found:
                continue

            causes_already_found.add(
                cause
            )

            key = (
                aspect,
                cause
            )

            if key not in root_cause_results:

                root_cause_results[key] = {

                    "aspect": aspect,

                    "cause": cause,

                    "count": 0,

                    "examples": [],

                    "evidence": []
                }

            root_cause_results[
                key
            ]["count"] += 1


            # Store up to 3 unique examples
            if (
                review
                not in root_cause_results[
                    key
                ]["examples"]
                and
                len(
                    root_cause_results[
                        key
                    ]["examples"]
                ) < 3
            ):

                root_cause_results[
                    key
                ]["examples"].append(
                    review
                )


            # Store evidence phrase
            if (
                evidence
                not in root_cause_results[
                    key
                ]["evidence"]
            ):

                root_cause_results[
                    key
                ]["evidence"].append(
                    evidence
                )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("===================================")
print("       ROOT CAUSE FINDINGS")
print("===================================")


if not root_cause_results:

    print(
        "\nNo root causes were identified."
    )


else:

    for key, information in (
        root_cause_results.items()
    ):

        print("\n-----------------------------------")

        print(
            f"Aspect: "
            f"{information['aspect']}"
        )

        print(
            f"Root Cause: "
            f"{information['cause']}"
        )

        print(
            f"Occurrences: "
            f"{information['count']}"
        )

        print("\nEvidence detected:")

        for evidence in information[
            "evidence"
        ]:

            print(
                f"- \"{evidence}\""
            )

        print("\nExample patient feedback:")

        for example in information[
            "examples"
        ]:

            print(
                f"- {example}"
            )


print("\n")
print("===================================")
print("       ANALYSIS COMPLETE")
print("===================================")
# ============================================================
# ROOT CAUSE SUMMARY
# ============================================================

print("\n")
print("===================================")
print("       ROOT CAUSE SUMMARY")
print("===================================")

summary = {}

for key, information in root_cause_results.items():

    aspect = information["aspect"]
    cause = information["cause"]
    count = information["count"]

    if aspect not in summary:
        summary[aspect] = []

    summary[aspect].append(
        (cause, count)
    )


for aspect, causes in summary.items():

    print(f"\n{aspect}")

    # Sort from highest occurrence to lowest
    causes.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for cause, count in causes:

        print(
            f"  {cause}: {count}"
        )


print("\n")
print("===================================")
print("       SUMMARY COMPLETE")
print("===================================")