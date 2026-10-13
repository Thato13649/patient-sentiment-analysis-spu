from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from aspect_analysis import find_aspects, determine_aspect_sentiment


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

model_path = "./distilbert_patient_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# ============================================================
# ROOT CAUSE KEYWORDS
# ============================================================

ROOT_CAUSES = {

    "Doctor": {
        "Rude behaviour": [
            "rude doctor",
            "rude doctors",
            "doctor was rude",
            "doctor is rude",
            "doctor rude",
            "rude behaviour"
        ],

        "Poor treatment": [
            "poor treatment",
            "bad treatment",
            "improper treatment"
        ]
    },


    "Staff": {
        "Rude staff": [
            "rude staff",
            "staff was rude",
            "staff were rude",
            "staff rude",
            "rude behaviour"
        ],

        "Poor service": [
            "poor service",
            "bad service",
            "terrible service"
        ]
    },


    "Waiting Time": {
        "Long waiting": [
            "waited for",
            "wait for",
            "waiting for",
            "made to wait",
            "had to wait",
            "hours",
            "hour"
        ],

        "Delays": [
            "delay",
            "delayed",
            "late"
        ]
    },


    "Cleanliness": {
        "Dirty toilets": [
            "dirty toilet",
            "dirty toilets",
            "toilet was dirty",
            "toilets were dirty",
            "toilet is dirty",
            "toilets are dirty",
            "blocked toilet",
            "blocked toilets",
            "unclean toilet",
            "unclean toilets"
        ],

        "Poor hygiene": [
            "poor hygiene",
            "hygiene conditions",
            "unclean"
        ],

        "Pests": [
            "rat",
            "rats",
            "cockroach",
            "cockroaches",
            "pest",
            "pests"
        ]
    },


    "Facilities": {
        "Poor facilities": [
            "poor facility",
            "poor facilities",
            "bad facilities",
            "worst facility",
            "worst facilities"
        ]
    },


    "Treatment": {
        "Poor treatment": [
            "poor treatment",
            "no proper treatment",
            "bad treatment",
            "improper treatment"
        ],

        "Unnecessary tests": [
            "unwanted test",
            "unwanted tests",
            "unnecessary test",
            "unnecessary tests"
        ]
    },


    "Communication": {
        "Poor communication": [
            "poor communication",
            "no communication",
            "bad communication",
            "lack of communication"
        ]
    },


    "Emergency": {
        "Emergency service problems": [
            "worst emergency",
            "emergency service",
            "emergency department",
            "emergency ward"
        ]
    }
}


# ============================================================
# ROOT CAUSE DETECTION
# ============================================================

def find_root_causes(review, detected_aspects):

    review_lower = review.lower()

    results = []

    for aspect in detected_aspects:

        # Only analyse negative aspects
        if detected_aspects[aspect] != "Negative":
            continue


        # ====================================================
        # CLEANLINESS
        # ====================================================

        if aspect == "Cleanliness":

            # Dirty toilets
            if (
                ("toilet" in review_lower or "toilets" in review_lower)
                and "dirty" in review_lower
            ):
                results.append(
                    ("Cleanliness", "Dirty toilets")
                )
                continue


            # Blocked toilets
            if (
                ("toilet" in review_lower or "toilets" in review_lower)
                and "blocked" in review_lower
            ):
                results.append(
                    ("Cleanliness", "Dirty toilets")
                )
                continue


            # Pests
            if any(
                word in review_lower
                for word in [
                    "rat",
                    "rats",
                    "cockroach",
                    "cockroaches",
                    "pest",
                    "pests"
                ]
            ):
                results.append(
                    ("Cleanliness", "Pests")
                )
                continue


            # Poor hygiene
            if any(
                phrase in review_lower
                for phrase in [
                    "poor hygiene",
                    "hygiene conditions",
                    "unclean"
                ]
            ):
                results.append(
                    ("Cleanliness", "Poor hygiene")
                )
                continue


            continue


        # ====================================================
        # OTHER ROOT CAUSES
        # ====================================================

        if aspect not in ROOT_CAUSES:
            continue


        for cause, phrases in ROOT_CAUSES[aspect].items():

            for phrase in phrases:

                if phrase in review_lower:

                    results.append(
                        (aspect, cause)
                    )

                    break


    return results


# ============================================================
# APPLICATION TITLE
# ============================================================

print("=" * 55)
print("          PATIENT EXPERIENCE ANALYSER")
print("=" * 55)

print("\nEnter a patient review to analyse.")
print("Type 'quit' to exit the application.")


# ============================================================
# MAIN APPLICATION
# ============================================================

while True:

    review = input("\nEnter a patient review: ")


    # Exit
    if review.lower().strip() in ["quit", "exit"]:

        print("\nThank you for using the Patient Experience Analyser.")
        break


    # Empty review
    if review.strip() == "":

        print("Please enter a review.")
        continue


    # ========================================================
    # SENTIMENT ANALYSIS
    # ========================================================

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )


    with torch.no_grad():

        outputs = model(**inputs)


    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )


    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()


    confidence = (
        probabilities[0][prediction].item()
        * 100
    )


    if prediction == 1:

        sentiment = "POSITIVE"

    else:

        sentiment = "NEGATIVE"


    # ========================================================
    # ASPECT ANALYSIS
    # ========================================================

    aspects = find_aspects(review)

    aspect_results = {}


    for aspect in aspects:

        aspect_sentiment = determine_aspect_sentiment(
            review,
            aspect
        )

        aspect_results[aspect] = aspect_sentiment


    # ========================================================
    # ROOT CAUSE ANALYSIS
    # ========================================================

    root_causes = find_root_causes(
        review,
        aspect_results
    )


    # ========================================================
    # DISPLAY REVIEW
    # ========================================================

    print("\n======================================================")
    print("                 PATIENT REVIEW")
    print("======================================================")

    print(review)


    # ========================================================
    # DISPLAY SENTIMENT
    # ========================================================

    print("\n------------------------------------------------------")
    print("              SENTIMENT ANALYSIS")
    print("------------------------------------------------------")

    print(f"Sentiment:  {sentiment}")
    print(f"Confidence: {confidence:.2f}%")


    # ========================================================
    # DISPLAY ASPECTS
    # ========================================================

    print("\n------------------------------------------------------")
    print("                 ASPECT ANALYSIS")
    print("------------------------------------------------------")


    if aspect_results:

        for aspect, result in aspect_results.items():

            print(
                f"{aspect}: {result}"
            )

    else:

        print(
            "No specific aspects detected."
        )


    # ========================================================
    # DISPLAY ROOT CAUSES
    # ========================================================

    print("\n------------------------------------------------------")
    print("               ROOT CAUSE ANALYSIS")
    print("------------------------------------------------------")


    if root_causes:

        for aspect, cause in root_causes:

            print(
                f"{aspect} -> {cause}"
            )

    else:

        print(
            "No specific root cause detected."
        )


    print("\n======================================================")