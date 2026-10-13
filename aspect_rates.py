import pandas as pd

from aspect_analysis import (
    find_aspects,
    determine_aspect_sentiment
)


# Load the hospital dataset
data = pd.read_csv("hospital.csv")

# Make sure reviews are treated as text
data["Feedback"] = data["Feedback"].fillna("").astype(str)


# Store the results
summary = {}


# Analyse every review
for review in data["Feedback"]:

    aspects = find_aspects(review)

    for aspect in aspects:

        sentiment = determine_aspect_sentiment(
            review,
            aspect
        )

        # Create the aspect if it does not exist
        if aspect not in summary:

            summary[aspect] = {
                "Positive": 0,
                "Negative": 0,
                "Unknown": 0
            }

        # Count the sentiment
        if sentiment in summary[aspect]:

            summary[aspect][sentiment] += 1


# Display negative rates
print("\n===================================")
print("       ASPECT NEGATIVE RATES")
print("===================================\n")


for aspect, counts in summary.items():

    positive = counts["Positive"]
    negative = counts["Negative"]

    classified = positive + negative

    if classified > 0:

        negative_rate = (
            negative / classified
        ) * 100

    else:

        negative_rate = 0


    print(f"{aspect}")

    print(f"  Positive: {positive}")

    print(f"  Negative: {negative}")

    print(
        f"  Negative Rate: "
        f"{negative_rate:.1f}%"
    )

    print()