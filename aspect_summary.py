import pandas as pd

from aspect_analysis import (
    find_aspects,
    determine_aspect_sentiment
)


# Load the hospital dataset
data = pd.read_csv("hospital.csv")


# Make sure the review column is treated as text
data["Feedback"] = data["Feedback"].fillna("").astype(str)


# Store our results
summary = {}


# Analyse every review
for review in data["Feedback"]:

    aspects = find_aspects(review)

    for aspect in aspects:

        sentiment = determine_aspect_sentiment(
            review,
            aspect
        )

        # Create the aspect if it does not exist yet
        if aspect not in summary:

            summary[aspect] = {
                "Positive": 0,
                "Negative": 0,
                "Unknown": 0
            }

        # Count the sentiment
        if sentiment in summary[aspect]:

            summary[aspect][sentiment] += 1


# Display the results
print("\n===================================")
print("       PATIENT ASPECT SUMMARY")
print("===================================\n")


for aspect, counts in summary.items():

    print(f"{aspect}")

    print(
        f"  Positive: {counts['Positive']}"
    )

    print(
        f"  Negative: {counts['Negative']}"
    )

    print(
        f"  Unknown:  {counts['Unknown']}"
    )

    print()