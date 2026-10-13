import pandas as pd
import matplotlib.pyplot as plt

from aspect_analysis import (
    ASPECT_KEYWORDS,
    find_aspects,
    determine_aspect_sentiment
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("hospital.csv")

df["Feedback"] = df["Feedback"].astype(str).str.strip()

df = df[df["Feedback"] != ""]

# Keep unique patient reviews
df = df.drop_duplicates(
    subset=["Feedback"],
    keep="first"
)

print(
    f"\nAnalysing {len(df)} unique patient reviews..."
)


# ============================================================
# ASPECT RESULTS
# ============================================================

aspect_results = {}

for aspect in ASPECT_KEYWORDS:

    aspect_results[aspect] = {
        "total": 0,
        "negative": 0
    }


# ============================================================
# ANALYSE EACH REVIEW
# ============================================================

for feedback in df["Feedback"]:

    aspects = find_aspects(feedback)

    for aspect in aspects:

        aspect_results[aspect]["total"] += 1

        sentiment = determine_aspect_sentiment(
            feedback,
            aspect
        )

        if sentiment == "Negative":

            aspect_results[aspect]["negative"] += 1


# ============================================================
# CALCULATE NEGATIVE RATES
# ============================================================

chart_data = []

for aspect, values in aspect_results.items():

    total = values["total"]
    negative = values["negative"]

    if total > 0:

        negative_rate = (
            negative / total
        ) * 100

        chart_data.append(
            (
                aspect,
                negative_rate,
                total,
                negative
            )
        )


# Highest negative rate first
chart_data.sort(
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("===================================")
print("       ASPECT ANALYSIS")
print("===================================")

for aspect, rate, total, negative in chart_data:

    print(
        f"{aspect}: "
        f"{rate:.1f}% negative "
        f"({negative}/{total})"
    )


# ============================================================
# CREATE CHART
# ============================================================

aspects = [
    item[0]
    for item in chart_data
]

rates = [
    item[1]
    for item in chart_data
]


plt.figure(figsize=(12, 7))

bars = plt.barh(
    aspects[::-1],
    rates[::-1]
)

plt.xlabel("Negative Feedback (%)")
plt.ylabel("Hospital Aspect")

plt.title(
    "Negative Patient Feedback by Hospital Aspect"
)

plt.xlim(0, 100)

plt.tight_layout()


# ============================================================
# ADD VALUES TO BARS
# ============================================================

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center"
    )


# ============================================================
# SAVE CHART
# ============================================================

plt.savefig(
    "aspect_negative_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nAspect chart created successfully!")
print("Saved as: aspect_negative_rate.png")