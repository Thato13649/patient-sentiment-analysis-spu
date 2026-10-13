import matplotlib.pyplot as plt


# ============================================================
# DATA FROM YOUR ANALYSIS
# ============================================================

model_names = [
    "VADER",
    "TF-IDF + Logistic Regression",
    "DistilBERT"
]

accuracy = [86, 81, 91]
precision = [89, 80, 95]
recall = [93, 99, 93]
f1_score = [91, 88, 94]


aspect_data = {
    "Waiting Time": 97.7,
    "Emergency": 78.9,
    "Communication": 28.6,
    "Facilities": 28.1,
    "Staff": 20.1,
    "Doctor": 15.0,
    "Cleanliness": 13.5,
    "Treatment": 4.9
}


root_causes = {
    "Long waiting": 10,
    "Delays": 7,
    "Emergency service problems": 3,
    "Poor treatment": 3,
    "Rude staff": 2,
    "Dirty toilets": 2,
    "Poor facilities": 2,
    "Unnecessary tests": 2,
    "Poor communication": 2,
    "Poor hygiene": 1,
    "Pests": 1
}


# ============================================================
# DASHBOARD TITLE
# ============================================================

print("\n")
print("=" * 70)
print("             PATIENT EXPERIENCE DASHBOARD")
print("=" * 70)

print("\nDataset:")
print("951 unique patient reviews analysed")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print("\n")
print("-" * 70)
print("                    MODEL PERFORMANCE")
print("-" * 70)

print(
    f"{'Model':<30}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<10}"
)

print("-" * 70)

for i in range(len(model_names)):

    print(
        f"{model_names[i]:<30}"
        f"{accuracy[i]}%{'':<7}"
        f"{precision[i]}%{'':<7}"
        f"{recall[i]}%{'':<7}"
        f"{f1_score[i]}%"
    )


# ============================================================
# ASPECT ANALYSIS
# ============================================================

print("\n")
print("-" * 70)
print("                 ASPECT NEGATIVE RATES")
print("-" * 70)

sorted_aspects = sorted(
    aspect_data.items(),
    key=lambda x: x[1],
    reverse=True
)

for aspect, rate in sorted_aspects:

    print(
        f"{aspect:<25} {rate:.1f}% negative"
    )


# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

print("\n")
print("-" * 70)
print("                    ROOT CAUSES")
print("-" * 70)

sorted_causes = sorted(
    root_causes.items(),
    key=lambda x: x[1],
    reverse=True
)

for cause, count in sorted_causes:

    print(
        f"{cause:<30} {count}"
    )


# ============================================================
# CREATE ASPECT CHART
# ============================================================

aspects = [
    item[0]
    for item in sorted_aspects
]

rates = [
    item[1]
    for item in sorted_aspects
]

plt.figure(figsize=(10, 6))

bars = plt.barh(
    aspects[::-1],
    rates[::-1]
)

plt.xlabel("Negative Rate (%)")
plt.ylabel("Patient Experience Aspect")
plt.title("Negative Rate by Patient Experience Aspect")

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center"
    )

plt.tight_layout()

plt.savefig(
    "dashboard_aspect_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CREATE ROOT CAUSE CHART
# ============================================================

causes = [
    item[0]
    for item in sorted_causes
]

counts = [
    item[1]
    for item in sorted_causes
]

plt.figure(figsize=(11, 7))

bars = plt.barh(
    causes[::-1],
    counts[::-1]
)

plt.xlabel("Number of Occurrences")
plt.ylabel("Root Cause")
plt.title("Patient Feedback Root Causes")

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.1,
        bar.get_y() + bar.get_height() / 2,
        str(int(width)),
        va="center"
    )

plt.tight_layout()

plt.savefig(
    "dashboard_root_cause_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL INSIGHT
# ============================================================

print("\n")
print("=" * 70)
print("                    KEY FINDINGS")
print("=" * 70)

print(
    "\n1. Waiting Time has the highest negative rate "
    "(97.7%)."
)

print(
    "2. Emergency has the second-highest negative rate "
    "(78.9%)."
)

print(
    "3. Long waiting is the most frequently detected "
    "root cause (10 occurrences)."
)

print(
    "4. Delays are the second most frequently detected "
    "root cause (7 occurrences)."
)

print(
    "5. DistilBERT achieved the highest reported F1-score "
    "(0.94) among the three models."
)

print("\n")
print("=" * 70)
print("              DASHBOARD GENERATION COMPLETE")
print("=" * 70)