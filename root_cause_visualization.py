import matplotlib.pyplot as plt

# ============================================================
# ROOT CAUSE DATA
# ============================================================

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
# SORT ROOT CAUSES
# ============================================================

sorted_causes = sorted(
    root_causes.items(),
    key=lambda x: x[1],
    reverse=True
)

causes = [item[0] for item in sorted_causes]
counts = [item[1] for item in sorted_causes]


# ============================================================
# CREATE BAR CHART
# ============================================================

plt.figure(figsize=(12, 7))

bars = plt.barh(
    causes[::-1],
    counts[::-1]
)

plt.xlabel("Number of Occurrences")
plt.ylabel("Root Cause")

plt.title(
    "Patient Feedback: Root Cause Analysis"
)

plt.tight_layout()


# ============================================================
# ADD NUMBERS TO BARS
# ============================================================

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.1,
        bar.get_y() + bar.get_height() / 2,
        str(int(width)),
        va="center"
    )


# ============================================================
# SAVE GRAPH
# ============================================================

plt.savefig(
    "root_cause_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nRoot cause chart created successfully!")
print("Saved as: root_cause_chart.png")