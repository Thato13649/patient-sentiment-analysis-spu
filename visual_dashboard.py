import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# PROJECT DATA
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
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Patient Experience Analysis Dashboard")

root.geometry("1200x800")

root.configure(bg="white")


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    root,
    text="PATIENT EXPERIENCE ANALYSIS DASHBOARD",
    font=("Arial", 24, "bold"),
    bg="white"
)

title.pack(pady=(20, 5))


subtitle = tk.Label(
    root,
    text="NLP-based analysis of 951 patient reviews",
    font=("Arial", 12),
    bg="white"
)

subtitle.pack(pady=(0, 15))


# ============================================================
# SUMMARY CARDS
# ============================================================

card_frame = tk.Frame(root, bg="white")

card_frame.pack(fill="x", padx=30)


def create_card(parent, title_text, value_text):

    frame = tk.Frame(
        parent,
        bg="white",
        highlightbackground="gray",
        highlightthickness=1
    )

    frame.pack(
        side="left",
        expand=True,
        fill="both",
        padx=8
    )

    tk.Label(
        frame,
        text=title_text,
        font=("Arial", 11, "bold"),
        bg="white"
    ).pack(pady=(12, 3))

    tk.Label(
        frame,
        text=value_text,
        font=("Arial", 22, "bold"),
        bg="white"
    ).pack(pady=(0, 12))


create_card(
    card_frame,
    "Reviews Analysed",
    "951"
)

create_card(
    card_frame,
    "Best Accuracy",
    "91%"
)

create_card(
    card_frame,
    "Best F1 Score",
    "94%"
)

create_card(
    card_frame,
    "Highest Negative Aspect",
    "Waiting Time"
)


# ============================================================
# NOTEBOOK / TABS
# ============================================================

notebook = ttk.Notebook(root)

notebook.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=25
)


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

model_tab = tk.Frame(notebook, bg="white")

notebook.add(
    model_tab,
    text="Model Performance"
)


fig1, ax1 = plt.subplots(
    figsize=(9, 4.5)
)

x = range(len(model_names))

width = 0.2

ax1.bar(
    [i - 1.5 * width for i in x],
    accuracy,
    width,
    label="Accuracy"
)

ax1.bar(
    [i - 0.5 * width for i in x],
    precision,
    width,
    label="Precision"
)

ax1.bar(
    [i + 0.5 * width for i in x],
    recall,
    width,
    label="Recall"
)

ax1.bar(
    [i + 1.5 * width for i in x],
    f1_score,
    width,
    label="F1"
)

ax1.set_xticks(x)

ax1.set_xticklabels(
    model_names,
    rotation=15
)

ax1.set_ylabel("Percentage (%)")

ax1.set_title(
    "Model Performance Comparison"
)

ax1.set_ylim(0, 110)

ax1.legend()

fig1.tight_layout()


canvas1 = FigureCanvasTkAgg(
    fig1,
    master=model_tab
)

canvas1.draw()

canvas1.get_tk_widget().pack(
    fill="both",
    expand=True
)


# ============================================================
# ASPECT ANALYSIS TAB
# ============================================================

aspect_tab = tk.Frame(
    notebook,
    bg="white"
)

notebook.add(
    aspect_tab,
    text="Aspect Analysis"
)


sorted_aspects = sorted(
    aspect_data.items(),
    key=lambda x: x[1],
    reverse=True
)


fig2, ax2 = plt.subplots(
    figsize=(9, 5)
)

aspects = [
    item[0]
    for item in sorted_aspects
]

rates = [
    item[1]
    for item in sorted_aspects
]


bars = ax2.barh(
    aspects[::-1],
    rates[::-1]
)

ax2.set_xlabel(
    "Negative Rate (%)"
)

ax2.set_title(
    "Negative Rate by Patient Experience Aspect"
)

ax2.set_xlim(
    0,
    105
)


for bar in bars:

    value = bar.get_width()

    ax2.text(
        value + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center"
    )


fig2.tight_layout()


canvas2 = FigureCanvasTkAgg(
    fig2,
    master=aspect_tab
)

canvas2.draw()

canvas2.get_tk_widget().pack(
    fill="both",
    expand=True
)


# ============================================================
# ROOT CAUSE TAB
# ============================================================

root_tab = tk.Frame(
    notebook,
    bg="white"
)

notebook.add(
    root_tab,
    text="Root Causes"
)


sorted_causes = sorted(
    root_causes.items(),
    key=lambda x: x[1],
    reverse=True
)


fig3, ax3 = plt.subplots(
    figsize=(9, 5)
)

causes = [
    item[0]
    for item in sorted_causes
]

counts = [
    item[1]
    for item in sorted_causes
]


bars = ax3.barh(
    causes[::-1],
    counts[::-1]
)

ax3.set_xlabel(
    "Number of Occurrences"
)

ax3.set_title(
    "Detected Patient Feedback Root Causes"
)


for bar in bars:

    value = bar.get_width()

    ax3.text(
        value + 0.1,
        bar.get_y() + bar.get_height() / 2,
        str(int(value)),
        va="center"
    )


fig3.tight_layout()


canvas3 = FigureCanvasTkAgg(
    fig3,
    master=root_tab
)

canvas3.draw()

canvas3.get_tk_widget().pack(
    fill="both",
    expand=True
)


# ============================================================
# KEY FINDINGS TAB
# ============================================================

findings_tab = tk.Frame(
    notebook,
    bg="white"
)

notebook.add(
    findings_tab,
    text="Key Findings"
)


heading = tk.Label(
    findings_tab,
    text="Key Findings from Patient Feedback",
    font=("Arial", 20, "bold"),
    bg="white"
)

heading.pack(
    pady=25
)


findings = [
    "951 unique patient reviews were analysed.",
    "DistilBERT achieved 91% accuracy and a 94% F1-score.",
    "Waiting Time had the highest negative rate at 97.7%.",
    "Emergency had a negative rate of 78.9%.",
    "Long waiting was detected 10 times as a root cause.",
    "Delays were detected 7 times.",
    "Other identified issues included rude staff, dirty toilets, poor facilities, unnecessary tests and poor communication."
]


for number, finding in enumerate(findings, start=1):

    label = tk.Label(
        findings_tab,
        text=f"{number}. {finding}",
        font=("Arial", 14),
        bg="white",
        anchor="w",
        justify="left",
        wraplength=1000
    )

    label.pack(
        fill="x",
        padx=60,
        pady=10
    )


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    root,
    text="Decision-support system — human review remains important",
    font=("Arial", 10),
    bg="white"
)

footer.pack(
    pady=(0, 10)
)


# ============================================================
# START DASHBOARD
# ============================================================

root.mainloop()