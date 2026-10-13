import matplotlib.pyplot as plt
import numpy as np

# Confusion matrices for the three models

vader_cm = np.array([
    [177, 80],
    [51, 643]
])

tfidf_cm = np.array([
    [18, 34],
    [2, 137]
])

distilbert_cm = np.array([
    [45, 7],
    [10, 129]
])

# Function to display a confusion matrix
def display_confusion_matrix(cm, title):

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(title)
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")

    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    # Display the values inside the matrix
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()


# Display each confusion matrix
display_confusion_matrix(
    vader_cm,
    "VADER Confusion Matrix"
)

display_confusion_matrix(
    tfidf_cm,
    "TF-IDF + Logistic Regression Confusion Matrix"
)

display_confusion_matrix(
    distilbert_cm,
    "DistilBERT Confusion Matrix"
)