import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download("vader_lexicon")

# Load the dataset
df = pd.read_csv("preprocessed_hospital.csv")

# Create the VADER analyser
sia = SentimentIntensityAnalyzer()


# Function to predict sentiment using VADER
def vader_sentiment(text):
    score = sia.polarity_scores(text)["compound"]

    # Convert the VADER score into a sentiment label
    if score >= 0:
        return 1  # Positive
    else:
        return 0  # Negative


# Apply VADER to each patient review
df["VADER_Prediction"] = df["Feedback"].apply(vader_sentiment)

# Display the first 10 results
print("VADER SENTIMENT PREDICTIONS")
print("---------------------------")

print(
    df[
        [
            "Feedback",
            "Sentiment Label",
            "VADER_Prediction"
        ]
    ].head(10)
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Actual sentiment labels
y_true = df["Sentiment Label"]

# VADER predictions
y_pred = df["VADER_Prediction"]

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nVADER PERFORMANCE")
print("-----------------")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nCONFUSION MATRIX")
print("----------------")
print(cm)