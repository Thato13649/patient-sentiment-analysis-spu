import pandas as pd
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_hospital.csv")

# Count sentiment labels
sentiment_counts = df["Sentiment Label"].value_counts().sort_index()

# Create the bar chart
plt.figure(figsize=(6, 4))

plt.bar(
    ["Negative (0)", "Positive (1)"],
    sentiment_counts.values
)

plt.title("Distribution of Patient Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

# Add the number above each bar
for i, value in enumerate(sentiment_counts.values):
    plt.text(i, value + 10, str(value), ha="center")

plt.tight_layout()
plt.show()

# Ratings distribution
rating_counts = df["Ratings"].value_counts().sort_index()

plt.figure(figsize=(6, 4))

plt.bar(
    rating_counts.index.astype(str),
    rating_counts.values
)

plt.title("Distribution of Patient Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

# Add numbers above each bar
for i, value in enumerate(rating_counts.values):
    plt.text(i, value + 5, str(value), ha="center")

plt.tight_layout()
plt.show()# Compare ratings with sentiment labels
sentiment_rating = pd.crosstab(
    df["Ratings"],
    df["Sentiment Label"]
)

print("\nRATING VS SENTIMENT:")
print(sentiment_rating)

# Most common words in positive and negative reviews

from collections import Counter

# Separate positive and negative reviews
positive_reviews = df[df["Sentiment Label"] == 1]["Processed_Feedback"]
negative_reviews = df[df["Sentiment Label"] == 0]["Processed_Feedback"]

# Combine all words
positive_words = " ".join(positive_reviews).split()
negative_words = " ".join(negative_reviews).split()

# Count the words
positive_word_counts = Counter(positive_words)
negative_word_counts = Counter(negative_words)

# Display the 15 most common words
print("\nMOST COMMON WORDS IN POSITIVE REVIEWS:")
print(positive_word_counts.most_common(15))

print("\nMOST COMMON WORDS IN NEGATIVE REVIEWS:")
print(negative_word_counts.most_common(15))

