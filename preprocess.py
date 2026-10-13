import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the cleaned dataset
df = pd.read_csv("cleaned_hospital.csv")

# Set up NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenise the text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatise words
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Join the words back together
    return " ".join(tokens)


# Apply preprocessing
df["Processed_Feedback"] = df["Feedback"].apply(preprocess_text)

# Save the preprocessed dataset
df.to_csv("preprocessed_hospital.csv", index=False)

# Display results
print("PREPROCESSING COMPLETE")
print("----------------------")

print("\nOriginal review:")
print(df["Feedback"].iloc[0])

print("\nProcessed review:")
print(df["Processed_Feedback"].iloc[0])

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)