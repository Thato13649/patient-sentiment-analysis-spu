import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_hospital.csv")

# Text used for the model
X = df["Processed_Feedback"]

# Target sentiment labels
y = df["Sentiment Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create the TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Learn the vocabulary from the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform the testing data using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

# Display the results
print("TF-IDF DATA PREPARATION")
print("-----------------------")

print("Training reviews:", X_train.shape[0])
print("Testing reviews:", X_test.shape[0])

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Display some predictions
print("\nLOGISTIC REGRESSION")
print("-------------------")

print("Model training complete.")
print("Number of predictions:", len(y_pred))

print("\nFirst 10 predictions:")
print(y_pred[:10])

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTF-IDF + LOGISTIC REGRESSION PERFORMANCE")
print("-----------------------------------------")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print("----------------")
print(cm)

