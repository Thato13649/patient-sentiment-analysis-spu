import pandas as pd

# Load the dataset
df = pd.read_csv("hospital.csv")

# Remove the completely empty column
df = df.drop(columns=["Unnamed: 3"])

# Remove rows with missing feedback or sentiment labels
df = df.dropna(subset=["Feedback", "Sentiment Label"])

# Remove duplicate reviews
df = df.drop_duplicates(subset=["Feedback"])

# Make sure sentiment labels are integers
df["Sentiment Label"] = df["Sentiment Label"].astype(int)

# Save the cleaned dataset
df.to_csv("cleaned_hospital.csv", index=False)

# Display information about the cleaned dataset
print("CLEANED DATASET")
print("----------------")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df["Sentiment Label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())