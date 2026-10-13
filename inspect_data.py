import pandas as pd

# Load the dataset
df = pd.read_csv("hospital.csv")

# Display the first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns)

# Display dataset size
print("\nDATASET SHAPE:")
print(df.shape)

# Display sentiment labels
print("\nSENTIMENT LABELS:")
print(df["Sentiment Label"].value_counts())

# Display ratings
print("\nRATINGS:")
print(df["Ratings"].value_counts())

# Check for missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())