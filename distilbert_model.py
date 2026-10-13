from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# Load the DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

print("DISTILBERT SETUP")
print("----------------")
print("Tokenizer loaded successfully.")
print("Model loaded successfully.")

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("preprocessed_hospital.csv")

# Use the original patient reviews
X = df["Feedback"].tolist()

# Use the sentiment labels
y = df["Sentiment Label"].tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDISTILBERT DATA")
print("---------------")
print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# Tokenise the patient reviews
train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=128
)

print("\nTOKENISATION")
print("------------")
print("Training data tokenised.")
print("Testing data tokenised.")
print("Number of training examples:", len(train_encodings["input_ids"]))
print("Number of testing examples:", len(test_encodings["input_ids"]))

import torch
from torch.utils.data import Dataset


class PatientReviewDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

    def __len__(self):
        return len(self.labels)


# Create PyTorch datasets
train_dataset = PatientReviewDataset(
    train_encodings,
    y_train
)

test_dataset = PatientReviewDataset(
    test_encodings,
    y_test
)

print("\nDATASET CREATION")
print("----------------")
print("Training dataset:", len(train_dataset))
print("Testing dataset:", len(test_dataset))

from transformers import TrainingArguments, Trainer

# Set up the training configuration
training_args = TrainingArguments(
    output_dir="./distilbert_results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=20,
    report_to="none"
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("\nDISTILBERT TRAINING SETUP")
print("-------------------------")
print("Training configuration created.")
print("Ready to train the model.")
print("\nSTARTING DISTILBERT TRAINING")
print("----------------------------")

trainer.train()

print("\nDISTILBERT TRAINING COMPLETE")
print("----------------------------")



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Make predictions on the test dataset
predictions = trainer.predict(test_dataset)

# Get the predicted sentiment labels
y_pred = predictions.predictions.argmax(axis=1)

# Actual sentiment labels
y_true = y_test

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nDISTILBERT PERFORMANCE")
print("----------------------")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nCONFUSION MATRIX")
print("----------------")
print(cm)

# Save the trained DistilBERT model
model.save_pretrained("./distilbert_patient_model")
tokenizer.save_pretrained("./distilbert_patient_model")

print("\nMODEL SAVED")
print("-----------")
print("DistilBERT model saved successfully.")