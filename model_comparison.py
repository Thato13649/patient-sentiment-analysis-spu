import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    "VADER",
    "TF-IDF + Logistic Regression",
    "DistilBERT"
]

# Performance scores
accuracy = [0.86, 0.81, 0.91]
precision = [0.89, 0.80, 0.95]
recall = [0.93, 0.99, 0.93]
f1 = [0.91, 0.88, 0.94]

# Display results in the terminal
print("MODEL PERFORMANCE COMPARISON")
print("============================")

for i in range(len(models)):
    print(f"\n{models[i]}")
    print(f"Accuracy:  {accuracy[i]:.2f}")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall:    {recall[i]:.2f}")
    print(f"F1-score:  {f1[i]:.2f}")

# Create a comparison chart
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5 * width, precision, width, label="Precision")
plt.bar(x + 0.5 * width, recall, width, label="Recall")
plt.bar(x + 1.5 * width, f1, width, label="F1-score")

plt.xlabel("Model")
plt.ylabel("Score (%))")
plt.title("Performance Comparison of Sentiment Analysis Models")

plt.xticks(x, models)
plt.ylim(0, 1.0)
plt.yticks(
    np.arange(0, 1.1, 0.1),
    [f"{int(x * 100)}%" for x in np.arange(0, 1.1, 0.1)]
)

plt.legend()

plt.tight_layout()
plt.show()