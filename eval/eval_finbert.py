
import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

LABELS = ["positive", "negative", "neutral"]

df = pd.read_csv("eval/headlines.csv")

# TODO 1: Check that every human label is valid
if df["human_label"].isna().any() or not df["human_label"].isin(LABELS).all():
    raise ValueError(
        f"Every human_label must be one of {LABELS}. "
        "Check for empty or invalid labels in eval/headlines.csv."
    )

# Load FinBERT sentiment model
clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# TODO 2: Predict sentiment for all headlines in one call
predictions = clf(df["headline"].tolist())

# TODO 3: Extract predicted labels and standardise case
df["finbert_label"] = [p["label"].lower() for p in predictions]

# Actual labels and predicted labels
y_true = df["human_label"]
y_pred = df["finbert_label"]

# TODO 4: Print evaluation metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro-F1:", f1_score(y_true, y_pred, labels=LABELS, average="macro"))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

# TODO 5: Print confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
print("\nConfusion Matrix (rows = human, columns = FinBERT):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

# TODO 6: Save predictions
df.to_csv("eval/finbert_predictions.csv", index=False)
print("\nSaved predictions to eval/finbert_predictions.csv")
