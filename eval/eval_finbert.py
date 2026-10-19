import os

import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

LABELS = ["positive", "negative", "neutral"]
INPUT_PATH = "eval/headlines.csv"
OUTPUT_PATH = "eval/finbert_predictions.csv"

# Optional S3 mode: set EVAL_BUCKET to read input from and write output to S3.
# boto3 is only needed in this mode, so local runs don't require it.
BUCKET = os.environ.get("EVAL_BUCKET")

if BUCKET:
    import boto3

    s3 = boto3.client("s3", region_name="eu-north-1")
    s3.download_file(BUCKET, INPUT_PATH, INPUT_PATH)
    print(f"Downloaded s3://{BUCKET}/{INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

# Check that every human label is valid before running the model
if df["human_label"].isna().any() or not df["human_label"].isin(LABELS).all():
    raise ValueError(
        f"Every human_label must be one of {LABELS}. "
        f"Check for empty or invalid labels in {INPUT_PATH}."
    )

# Load FinBERT sentiment model
clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Predict sentiment for all headlines in one call
predictions = clf(df["headline"].tolist())

# Extract predicted labels and standardise case
df["finbert_label"] = [p["label"].lower() for p in predictions]

y_true = df["human_label"]
y_pred = df["finbert_label"]

# Evaluation metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro-F1:", f1_score(y_true, y_pred, labels=LABELS, average="macro"))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
print("\nConfusion Matrix (rows = human, columns = FinBERT):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

# Save predictions locally, then upload to S3 if in S3 mode
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved predictions to {OUTPUT_PATH}")

if BUCKET:
    s3.upload_file(OUTPUT_PATH, BUCKET, OUTPUT_PATH)
    print(f"Uploaded to s3://{BUCKET}/{OUTPUT_PATH}")