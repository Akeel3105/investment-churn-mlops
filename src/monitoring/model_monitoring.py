
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
DATA_PATH = ROOT_DIR / "data" / "monitoring" / "new_customers.csv"

# Model aur new customer data load karein
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Actual outcome aur input features alag karein
y_actual = df["churn"]
X_new = df.drop(columns=["churn"])

# Model se predictions lein
y_pred = model.predict(X_new)

# Performance metrics calculate karein
print("\nMODEL MONITORING REPORT")
print("=" * 50)
print("Customers monitored:", len(df))
print("Accuracy :", round(accuracy_score(y_actual, y_pred), 4))
print("Precision:", round(precision_score(y_actual, y_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_actual, y_pred, zero_division=0), 4))
print("F1-score :", round(f1_score(y_actual, y_pred, zero_division=0), 4))

print("\nConfusion Matrix (actual rows, predicted columns):")
print(confusion_matrix(y_actual, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(y_actual, y_pred, zero_division=0))

# Actual aur predicted results save karein
df["predicted_churn"] = y_pred
OUTPUT_PATH = ROOT_DIR / "data" / "monitoring" / "monitoring_results.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:", OUTPUT_PATH)



METRICS_PATH = (
    ROOT_DIR / "data" / "monitoring" / "monitoring_metrics.json"
)

metrics = {
    "accuracy": float(accuracy_score(y_actual, y_pred)),
    "precision": float(precision_score(y_actual, y_pred, zero_division=0)),
    "recall": float(recall_score(y_actual, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_actual, y_pred, zero_division=0)),
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved to:", METRICS_PATH)