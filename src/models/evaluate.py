from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
OUTPUT_PATH = ROOT_DIR / "models" / "final_test_metrics.json"


# Load test data
test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(columns=["churn"])
y_test = test_df["churn"]


# Load best model
model = joblib.load(MODEL_PATH)


# Predictions
y_pred = model.predict(X_test)

# Probability for ROC-AUC
y_prob = model.predict_proba(X_test)[:, 1]


# Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob)
}


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)


# Print results
print("\n========== FINAL TEST RESULTS ==========")

for name, value in metrics.items():
    print(f"{name.upper():10}: {value:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# Save metrics
with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to: {OUTPUT_PATH}")