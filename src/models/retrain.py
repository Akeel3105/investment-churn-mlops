
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT_DIR / "data" / "processed" / "validation.csv"
NEW_DATA_PATH = ROOT_DIR / "data" / "monitoring" / "new_customers.csv"

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
CANDIDATE_PATH = ROOT_DIR / "models" / "candidate_model.pkl"
REPORT_PATH = ROOT_DIR / "models" / "retraining_report.json"

# Business acceptance criteria for this learning exercise
MIN_F1_IMPROVEMENT = 0.0
ALLOW_RECALL_REGRESSION = False

mlflow.set_tracking_uri(f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}")
mlflow.set_experiment("investment-customer-churn")


def evaluate(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def main():
    print("=" * 60)
    print("AUTOMATED RETRAINING")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    new_df = pd.read_csv(NEW_DATA_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    # New data ka schema training data se match hona chahiye
    if set(train_df.columns) != set(new_df.columns):
        raise ValueError(
            "Training aur new data ke columns match nahi karte."
        )

    new_df = new_df[train_df.columns].copy()

    # Hamara demo data training data se sample hua hai.
    # IDs ko unique banayein; ye real new customers ka proof nahi hai.
    if "customer_id" in new_df.columns:
        new_df["customer_id"] = [
            f"DEMO_NEW_{i:06d}" for i in range(len(new_df))
        ]

    combined_df = pd.concat(
        [train_df, new_df],
        ignore_index=True
    )

    X_train = combined_df.drop(columns=["churn"])
    y_train = combined_df["churn"]

    X_val = validation_df.drop(columns=["churn"])
    y_val = validation_df["churn"]

    # Current production-style model ko preserve karein
    current_model = joblib.load(MODEL_PATH)

    baseline_metrics = evaluate(current_model, X_val, y_val)

    # Same model architecture, fresh training
    candidate_model = clone(current_model)
    candidate_model.fit(X_train, y_train)

    candidate_metrics = evaluate(candidate_model, X_val, y_val)

    print(f"\nOld training rows: {len(train_df)}")
    print(f"New labelled rows: {len(new_df)}")
    print(f"Combined rows:     {len(combined_df)}")

    print("\nValidation comparison")
    print("-" * 60)

    for metric in baseline_metrics:
        print(
            f"{metric:12} | "
            f"Old: {baseline_metrics[metric]:.4f} | "
            f"Candidate: {candidate_metrics[metric]:.4f}"
        )

    # Candidate ko MLflow mein record karein
    with mlflow.start_run(run_name="automated_retraining_candidate"):
        mlflow.log_param("training_rows", len(combined_df))
        mlflow.log_param("new_data_rows", len(new_df))
        mlflow.log_param("selection_metric", "validation_f1_score")

        for metric, value in candidate_metrics.items():
            mlflow.log_metric(f"candidate_{metric}", value)

        mlflow.sklearn.log_model(
            candidate_model,
            name="candidate_model"
        )

    f1_improved = (
        candidate_metrics["f1_score"]
        > baseline_metrics["f1_score"] + MIN_F1_IMPROVEMENT
    )

    recall_ok = (
        ALLOW_RECALL_REGRESSION
        or candidate_metrics["recall"] >= baseline_metrics["recall"]
    )

    accepted = f1_improved and recall_ok

    # Candidate artifact ko alag save karein
    joblib.dump(candidate_model, CANDIDATE_PATH)

    report = {
        "baseline_validation_metrics": baseline_metrics,
        "candidate_validation_metrics": candidate_metrics,
        "f1_improved": f1_improved,
        "recall_acceptable": recall_ok,
        "candidate_accepted": accepted,
        "model_promoted": False,
    }

    if accepted:
        # Purana model sirf acceptance checks pass hone par replace hoga
        joblib.dump(candidate_model, MODEL_PATH)
        report["model_promoted"] = True

        print("\nACCEPTED: Candidate passed the criteria.")
        print("Updated models/best_model.pkl")
    else:
        print("\nREJECTED: Candidate did not pass the criteria.")
        print("Existing best_model.pkl remains unchanged.")

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nCandidate saved: {CANDIDATE_PATH}")
    print(f"Report saved:    {REPORT_PATH}")


if __name__ == "__main__":
    main()
