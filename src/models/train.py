import json
from pathlib import Path

import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("investment-customer-churn")


# ============================================================
# Load Data
# ============================================================

print("=" * 70)
print("INVESTMENT CUSTOMER CHURN - MODEL TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

X_train = train_df.drop(columns=["churn"])
y_train = train_df["churn"]

X_validation = validation_df.drop(columns=["churn"])
y_validation = validation_df["churn"]

print(f"\nTraining data   : {X_train.shape}")
print(f"Validation data : {X_validation.shape}")


# ============================================================
# Feature Types
# ============================================================

categorical_features = [
    "risk_profile"
]

numerical_features = [
    "age",
    "tenure_months",
    "monthly_income",
    "total_investment",
    "monthly_sip",
    "num_investments",
    "portfolio_return_pct",
    "last_investment_days",
    "last_login_days",
    "advisor_interactions",
    "withdrawal_count_6m",
    "support_tickets_6m"
]


# ============================================================
# Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# Models
# ============================================================

models = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# Training
# ============================================================

results = []

best_model = None
best_model_name = None
best_f1 = -1
best_run_id = None


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    with mlflow.start_run(run_name=model_name) as run:

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(X_train, y_train)

        # ----------------------------------------------------
        # Validation Prediction
        # ----------------------------------------------------

        y_pred = pipeline.predict(X_validation)

        y_probability = pipeline.predict_proba(
            X_validation
        )[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_validation,
            y_pred
        )

        precision = precision_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_validation,
            y_probability
        )

        # ----------------------------------------------------
        # MLflow Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            model_name
        )

        mlflow.log_param(
            "training_rows",
            len(X_train)
        )

        mlflow.log_param(
            "validation_rows",
            len(X_validation)
        )

        mlflow.log_param(
            "num_features",
            len(numerical_features)
        )

        mlflow.log_param(
            "categorical_features",
            len(categorical_features)
        )

        # ----------------------------------------------------
        # MLflow Metrics
        # ----------------------------------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # ----------------------------------------------------
        # Log Model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="model"
        )

        # ----------------------------------------------------
        # Save Result
        # ----------------------------------------------------

        result = {
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "run_id": run.info.run_id
        }

        results.append(result)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

        # ----------------------------------------------------
        # Select Best Model
        # ----------------------------------------------------

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = model_name
            best_run_id = run.info.run_id


# ============================================================
# Save Best Model
# ============================================================

best_model_path = MODEL_DIR / "best_model.pkl"

import joblib

joblib.dump(
    best_model,
    best_model_path
)


# ============================================================
# Save Results
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)

results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)


best_model_info = {
    "model": best_model_name,
    "f1_score": round(best_f1, 4),
    "run_id": best_run_id,
    "model_path": str(best_model_path)
}

with open(
    MODEL_DIR / "best_model_info.json",
    "w"
) as f:
    json.dump(
        best_model_info,
        f,
        indent=4
    )


# ============================================================
# Final Output
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model     : {best_model_name}")
print(f"F1 Score  : {best_f1:.4f}")
print(f"Run ID    : {best_run_id}")

print("\nSaved:")
print(f"  - {best_model_path}")
print("  - models/model_comparison.csv")
print("  - models/best_model_info.json")

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)