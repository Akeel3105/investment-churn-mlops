
from pathlib import Path
import pandas as pd
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "monitoring" / "new_customers.csv"

REQUIRED_COLUMNS = {
    "age", "tenure_months", "monthly_income",
    "total_investment", "monthly_sip", "num_investments",
    "portfolio_return_pct", "last_investment_days",
    "last_login_days", "advisor_interactions",
    "withdrawal_count_6m", "support_tickets_6m",
    "risk_profile", "churn"
}


def main():
    df = pd.read_csv(DATA_FILE)
    errors = []

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {sorted(missing_columns)}")

    if df.empty:
        errors.append("Dataset is empty")

    #if "customer_id" in df.columns:
    #    if df["customer_id"].isna().any():
    #        errors.append("customer_id contains missing values")
    #    if df["customer_id"].duplicated().any():
    #        errors.append("customer_id contains duplicates")

    if df.isna().any().any():
        errors.append("Dataset contains missing values")

    if "churn" in df.columns:
        if not df["churn"].isin([0, 1]).all():
            errors.append("churn must contain only 0 or 1")

    if "age" in df.columns:
        if (df["age"] <= 0).any():
            errors.append("age must be greater than 0")

    if "risk_profile" in df.columns:
        allowed = {"Conservative", "Moderate", "Aggressive"}
        invalid = set(df["risk_profile"].dropna().unique()) - allowed
        if invalid:
            errors.append(f"Invalid risk_profile values: {invalid}")

    if errors:
        print("\nDATA VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print(f"DATA VALIDATION PASSED: {len(df)} rows checked")


if __name__ == "__main__":
    main()
