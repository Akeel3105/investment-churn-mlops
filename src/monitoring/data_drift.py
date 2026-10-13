
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

ROOT_DIR = Path(__file__).resolve().parents[2]

REFERENCE_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
NEW_DATA_PATH = ROOT_DIR / "data" / "monitoring" / "new_customers.csv"

reference_df = pd.read_csv(REFERENCE_PATH)
new_df = pd.read_csv(NEW_DATA_PATH)

# Sirf numerical features check karenge
numeric_cols = reference_df.select_dtypes(include="number").columns
numeric_cols = [col for col in numeric_cols if col != "churn"]

print("\nDATA DRIFT REPORT")
print("=" * 60)

for col in numeric_cols:
    result = ks_2samp(reference_df[col], new_df[col])

    drift_detected = result.pvalue < 0.05
    status = "DRIFT DETECTED" if drift_detected else "No significant drift"

    print(
        f"{col:28} | "
        f"p-value: {result.pvalue:.4f} | {status}"
    )