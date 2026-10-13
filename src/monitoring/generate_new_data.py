
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "monitoring" / "new_customers.csv"

# Training data load karein
train_df = pd.read_csv(TRAIN_PATH)

# Reproducible sample banayein
new_df = train_df.sample(
    n=1000,
    replace=True,
    random_state=42
).copy()

# Artificial drift simulate karein:
# naye customers ki income ko 40% badha dein
new_df["monthly_income"] = new_df["monthly_income"] * 1.4

# Naye customer data ko save karein
new_df.to_csv(OUTPUT_PATH, index=False)

print("New customer data generated successfully!")
print("Shape:", new_df.shape)
print("Saved to:", OUTPUT_PATH)
print("\nMonthly income comparison:")
print("Training mean:", round(train_df["monthly_income"].mean(), 2))
print("New data mean:", round(new_df["monthly_income"].mean(), 2))