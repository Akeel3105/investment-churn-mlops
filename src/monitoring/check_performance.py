
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]

METRICS_PATH = (
    ROOT_DIR / "data" / "monitoring" / "monitoring_metrics.json"
)

# Learning ke liye threshold
F1_THRESHOLD = 0.75

if not METRICS_PATH.exists():
    raise FileNotFoundError(
        f"Metrics file not found: {METRICS_PATH}"
    )

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

f1_score = metrics["f1_score"]

print("MODEL PERFORMANCE CHECK")
print("=" * 35)
print(f"Current F1-score : {f1_score:.4f}")
print(f"Required F1-score: {F1_THRESHOLD:.4f}")

if f1_score < F1_THRESHOLD:
    print("\nALERT: Performance threshold breached!")
    print("Action: Retraining recommended.")
else:
    print("\nModel performance is within threshold.")
    print("Action: No retraining required.")