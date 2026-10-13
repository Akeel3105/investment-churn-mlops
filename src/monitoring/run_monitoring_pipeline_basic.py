
from pathlib import Path
import json
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]

DRIFT_SCRIPT = ROOT_DIR / "src" / "monitoring" / "data_drift.py"
MONITOR_SCRIPT = ROOT_DIR / "src" / "monitoring" / "model_monitoring.py"
RETRAIN_SCRIPT = ROOT_DIR / "src" / "models" / "retrain.py"

METRICS_FILE = ROOT_DIR / "data" / "monitoring" / "monitoring_metrics.json"

F1_THRESHOLD = 0.75


def run_script(script_path):
    print(f"\nRunning: {script_path.name}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT_DIR,
        check=True
    )


def main():
    print("\n=== MLOPS MONITORING PIPELINE ===")

    # Data Validation
    VALIDATE_SCRIPT = ROOT_DIR / "src" / "monitoring" / "validate_data.py"
    run_script(VALIDATE_SCRIPT)

    # 1. Check data drift
    run_script(DRIFT_SCRIPT)

    # 2. Monitor current model
    run_script(MONITOR_SCRIPT)

    # 3. Read monitoring metrics
    with open(METRICS_FILE, "r") as file:
        metrics = json.load(file)

    current_f1 = metrics["f1_score"]

    print(f"\nCurrent F1-score: {current_f1:.4f}")
    print(f"Required F1-score: {F1_THRESHOLD:.4f}")

    # 4. Trigger retraining if performance is below threshold
    if current_f1 < F1_THRESHOLD:
        print("\nALERT: Performance threshold breached!")
        print("Starting retraining...")
        run_script(RETRAIN_SCRIPT)
    else:
        print("\nPerformance is acceptable.")
        print("Retraining not required.")

    print("\n=== PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()
