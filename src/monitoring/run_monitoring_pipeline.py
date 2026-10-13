
from pathlib import Path
import json
import logging
import subprocess
import sys
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[2]

MONITORING_DIR = ROOT_DIR / "data" / "monitoring"
LOG_FILE = MONITORING_DIR / "pipeline.log"
METRICS_FILE = MONITORING_DIR / "monitoring_metrics.json"

VALIDATE_SCRIPT = ROOT_DIR / "src" / "monitoring" / "validate_data.py"
DRIFT_SCRIPT = ROOT_DIR / "src" / "monitoring" / "data_drift.py"
MONITOR_SCRIPT = ROOT_DIR / "src" / "monitoring" / "model_monitoring.py"
RETRAIN_SCRIPT = ROOT_DIR / "src" / "models" / "retrain.py"

F1_THRESHOLD = 0.75


def setup_logging():
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def run_script(script_path):
    logging.info("Starting step: %s", script_path.name)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT_DIR,
        check=True,
    )

    logging.info("Completed step: %s", script_path.name)


def main():
    setup_logging()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.info("=" * 55)
    logging.info("PIPELINE STARTED | Run ID: %s", run_id)

    try:
        run_script(VALIDATE_SCRIPT)
        run_script(DRIFT_SCRIPT)
        run_script(MONITOR_SCRIPT)

        with open(METRICS_FILE, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        current_f1 = float(metrics["f1_score"])
        logging.info(
            "Model monitoring F1: %.4f | Threshold: %.4f",
            current_f1,
            F1_THRESHOLD,
        )

        if current_f1 < F1_THRESHOLD:
            logging.warning("Performance threshold breached; starting retraining.")
            run_script(RETRAIN_SCRIPT)
            logging.info(
                "Retraining script completed. Check retraining_report.json "
                "for candidate acceptance/rejection."
            )
        else:
            logging.info("Performance acceptable; retraining skipped.")

        logging.info("PIPELINE COMPLETED SUCCESSFULLY | Run ID: %s", run_id)

    except Exception:
        logging.exception("PIPELINE FAILED | Run ID: %s", run_id)
        raise


if __name__ == "__main__":
    main()
