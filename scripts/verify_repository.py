"""Quick pre-push repository consistency check for the computational build."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = []


def check(name, ok, detail):
    checks.append((name, bool(ok), detail))


predict_text = (ROOT / "src/models/predict.py").read_text()
business_text = (ROOT / "src/business.py").read_text()
score_text = (ROOT / "views/2_Score_Order.py").read_text()
monitor_text = (ROOT / "views/7_Model_Monitoring.py").read_text()
monitor_metrics_text = (ROOT / "src/monitoring/metrics.py").read_text()
mlflow_view_text = (ROOT / "views/8_MLflow_Experiments.py").read_text()
train_text = (ROOT / "src/models/train.py").read_text()
workflows = list((ROOT / ".github/workflows").glob("*.yml")) + list((ROOT / ".github/workflows").glob("*.yaml"))

check("Packaged production model", (ROOT / "artifacts/production_model.pkl").exists(), "artifacts/production_model.pkl")
check("Scoring telemetry backend", "scoring_mode=" in predict_text, "src/models/predict.py")
check("Score page compatibility", "_score_order_compat" in score_text, "views/2_Score_Order.py")
check("Score page rolling-upgrade compatibility", "from src.ui.common import" not in score_text and "getattr(ui_common, \"load_historical_demo_orders\"" in score_text, "historical helper is optional during partial upgrades")
check("Business evaluation functions", "def evaluate_business_policy" in business_text or (ROOT / "src/business_evaluation.py").exists(), "business evaluation module")
check("Monitoring compatibility", "src.business_evaluation" in monitor_text, "views/7_Model_Monitoring.py")
check("GitHub Actions workflow", bool(workflows), ".github/workflows/*.yml")
check("Visible CI recovery template", (ROOT / "GITHUB_ACTIONS_CI.yml").exists(), "GITHUB_ACTIONS_CI.yml")
check("Dockerfile", (ROOT / "Dockerfile").exists(), "Dockerfile")
check("Stage-by-stage MLflow progress", "stage_callback=update_stage" in mlflow_view_text and "Train LightGBM" in train_text, "views/8_MLflow_Experiments.py + src/models/train.py")
check("Fast manual experiment logging", "log_model_artifact=False" in train_text, "manual MLflow runs skip non-promotable model serialization")
check("Callback isolation", "n_estimators=70" in train_text and "stage_callback=stage_callback" in train_text, "manual callbacks stay outside LightGBM hyperparameters")
check("Balanced live simulation", (ROOT / "artifacts/demo_orders.csv.gz").exists() and (ROOT / "artifacts/historical_demo_orders.csv.gz").exists(), "separate live and historical queues")
check("Production-only drift population", "PRODUCTION_SCORING_MODES" in monitor_metrics_text and "split_monitoring_population" in monitor_metrics_text, "simulation/evaluation traffic excluded from drift")
check("Minimum drift sample guard", "MIN_MONITORING_OBSERVATIONS = 100" in monitor_metrics_text and "INSUFFICIENT RUNTIME DATA" in monitor_text, "no drift label before 100 eligible production scores")
check("Batch prediction telemetry", '"batch"' in predict_text and "executemany" in predict_text, "batch predictions are persisted for production monitoring")

failed = False
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    failed = failed or not ok

if failed:
    raise SystemExit("Repository is incomplete. Copy the full project, including hidden .github files, before pushing.")
print("Repository consistency check passed.")
