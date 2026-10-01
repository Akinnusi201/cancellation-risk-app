"""Quick pre-push repository consistency check for the end-to-end MLOps build."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
checks = []

def check(name, ok, detail):
    checks.append((name, bool(ok), detail))

def text(path):
    return (ROOT / path).read_text() if (ROOT / path).exists() else ""

predict_text = text("src/models/predict.py")
score_text = text("views/2_Score_Order.py")
monitor_text = text("views/7_Model_Monitoring.py")
metrics_text = text("src/monitoring/metrics.py")
modelops_text = text("views/6_ModelOps.py")
experiments_text = text("views/8_MLflow_Experiments.py")
suite_text = text("src/models/suite.py")
retraining_text = text("src/retraining.py")
readme_text = text("README.md")
workflows = list((ROOT / ".github/workflows").glob("*.yml")) + list((ROOT / ".github/workflows").glob("*.yaml"))

check("Packaged production model", (ROOT / "artifacts/production_model.pkl").exists(), "artifacts/production_model.pkl")
check("Five-model starter registry", (ROOT / "artifacts/model_registry/registry.json").exists(), "artifacts/model_registry/registry.json")
if (ROOT / "artifacts/model_registry/registry.json").exists():
    registry = json.loads((ROOT / "artifacts/model_registry/registry.json").read_text())
    families = {m.get("model_family") for m in registry.get("models", [])}
    check("All five model families", families == {"logistic_regression", "random_forest", "extra_trees", "lightgbm", "xgboost"}, str(sorted(families)))
    check("LightGBM initially production", any(m.get("model_family") == "lightgbm" and m.get("status") == "PRODUCTION" for m in registry.get("models", [])), "starter status")
check("Colab end-to-end notebook", (ROOT / "notebooks/end_to_end_ml_workflow.ipynb").exists(), "notebooks/end_to_end_ml_workflow.ipynb")
check("Multi-model training suite", all(x in suite_text for x in ["RandomForestClassifier", "ExtraTreesClassifier", "XGBClassifier", "LGBMClassifier", "LogisticRegression"]), "src/models/suite.py")
check("MLflow tracking in suite", "mlflow.log_metrics" in suite_text and "mlflow.log_params" in suite_text, "parameters + metrics logged")
check("Candidate qualification gates", "DEFAULT_GATES" in suite_text and "select_candidate" in suite_text, "qualification before candidate")
check("Manual promotion governance", "Promote Candidate to Production" in modelops_text and "promote_registered_model" in modelops_text, "developer approval required")
check("Optional candidate package import", "import_candidate_package" in modelops_text, "optional notebook handoff")
check("Monitoring retraining policy", "evaluate_retraining_policy" in monitor_text and "labeled_runtime_performance" in metrics_text, "hybrid drift + labeled performance")
check("Direct retraining", "run_local_retraining" in retraining_text and "Run Five-Model Retraining Now" in modelops_text, "five-model suite runs in app")
check("Automatic retraining setting", "automatic_retraining_enabled" in retraining_text and "Automatically retrain" in modelops_text, "developer-controlled automation")
check("No managed cloud-job dependency", "notebookExecutionJobs" not in retraining_text and "google.auth" not in retraining_text, "in-app automation")
check("Production-only drift population", "PRODUCTION_SCORING_MODES" in metrics_text and "simulation_live" not in text("src/monitoring/metrics.py").split("PRODUCTION_SCORING_MODES",1)[1].split("}",1)[0], "simulation excluded from drift")
check("Configurable demo business impact", "OPERATIONS_BUSINESS_MODES" in metrics_text and '"simulation_live"' in metrics_text, "simulation can be included in business economics")
check("Scoring telemetry", "scoring_mode=" in predict_text and "latency_ms" in predict_text, "runtime monitoring")
check("Role-safe score page", "_score_order_compat" in score_text, "rolling-upgrade compatibility")
check("GitHub Actions", bool(workflows), ".github/workflows")
check("Dockerfile", (ROOT / "Dockerfile").exists(), "Dockerfile")
check("First-deployment README", all(x in readme_text for x in ["What problem does this solve?", "Starter model registry", "Retraining settings", "Candidate selection and deployment", "Monitoring and retraining", "Reproducibility"]), "README.md")
check("Experiments GUI", "Experiment Runs" in experiments_text and "Compare Models" in experiments_text and "MLflow Tracking" in experiments_text, "plain-language MLflow-style interface")
check("Manual tuned experiments", all(x in experiments_text for x in ["Run Experiment", "Tune parameters", "run_manual_experiment", "Mark latest experiment as Candidate"]) and "train_single_model_experiment" in suite_text and "run_manual_experiment" in retraining_text, "in-app developer tuning + MLflow tracking")
check("Validation-only candidate selection", "test_holdout_role" in suite_text and "selection_split" in suite_text and "_selection_evidence" in suite_text, "test holdout excluded from candidate ranking")
check("Feature schema compatibility", "except ImportError" in suite_text and 'FEATURE_SCHEMA_VERSION = "order_features_v1"' in suite_text, "pre-v4.4 config.py does not crash suite import")
check("Final fair benchmark", (ROOT / "artifacts/final_benchmark/benchmark_summary.json").exists() and (ROOT / "artifacts/final_benchmark/benchmark_comparison.csv").exists(), "full-data five-model evidence")
if (ROOT / "artifacts/final_benchmark/benchmark_summary.json").exists():
    benchmark = json.loads((ROOT / "artifacts/final_benchmark/benchmark_summary.json").read_text())
    contract = benchmark.get("benchmark_contract", {})
    check("Benchmark reproducibility contract", all(contract.get(k) for k in ["same_dataset_version", "same_dataset_fingerprint", "same_temporal_split", "same_feature_schema", "same_business_policy"]), "same dataset/split/schema/policy")
    check("Benchmark test holdout guard", contract.get("candidate_selection_uses") == "validation_only" and contract.get("test_holdout_use") == "final_unbiased_evaluation_only", "validation selects; test reports")

failed = False
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    failed = failed or not ok
if failed:
    raise SystemExit("Repository is incomplete. Fix failed checks before pushing.")
print("Repository consistency check passed.")
