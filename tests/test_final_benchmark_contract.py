from pathlib import Path

from src.models.suite import select_candidate
from src.models.benchmark import load_final_benchmark, load_final_benchmark_table

ROOT = Path(__file__).resolve().parents[1]


def _record(model_id, val_roc, test_roc, val_pr=0.90, val_brier=0.10, val_recall90=0.60, val_savings=1000):
    return {
        "model_id": model_id,
        "val_metrics": {
            "roc_auc": val_roc,
            "pr_auc": val_pr,
            "brier": val_brier,
            "recall_at_90_precision": val_recall90,
        },
        "test_metrics": {
            "roc_auc": test_roc,
            "pr_auc": 0.99,
            "brier": 0.01,
            "recall_at_90_precision": 0.99,
        },
        "val_business_metrics": {"net_savings_per_1000_orders": val_savings},
        "business_metrics": {"net_savings_per_1000_orders": 999999},
    }


def test_candidate_selection_never_ranks_on_test_holdout():
    # Model A looks much better on the final test set, but model B is stronger on
    # validation. Candidate selection must choose B to preserve the test holdout.
    a = _record("model_a", val_roc=0.82, test_roc=0.99, val_savings=500)
    b = _record("model_b", val_roc=0.93, test_roc=0.80, val_savings=1500)
    selected = select_candidate([a, b])
    assert selected["model_id"] == "model_b"
    assert selected["qualification"]["evidence_split"] == "validation"
    assert selected["selection_evidence"]["split"] == "validation"


def test_final_benchmark_workflow_is_exposed_in_ui_and_script():
    page = (ROOT / "views/8_MLflow_Experiments.py").read_text()
    script = (ROOT / "scripts/run_final_benchmark.py").read_text()
    benchmark = (ROOT / "src/models/benchmark.py").read_text()
    assert "Final Benchmark" in page
    assert "Run Full Five-Model Benchmark" in page
    assert "run_final_benchmark" in page
    assert "candidate_selection_uses" in benchmark
    assert "test_holdout_use" in benchmark
    assert "persist_model_artifacts=False" in script


def test_packaged_final_benchmark_is_full_data_and_fair():
    summary = load_final_benchmark()
    table = load_final_benchmark_table()
    assert summary is not None
    assert summary["dataset_version"] == "pakistan_seed_v1"
    assert int(summary["source_rows"]) == 318135
    assert len(table) == 5
    assert set(table["model_family"]) == {
        "logistic_regression", "random_forest", "extra_trees", "lightgbm", "xgboost"
    }
    contract = summary["benchmark_contract"]
    assert contract["candidate_selection_uses"] == "validation_only"
    assert contract["test_holdout_use"] == "final_unbiased_evaluation_only"
    split_ids = {m.get("split_id") for m in summary["models"]}
    fingerprints = {m.get("dataset_fingerprint") for m in summary["models"]}
    assert split_ids == {summary["split_manifest"]["split_id"]}
    assert fingerprints == {summary["dataset_fingerprint"]}
