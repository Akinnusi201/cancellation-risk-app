import json
from pathlib import Path

from src.models.suite import qualify_model, select_candidate

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_registry_contains_five_model_families_with_lightgbm_production():
    registry = json.loads((ROOT / "artifacts/model_registry/registry.json").read_text())
    models = registry["models"]
    assert {m["model_family"] for m in models} == {
        "logistic_regression", "random_forest", "extra_trees", "lightgbm", "xgboost"
    }
    production = [m for m in models if m["status"] == "PRODUCTION"]
    assert len(production) == 1
    assert production[0]["model_family"] == "lightgbm"
    active = json.loads((ROOT / "artifacts/active_model.json").read_text())
    assert active["model_version"] == production[0]["model_id"]
    for model in models:
        assert (ROOT / model["model_artifact"]).exists()


def _record(name, roc, pr, brier, recall90, savings):
    return {
        "model_id": name,
        "test_metrics": {
            "roc_auc": roc,
            "pr_auc": pr,
            "brier": brier,
            "recall_at_90_precision": recall90,
        },
        "business_metrics": {"net_savings_per_1000_orders": savings},
    }


def test_candidate_selection_uses_gates_before_weighted_score():
    weak = _record("weak", 0.70, 0.90, 0.10, 0.90, 1_000_000)
    good = _record("good", 0.91, 0.97, 0.09, 0.98, 500_000)
    better = _record("better", 0.92, 0.975, 0.08, 0.99, 600_000)
    assert qualify_model(weak)[0] is False
    selected = select_candidate([weak, good, better])
    assert selected["model_id"] == "better"
    assert selected["qualification"]["passed"] is True


def test_colab_notebook_contains_end_to_end_suite_and_candidate_handoff():
    notebook = json.loads((ROOT / "notebooks/end_to_end_ml_workflow.ipynb").read_text())
    source = "\n".join(
        cell_source if isinstance((cell_source := cell.get("source", "")), str) else "".join(cell_source)
        for cell in notebook["cells"]
    )
    assert "train_model_suite" in source
    assert "MLFLOW_TRACKING_URI" in source
    assert "candidate_package" in source
    assert "T4 GPU" in source
