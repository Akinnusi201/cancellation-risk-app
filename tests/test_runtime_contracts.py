import inspect
from pathlib import Path

import pytest

from src.business import economic_decision
from src.business_evaluation import evaluate_business_policy, sensitivity_analysis
from src.config import ROOT


def test_score_order_supports_runtime_telemetry_contract():
    pytest.importorskip("duckdb")
    from src.models.predict import score_order
    assert "scoring_mode" in inspect.signature(score_order).parameters


def test_business_evaluation_api_is_importable():
    result = evaluate_business_policy([0, 1], [0.1, 0.9])
    assert result["orders"] == 2
    assert callable(sensitivity_analysis)
    assert callable(economic_decision)


def test_github_actions_workflow_is_packaged():
    workflow_dir = ROOT / ".github" / "workflows"
    workflows = list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))
    assert workflows, "No GitHub Actions workflow is packaged"
    assert (ROOT / "GITHUB_ACTIONS_CI.yml").exists(), "Visible CI recovery template missing"


def test_manual_experiment_callback_is_not_a_lightgbm_parameter(monkeypatch):
    pytest.importorskip("mlflow")
    pytest.importorskip("duckdb")
    from src.models import train

    captured = {}

    def fake_fit_and_log(snapshot, dataset_version, estimator, *args, **kwargs):
        captured["params"] = estimator.get_params()
        captured["stage_callback"] = kwargs.get("stage_callback")
        return {
            "run_id": "test-run",
            "threshold": 0.5,
            "val": {},
            "test": {},
        }

    monkeypatch.setattr(train, "_fit_and_log", fake_fit_and_log)
    callback = lambda *args: None
    result = train.run_manual_experiment(
        snapshot=None,
        dataset_version="test",
        stage_callback=callback,
        n_estimators=35,
        learning_rate=0.07,
        num_leaves=15,
        log_evaluation_plots=False,
    )
    assert result["run_id"] == "test-run"
    assert captured["stage_callback"] is callback
    assert "stage_callback" not in captured["params"]
    assert "progress_callback" not in captured["params"]
    assert captured["params"]["n_estimators"] == 35
