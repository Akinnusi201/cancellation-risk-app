from pathlib import Path

import pytest

from src.models.suite import build_estimator, model_parameter_defaults


ROOT = Path(__file__).resolve().parents[1]


def test_manual_experiment_parameter_overrides_are_applied():
    defaults = model_parameter_defaults("logistic_regression")
    assert defaults["C"] == 1.0
    model = build_estimator(
        "logistic_regression",
        random_seed=123,
        overrides={"C": 0.75, "max_iter": 300, "class_weight": "balanced"},
    )
    assert model.C == 0.75
    assert model.max_iter == 300
    assert model.class_weight == "balanced"
    assert model.random_state == 123


def test_manual_experiment_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="Unsupported"):
        build_estimator("logistic_regression", overrides={"not_a_parameter": 1})


def test_experiments_page_has_direct_tuning_and_candidate_handoff():
    page = (ROOT / "views" / "8_MLflow_Experiments.py").read_text()
    suite = (ROOT / "src" / "models" / "suite.py").read_text()
    retraining = (ROOT / "src" / "retraining.py").read_text()
    registry = (ROOT / "src" / "models" / "registry.py").read_text()
    assert "Run Experiment" in page
    assert "Tune parameters" in page
    assert "run_manual_experiment" in page
    assert "Mark latest experiment as Candidate" in page
    assert "train_single_model_experiment" in suite
    assert "run_role\": \"manual_experiment" in suite
    assert "run_manual_experiment" in retraining
    assert "register_experiment_model" in registry
