import inspect
from pathlib import Path

from src.business import economic_decision
from src.business_evaluation import evaluate_business_policy, sensitivity_analysis
from src.config import ROOT
from src.models.predict import score_order


def test_score_order_supports_runtime_telemetry_contract():
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
