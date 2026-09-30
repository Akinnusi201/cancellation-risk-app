import json
from pathlib import Path

import pandas as pd


def test_packaged_monitoring_artifacts_exist():
    eval_dir = Path("artifacts/production_evaluation")
    predictions = pd.read_csv(eval_dir / "test_predictions.csv.gz")
    comparison = pd.read_csv(eval_dir / "baseline_comparison.csv")
    meta = json.loads(Path("artifacts/active_model.json").read_text())
    assert len(predictions) > 1000
    assert {"is_canceled", "probability"}.issubset(predictions.columns)
    assert set(comparison["Model"]) == {"Logistic Regression", "LightGBM Production"}
    assert "recall_at_90_precision" in meta["test_metrics"]
