import numpy as np

from src.models.evaluate import metrics, recall_at_fixed_precision


def test_recall_at_fixed_precision_respects_precision_constraint():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.95, 0.90, 0.80, 0.20, 0.85, 0.10])
    result = recall_at_fixed_precision(y, p, min_precision=0.90)
    assert result["precision"] >= 0.90
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["threshold"] <= 1


def test_metrics_include_fixed_precision_operating_points():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.95, 0.90, 0.80, 0.20, 0.85, 0.10])
    result = metrics(y, p, threshold=0.5)
    assert "recall_at_80_precision" in result
    assert "recall_at_90_precision" in result
    assert "threshold_at_90_precision" in result
