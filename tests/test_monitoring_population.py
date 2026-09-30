import pandas as pd

from src.monitoring.metrics import (
    MIN_MONITORING_OBSERVATIONS,
    split_monitoring_population,
)


def test_monitoring_population_excludes_demo_and_historical_traffic():
    predictions = pd.DataFrame({
        "scoring_mode": [
            "manual",
            "batch",
            "simulation_live",
            "simulation_historical",
            "simulation",
            None,
        ],
        "probability": [0.1, 0.2, 0.9, 0.8, 0.7, 0.6],
    })
    eligible, excluded, counts = split_monitoring_population(predictions)

    assert eligible["monitoring_mode"].tolist() == ["manual", "batch"]
    assert len(excluded) == 4
    assert counts["simulation_live"] == 1
    assert counts["simulation_historical"] == 1
    assert counts["legacy_or_unknown"] == 1


def test_monitoring_requires_substantial_production_sample():
    # A deliberate guard against declaring drift from a handful of demo/manual scores.
    assert MIN_MONITORING_OBSERVATIONS >= 100
