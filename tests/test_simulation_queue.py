import numpy as np
import pandas as pd

from src.simulation import build_balanced_live_queue, queue_mix


def test_balanced_live_queue_stratifies_predicted_risk_without_changing_rows():
    rows = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(300)],
        "is_canceled": ([0, 1] * 150),
    })
    probabilities = np.concatenate([
        np.linspace(0.02, 0.30, 100),
        np.linspace(0.40, 0.65, 100),
        np.linspace(0.75, 0.98, 100),
    ])
    live = build_balanced_live_queue(rows, probabilities, max_rows=200)
    mix = queue_mix(live)
    assert len(live) == 200
    assert set(live["order_id"]).issubset(set(rows["order_id"]))
    assert set(live["simulation_risk_band"]) == {"LOW", "MEDIUM", "HIGH"}
    assert abs(mix["LOW"] - 0.35) < 0.02
    assert abs(mix["MEDIUM"] - 0.30) < 0.02
    assert abs(mix["HIGH"] - 0.35) < 0.02


def test_packaged_live_and_historical_queues_have_distinct_purposes():
    live = pd.read_csv("artifacts/demo_orders.csv.gz", low_memory=False)
    historical = pd.read_csv("artifacts/historical_demo_orders.csv.gz", low_memory=False)
    assert {"simulation_probability", "simulation_risk_band"}.issubset(live.columns)
    mix = live["simulation_risk_band"].value_counts(normalize=True)
    assert 0.30 <= float(mix.get("LOW", 0)) <= 0.40
    assert 0.25 <= float(mix.get("MEDIUM", 0)) <= 0.35
    assert 0.30 <= float(mix.get("HIGH", 0)) <= 0.40
    # Historical evaluation keeps the naturally high late-period cancellation rate.
    assert historical["is_canceled"].mean() > 0.75
