import pandas as pd

from src.monitoring.metrics import summarize_operations_business


def test_operations_business_impact_excludes_simulation_and_historical_and_dedupes_orders():
    predictions = pd.DataFrame([
        {
            "prediction_id": "p1",
            "order_id": "A",
            "predicted_at": "2026-09-30 10:00:00",
            "probability": 0.80,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 80.0,
            "expected_false_positive_cost": 2.0,
            "net_expected_savings": 68.0,
            "intervention_cost": 10.0,
            "scoring_mode": "manual",
        },
        # Same order rescored later. Only this latest record should count.
        {
            "prediction_id": "p2",
            "order_id": "A",
            "predicted_at": "2026-09-30 10:05:00",
            "probability": 0.90,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 90.0,
            "expected_false_positive_cost": 1.0,
            "net_expected_savings": 79.0,
            "intervention_cost": 10.0,
            "scoring_mode": "manual",
        },
        {
            "prediction_id": "p3",
            "order_id": "B",
            "predicted_at": "2026-09-30 10:06:00",
            "probability": 0.20,
            "recommendation": "Approve for Fulfillment",
            "expected_avoidable_cost": 20.0,
            "expected_false_positive_cost": 4.0,
            "net_expected_savings": -1.0,
            "intervention_cost": 17.0,
            "scoring_mode": "batch",
        },
        # Demo/evaluation traffic must not contaminate live business impact.
        {
            "prediction_id": "p4",
            "order_id": "SIM",
            "predicted_at": "2026-09-30 10:07:00",
            "probability": 0.99,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 999.0,
            "expected_false_positive_cost": 0.0,
            "net_expected_savings": 989.0,
            "intervention_cost": 10.0,
            "scoring_mode": "simulation_live",
        },
        {
            "prediction_id": "p5",
            "order_id": "HIST",
            "predicted_at": "2026-09-30 10:08:00",
            "probability": 0.95,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 950.0,
            "expected_false_positive_cost": 0.0,
            "net_expected_savings": 940.0,
            "intervention_cost": 10.0,
            "scoring_mode": "simulation_historical",
        },
    ])

    used, summary = summarize_operations_business(predictions)

    assert summary["scoring_records"] == 3  # two A scores + one B score before dedupe
    assert summary["orders_scored"] == 2
    assert set(used["order_id"]) == {"A", "B"}
    assert summary["orders_verified"] == 1
    assert summary["verification_rate"] == 0.5
    assert summary["expected_cancellations_reached"] == 0.9
    assert summary["expected_cost_prevented"] == 90.0
    assert summary["verification_cost"] == 10.0
    assert summary["expected_unnecessary_check_cost"] == 1.0
    assert summary["expected_net_savings"] == 79.0
    assert summary["net_savings_per_1000_orders"] == 39500.0


def test_operations_business_impact_is_empty_for_demo_only():
    predictions = pd.DataFrame({
        "prediction_id": ["p1", "p2"],
        "order_id": ["A", "B"],
        "predicted_at": ["2026-09-30", "2026-09-30"],
        "probability": [0.9, 0.8],
        "recommendation": ["Hold for Verification", "Hold for Verification"],
        "scoring_mode": ["simulation_live", "simulation_historical"],
    })
    used, summary = summarize_operations_business(predictions)
    assert used.empty
    assert summary["orders_scored"] == 0
    assert summary["expected_net_savings"] == 0.0
