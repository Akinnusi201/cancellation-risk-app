import pandas as pd

from src.monitoring.metrics import summarize_operations_business


def test_operations_business_impact_includes_live_simulation_but_not_historical_and_dedupes_orders():
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
            "manager_decision": None,
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
            "manager_decision": None,
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
            "manager_decision": None,
        },
        # Live demo simulation can count in Business Impact, but savings follow the manager's action.
        {
            "prediction_id": "p4",
            "order_id": "SIM-VERIFY",
            "predicted_at": "2026-09-30 10:07:00",
            "probability": 0.70,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 70.0,
            "expected_false_positive_cost": 3.0,
            "net_expected_savings": 57.0,
            "intervention_cost": 10.0,
            "scoring_mode": "simulation_live",
            "manager_decision": "Hold for Verification",
        },
        {
            "prediction_id": "p5",
            "order_id": "SIM-PENDING",
            "predicted_at": "2026-09-30 10:08:00",
            "probability": 0.95,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 95.0,
            "expected_false_positive_cost": 1.0,
            "net_expected_savings": 84.0,
            "intervention_cost": 10.0,
            "scoring_mode": "simulation_live",
            "manager_decision": None,
        },
        # Historical evaluation remains excluded.
        {
            "prediction_id": "p6",
            "order_id": "HIST",
            "predicted_at": "2026-09-30 10:09:00",
            "probability": 0.95,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 950.0,
            "expected_false_positive_cost": 0.0,
            "net_expected_savings": 940.0,
            "intervention_cost": 10.0,
            "scoring_mode": "simulation_historical",
            "manager_decision": "Hold for Verification",
        },
    ])

    used, summary = summarize_operations_business(predictions)

    assert summary["scoring_records"] == 5  # A twice, B, and two live simulation records
    assert summary["orders_scored"] == 4
    assert set(used["order_id"]) == {"A", "B", "SIM-VERIFY", "SIM-PENDING"}
    assert summary["orders_pending_review"] == 2  # manual A + pending demo order
    assert summary["orders_released"] == 1  # batch fallback recommendation
    assert summary["orders_verified"] == 1  # only manager-approved SIM-VERIFY
    assert summary["expected_cancellations_reached"] == 0.7
    assert summary["expected_cost_prevented"] == 70.0
    assert summary["verification_cost"] == 10.0
    assert summary["expected_unnecessary_check_cost"] == 3.0
    assert summary["expected_net_savings"] == 57.0
    assert summary["net_savings_per_1000_orders"] == 14250.0


def test_pending_live_simulation_counts_as_order_but_claims_no_savings_until_manager_decides():
    predictions = pd.DataFrame([
        {
            "prediction_id": "p1",
            "order_id": "SIM",
            "predicted_at": "2026-09-30",
            "probability": 0.9,
            "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 90.0,
            "expected_false_positive_cost": 1.0,
            "net_expected_savings": 79.0,
            "intervention_cost": 10.0,
            "scoring_mode": "simulation_live",
            "manager_decision": None,
        }
    ])
    used, summary = summarize_operations_business(predictions)
    assert not used.empty
    assert summary["orders_scored"] == 1
    assert summary["orders_pending_review"] == 1
    assert summary["orders_verified"] == 0
    assert summary["expected_net_savings"] == 0.0


def test_business_impact_can_exclude_demo_simulation():
    predictions = pd.DataFrame([
        {
            "prediction_id": "m1", "order_id": "M", "predicted_at": "2026-09-30",
            "probability": 0.5, "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 50.0, "expected_false_positive_cost": 2.0,
            "net_expected_savings": 38.0, "intervention_cost": 10.0,
            "scoring_mode": "manual", "manager_decision": "Hold for Verification",
        },
        {
            "prediction_id": "s1", "order_id": "S", "predicted_at": "2026-09-30",
            "probability": 0.9, "recommendation": "Hold for Verification",
            "expected_avoidable_cost": 90.0, "expected_false_positive_cost": 1.0,
            "net_expected_savings": 79.0, "intervention_cost": 10.0,
            "scoring_mode": "simulation_live", "manager_decision": "Hold for Verification",
        },
    ])
    used, summary = summarize_operations_business(predictions, include_demo=False)
    assert set(used["order_id"]) == {"M"}
    assert summary["orders_scored"] == 1
    assert summary["expected_net_savings"] == 38.0
