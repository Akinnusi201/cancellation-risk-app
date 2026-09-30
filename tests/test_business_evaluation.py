import numpy as np

from src.business import evaluate_business_policy, sensitivity_analysis


def test_business_policy_uses_actual_holdout_outcomes_for_estimated_savings():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    policy = {
        "avoidable_fulfillment_cost": 100.0,
        "intervention_effectiveness": 0.5,
        "intervention_cost": 10.0,
        "false_positive_friction_cost": 5.0,
    }
    result = evaluate_business_policy(y, p, policy)
    assert result["interventions"] == 2
    assert result["cancellations_intervened"] == 1
    assert result["false_interventions"] == 1
    assert result["estimated_avoided_cost"] == 50.0
    assert result["intervention_cost_total"] == 20.0
    assert result["false_intervention_cost_total"] == 5.0
    assert result["net_savings"] == 25.0
    assert result["net_savings_per_1000_orders"] == 6250.0


def test_sensitivity_analysis_returns_all_scenarios():
    out = sensitivity_analysis(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.2, 0.1],
        fulfillment_costs=[100, 200],
        effectiveness_values=[0.25, 0.5],
        intervention_costs=[10, 20],
        false_positive_friction_cost=5,
    )
    assert len(out) == 8
    assert "net_savings_per_1000_orders" in out.columns
