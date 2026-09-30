import json
from itertools import product
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR

POLICY_PATH = ARTIFACT_DIR / "business_policy.json"
DEFAULT_POLICY = {
    "avoidable_fulfillment_cost": 2000.0,
    "intervention_effectiveness": 0.55,
    "intervention_cost": 150.0,
    "false_positive_friction_cost": 50.0,
}


def load_policy() -> Dict[str, float]:
    if POLICY_PATH.exists():
        try:
            data = json.loads(POLICY_PATH.read_text())
            return {**DEFAULT_POLICY, **{k: float(v) for k, v in data.items() if k in DEFAULT_POLICY}}
        except Exception:
            pass
    return DEFAULT_POLICY.copy()


def economic_decision(probability: float, policy=None):
    p = float(max(0.0, min(1.0, probability)))
    policy = {**DEFAULT_POLICY, **(policy or {})}
    avoidable = float(policy["avoidable_fulfillment_cost"])
    effectiveness = float(policy["intervention_effectiveness"])
    intervention_cost = float(policy["intervention_cost"])
    friction_cost = float(policy["false_positive_friction_cost"])

    expected_avoidable = p * avoidable * effectiveness
    expected_false_positive_cost = (1.0 - p) * friction_cost
    net_expected_savings = expected_avoidable - intervention_cost - expected_false_positive_cost
    recommendation = "Hold for Verification" if net_expected_savings > 0 else "Approve for Fulfillment"
    return {
        "expected_avoidable_cost": expected_avoidable,
        "expected_false_positive_cost": expected_false_positive_cost,
        "net_expected_savings": net_expected_savings,
        "recommendation": recommendation,
        "policy": policy,
    }


def evaluate_business_policy(y_true, probabilities, policy=None):
    """Evaluate the profit-aware intervention rule on labeled historical orders.

    This is an estimated counterfactual business analysis. A canceled order selected for
    intervention receives avoidable_cost * effectiveness as estimated benefit. Every
    intervention pays its intervention cost; interventions on completed orders also pay
    the configured friction cost.
    """
    policy = {**DEFAULT_POLICY, **(policy or {})}
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have the same length")
    if len(y) == 0:
        raise ValueError("At least one labeled order is required")

    avoidable = float(policy["avoidable_fulfillment_cost"])
    effectiveness = float(policy["intervention_effectiveness"])
    intervention_cost = float(policy["intervention_cost"])
    friction_cost = float(policy["false_positive_friction_cost"])

    expected_net = p * avoidable * effectiveness - intervention_cost - (1.0 - p) * friction_cost
    intervene = expected_net > 0
    canceled_interventions = intervene & (y == 1)
    false_interventions = intervene & (y == 0)

    estimated_avoided_cost = float(canceled_interventions.sum() * avoidable * effectiveness)
    total_intervention_cost = float(intervene.sum() * intervention_cost)
    total_false_intervention_cost = float(false_interventions.sum() * friction_cost)
    net_savings = estimated_avoided_cost - total_intervention_cost - total_false_intervention_cost

    n = len(y)
    canceled = int((y == 1).sum())
    interventions = int(intervene.sum())
    caught = int(canceled_interventions.sum())
    false_count = int(false_interventions.sum())

    return {
        "orders": int(n),
        "canceled_orders": canceled,
        "interventions": interventions,
        "intervention_rate": float(interventions / n),
        "cancellations_intervened": caught,
        "cancellation_capture_rate": float(caught / canceled) if canceled else 0.0,
        "false_interventions": false_count,
        "false_intervention_rate": float(false_count / interventions) if interventions else 0.0,
        "estimated_avoided_cost": estimated_avoided_cost,
        "intervention_cost_total": total_intervention_cost,
        "false_intervention_cost_total": total_false_intervention_cost,
        "net_savings": float(net_savings),
        "net_savings_per_1000_orders": float(net_savings / n * 1000.0),
        "average_net_savings_per_intervention": float(net_savings / interventions) if interventions else 0.0,
        "policy": policy,
    }


def sensitivity_analysis(
    y_true,
    probabilities,
    fulfillment_costs: Iterable[float],
    effectiveness_values: Iterable[float],
    intervention_costs: Iterable[float],
    false_positive_friction_cost: float = 50.0,
):
    """Evaluate a transparent grid of business assumptions."""
    rows = []
    for fulfillment_cost, effectiveness, intervention_cost in product(
        fulfillment_costs, effectiveness_values, intervention_costs
    ):
        policy = {
            "avoidable_fulfillment_cost": float(fulfillment_cost),
            "intervention_effectiveness": float(effectiveness),
            "intervention_cost": float(intervention_cost),
            "false_positive_friction_cost": float(false_positive_friction_cost),
        }
        result = evaluate_business_policy(y_true, probabilities, policy)
        rows.append({
            **policy,
            "interventions": result["interventions"],
            "intervention_rate": result["intervention_rate"],
            "cancellation_capture_rate": result["cancellation_capture_rate"],
            "false_intervention_rate": result["false_intervention_rate"],
            "net_savings": result["net_savings"],
            "net_savings_per_1000_orders": result["net_savings_per_1000_orders"],
        })
    return pd.DataFrame(rows)
