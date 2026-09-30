"""Business-policy evaluation helpers kept separate from live scoring.

This module is intentionally independent of the Streamlit UI and provides a
compatibility path for deployments that still have an older ``src.business``
module from a previous app version.
"""
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from src.business import DEFAULT_POLICY


def evaluate_business_policy(y_true, probabilities, policy=None):
    """Evaluate the profit-aware intervention rule on labeled historical orders."""
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
    """Evaluate a grid of economic assumptions."""
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
