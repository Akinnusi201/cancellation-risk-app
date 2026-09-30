import json
from pathlib import Path
from typing import Dict

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
