import uuid
import numpy as np
import pandas as pd

from src.business import economic_decision
from src.config import FEATURES
from src.database.duckdb_manager import connect, now
from src.models.registry import load_active_model


def explain_by_baseline(model, row: pd.DataFrame, reference: pd.DataFrame, top_k=3):
    base_prob = float(model.predict_proba(row[FEATURES])[:, 1][0])
    impacts = []
    for f in FEATURES:
        alt = row.copy()
        if pd.api.types.is_numeric_dtype(reference[f]):
            alt[f] = reference[f].median()
        else:
            m = reference[f].mode()
            alt[f] = m.iat[0] if len(m) else "unknown"
        p = float(model.predict_proba(alt[FEATURES])[:, 1][0])
        impacts.append((f, base_prob - p))
    return sorted([x for x in impacts if x[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]


def _actual_outcome(row: pd.DataFrame):
    if "is_canceled" not in row.columns or pd.isna(row.iloc[0].get("is_canceled")):
        return None
    return "Canceled" if int(row.iloc[0]["is_canceled"]) == 1 else "Completed"


def score_order(row: pd.DataFrame, reference: pd.DataFrame, policy=None, persist=True, explain=True):
    model, meta = load_active_model()
    if model is None:
        raise RuntimeError("No packaged production model is available.")
    prob = float(model.predict_proba(row[FEATURES])[:, 1][0])
    threshold = float(meta.get("threshold", 0.5))
    economics = economic_decision(prob, policy)
    reasons = explain_by_baseline(model, row, reference) if explain else []
    prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
    actual = _actual_outcome(row)

    if persist:
        with connect() as con:
            con.execute(
                """
                INSERT INTO predictions
                (prediction_id, order_id, predicted_at, model_name, model_version, dataset_version,
                 probability, threshold, recommendation, actual_outcome, expected_avoidable_cost,
                 expected_false_positive_cost, net_expected_savings, avoidable_fulfillment_cost,
                 intervention_effectiveness, intervention_cost, false_positive_friction_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    prediction_id, str(row.iloc[0]["order_id"]), now(), meta.get("model_name", "lightgbm"),
                    str(meta.get("model_version", "unknown")), meta.get("dataset_version", "unknown"), prob,
                    threshold, economics["recommendation"], actual, economics["expected_avoidable_cost"],
                    economics["expected_false_positive_cost"], economics["net_expected_savings"],
                    economics["policy"]["avoidable_fulfillment_cost"], economics["policy"]["intervention_effectiveness"],
                    economics["policy"]["intervention_cost"], economics["policy"]["false_positive_friction_cost"],
                ],
            )
    return {
        "prediction_id": prediction_id,
        "probability": prob,
        "threshold": threshold,
        "risk_level": "HIGH" if prob >= threshold else "LOW",
        "recommendation": economics["recommendation"],
        "reasons": reasons,
        "actual_outcome": actual,
        "meta": meta,
        **economics,
    }


def score_batch(rows: pd.DataFrame, policy=None):
    model, meta = load_active_model()
    if model is None:
        raise RuntimeError("No packaged production model is available.")
    probs = model.predict_proba(rows[FEATURES])[:, 1]
    threshold = float(meta.get("threshold", 0.5))
    output = rows.copy()
    output["cancellation_probability"] = probs
    output["risk_level"] = np.where(probs >= threshold, "HIGH", "LOW")
    economic = [economic_decision(float(p), policy) for p in probs]
    output["expected_avoidable_cost"] = [x["expected_avoidable_cost"] for x in economic]
    output["expected_false_positive_cost"] = [x["expected_false_positive_cost"] for x in economic]
    output["net_expected_savings"] = [x["net_expected_savings"] for x in economic]
    output["recommendation"] = [x["recommendation"] for x in economic]
    return output


def record_decision(scored, order_id, decision):
    decision_id = f"dec_{uuid.uuid4().hex[:12]}"
    with connect() as con:
        con.execute(
            """
            INSERT INTO manager_decisions
            (decision_id, prediction_id, order_id, decided_at, manager_decision, recommendation,
             probability, threshold, actual_outcome, model_name, model_version, net_expected_savings,
             avoidable_fulfillment_cost, intervention_effectiveness, intervention_cost, false_positive_friction_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                decision_id, scored["prediction_id"], str(order_id), now(), decision, scored["recommendation"],
                scored["probability"], scored["threshold"], scored.get("actual_outcome"),
                scored["meta"].get("model_name", "lightgbm"), str(scored["meta"].get("model_version", "unknown")),
                scored.get("net_expected_savings"), scored.get("policy", {}).get("avoidable_fulfillment_cost"),
                scored.get("policy", {}).get("intervention_effectiveness"), scored.get("policy", {}).get("intervention_cost"),
                scored.get("policy", {}).get("false_positive_friction_cost"),
            ],
        )
    return decision_id
