import json

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Only explicitly production-like scoring paths contribute to runtime model
# monitoring. Demo simulation and retrospective evaluation are intentionally
# excluded because their sampling design is not representative of live traffic.
PRODUCTION_SCORING_MODES = {
    "manual",
    "batch",
    "production",
    "production_manual",
    "production_batch",
    "api",
}
MIN_MONITORING_OBSERVATIONS = 100


# Prototype business-impact reporting includes live simulated customer orders
# alongside manual/batch Operations scoring. This is intentionally broader than
# technical drift monitoring, where simulation remains excluded.
OPERATIONS_BUSINESS_MODES = {
    "manual",
    "batch",
    "production_manual",
    "production_batch",
    "simulation_live",
}


def _empty_operations_business_summary():
    return {
        "scoring_records": 0,
        "orders_scored": 0,
        "orders_pending_review": 0,
        "orders_released": 0,
        "orders_verified": 0,
        "verification_rate": 0.0,
        "expected_cancellations_reached": 0.0,
        "expected_cost_prevented": 0.0,
        "verification_cost": 0.0,
        "expected_unnecessary_check_cost": 0.0,
        "expected_net_savings": 0.0,
        "net_savings_per_1000_orders": 0.0,
        "average_net_savings_per_verified_order": 0.0,
        "mode_counts": {},
        "first_scored_at": None,
        "last_scored_at": None,
    }


def summarize_operations_business(predictions: pd.DataFrame):
    """Summarize prototype business value from Operations-facing scoring.

    Live simulation counts for the prototype, but it remains excluded from model
    drift monitoring. For simulated customer orders, economic impact is counted
    only after the Operations Manager makes a decision. Manual/batch records that
    have no explicit manager decision fall back to the system recommendation.
    """
    empty_summary = _empty_operations_business_summary()
    if predictions is None or predictions.empty:
        return pd.DataFrame(), empty_summary

    df = predictions.copy()
    if "scoring_mode" not in df.columns:
        return pd.DataFrame(), empty_summary
    df["monitoring_mode"] = df["scoring_mode"].map(normalize_scoring_mode)
    df = df[df["monitoring_mode"].isin(OPERATIONS_BUSINESS_MODES)].copy()
    if df.empty:
        return df, empty_summary

    scoring_records = len(df)
    if "predicted_at" in df.columns:
        df["predicted_at"] = pd.to_datetime(df["predicted_at"], errors="coerce")
        df = df.sort_values("predicted_at")

    if "order_id" in df.columns:
        order_key = df["order_id"].astype("string")
        if "prediction_id" in df.columns:
            order_key = order_key.fillna(df["prediction_id"].astype("string"))
        df = df.assign(_order_key=order_key).drop_duplicates("_order_key", keep="last")

    # Actual manager decisions control the prototype action when available.
    manager = df.get("manager_decision", pd.Series(index=df.index, dtype="object"))
    manager = manager.astype("string")
    action = manager.copy()
    no_manager_action = action.isna() | action.eq("") | action.eq("<NA>")

    # A newly created simulated customer order remains pending until reviewed.
    simulation_pending = df["monitoring_mode"].eq("simulation_live") & no_manager_action
    fallback = no_manager_action & ~simulation_pending
    action.loc[fallback] = df.loc[fallback, "recommendation"].astype("string")
    action.loc[simulation_pending] = "Awaiting Operations Review"
    df["business_action"] = action

    verified = df["business_action"].eq("Hold for Verification")
    released = df["business_action"].eq("Approve for Fulfillment")
    pending = df["business_action"].eq("Awaiting Operations Review")
    verified_df = df.loc[verified].copy()
    orders = int(len(df))
    verified_n = int(verified.sum())

    def numeric_sum(column):
        if column not in verified_df.columns:
            return 0.0
        return float(pd.to_numeric(verified_df[column], errors="coerce").fillna(0.0).sum())

    probs = pd.to_numeric(verified_df.get("probability", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    expected_cancellations = float(probs.sum())
    cost_prevented = numeric_sum("expected_avoidable_cost")
    verification_cost = numeric_sum("intervention_cost")
    unnecessary_cost = numeric_sum("expected_false_positive_cost")
    net_savings = numeric_sum("net_expected_savings")

    summary = {
        "scoring_records": int(scoring_records),
        "orders_scored": orders,
        "orders_pending_review": int(pending.sum()),
        "orders_released": int(released.sum()),
        "orders_verified": verified_n,
        "verification_rate": float(verified_n / orders) if orders else 0.0,
        "expected_cancellations_reached": expected_cancellations,
        "expected_cost_prevented": cost_prevented,
        "verification_cost": verification_cost,
        "expected_unnecessary_check_cost": unnecessary_cost,
        "expected_net_savings": net_savings,
        "net_savings_per_1000_orders": float(net_savings / orders * 1000.0) if orders else 0.0,
        "average_net_savings_per_verified_order": float(net_savings / verified_n) if verified_n else 0.0,
        "mode_counts": {str(k): int(v) for k, v in df["monitoring_mode"].value_counts().to_dict().items()},
        "first_scored_at": df["predicted_at"].min() if "predicted_at" in df.columns else None,
        "last_scored_at": df["predicted_at"].max() if "predicted_at" in df.columns else None,
    }
    return df.drop(columns=["_order_key"], errors="ignore"), summary


def operations_business_summary(model_version=None):
    """Load Operations/prototype predictions and their latest manager decisions."""
    from src.database.duckdb_manager import dataframe

    where = "" if model_version is None else " WHERE model_version = ?"
    params = [] if model_version is None else [str(model_version)]
    try:
        predictions = dataframe(
            f"""
            SELECT prediction_id, order_id, predicted_at, probability, recommendation,
                   expected_avoidable_cost, expected_false_positive_cost,
                   net_expected_savings, intervention_cost, scoring_mode
            FROM predictions{where}
            ORDER BY predicted_at
            """,
            params,
        )
        decisions = dataframe(
            """
            SELECT prediction_id, manager_decision, decided_at
            FROM manager_decisions
            ORDER BY decided_at
            """
        )
        if not decisions.empty:
            decisions["decided_at"] = pd.to_datetime(decisions["decided_at"], errors="coerce")
            decisions = decisions.sort_values("decided_at").drop_duplicates("prediction_id", keep="last")
            predictions = predictions.merge(
                decisions[["prediction_id", "manager_decision"]],
                on="prediction_id",
                how="left",
            )
        else:
            predictions["manager_decision"] = None
    except Exception:
        predictions = pd.DataFrame()
    return summarize_operations_business(predictions)

def population_stability_index(reference, current, bins=10):
    """Population Stability Index using reference quantile bins."""
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if len(ref) < 20 or len(cur) < 20:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = pd.cut(ref, edges, include_lowest=True).value_counts(sort=False)
    cur_counts = pd.cut(cur, edges, include_lowest=True).value_counts(sort=False)
    eps = 1e-6
    ref_pct = np.clip(ref_counts.to_numpy(dtype=float) / len(ref), eps, None)
    cur_pct = np.clip(cur_counts.to_numpy(dtype=float) / len(cur), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_total_variation(reference, current):
    ref = pd.Series(reference).fillna("<missing>").astype(str)
    cur = pd.Series(current).fillna("<missing>").astype(str)
    if len(ref) < 20 or len(cur) < 20:
        return None
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    rp = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cp = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    return float(0.5 * np.abs(rp - cp).sum())


def drift_label(value, metric="psi"):
    if value is None or pd.isna(value):
        return "INSUFFICIENT DATA"
    if metric == "psi":
        if value < 0.10:
            return "STABLE"
        if value < 0.25:
            return "WATCH"
        return "DRIFT"
    # Total variation has no universal threshold; these are transparent project heuristics.
    if value < 0.10:
        return "STABLE"
    if value < 0.20:
        return "WATCH"
    return "DRIFT"


def normalize_scoring_mode(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "legacy_or_unknown"
    text = str(value).strip().lower()
    return text or "legacy_or_unknown"


def split_monitoring_population(predictions: pd.DataFrame):
    """Split logged predictions into eligible production traffic and excluded traffic.

    The returned eligible frame is the only population that should be used for
    runtime drift, latency, intervention-rate, and live business statistics.
    Simulation and historical-evaluation predictions remain available in the
    excluded frame for auditability.
    """
    if predictions is None or predictions.empty:
        empty = pd.DataFrame() if predictions is None else predictions.copy()
        return empty, empty.copy(), {}

    df = predictions.copy()
    if "scoring_mode" not in df.columns:
        df["scoring_mode"] = "legacy_or_unknown"
    df["monitoring_mode"] = df["scoring_mode"].map(normalize_scoring_mode)
    eligible_mask = df["monitoring_mode"].isin(PRODUCTION_SCORING_MODES)
    eligible = df.loc[eligible_mask].copy()
    excluded = df.loc[~eligible_mask].copy()
    mode_counts = df["monitoring_mode"].value_counts(dropna=False).to_dict()
    return eligible, excluded, {str(k): int(v) for k, v in mode_counts.items()}


def runtime_prediction_summary(model_version=None, production_only=True):
    from src.database.duckdb_manager import dataframe

    where = "" if model_version is None else " WHERE model_version = ?"
    params = [] if model_version is None else [str(model_version)]
    # New deployments include telemetry columns. During a rolling upgrade, an
    # older runtime database/module may not have them yet, so fall back to the
    # core prediction fields instead of crashing the monitoring page.
    try:
        all_predictions = dataframe(
            f"""
            SELECT predicted_at, probability, threshold, recommendation, latency_ms,
                   net_expected_savings, scoring_mode, feature_json
            FROM predictions{where}
            ORDER BY predicted_at DESC
            """,
            params,
        )
    except Exception:
        all_predictions = dataframe(
            f"""
            SELECT predicted_at, probability, threshold, recommendation, net_expected_savings
            FROM predictions{where}
            ORDER BY predicted_at DESC
            """,
            params,
        )
        for col in ["latency_ms", "scoring_mode", "feature_json"]:
            all_predictions[col] = np.nan

    if all_predictions.empty:
        return all_predictions, {
            "total_predictions": 0,
            "eligible_predictions": 0,
            "excluded_predictions": 0,
            "mode_counts": {},
            "minimum_required": MIN_MONITORING_OBSERVATIONS,
            "monitoring_ready": False,
        }

    eligible, excluded, mode_counts = split_monitoring_population(all_predictions)
    df = eligible if production_only else all_predictions

    summary = {
        "total_predictions": int(len(all_predictions)),
        "eligible_predictions": int(len(eligible)),
        "excluded_predictions": int(len(excluded)),
        "mode_counts": mode_counts,
        "minimum_required": MIN_MONITORING_OBSERVATIONS,
        "monitoring_ready": bool(len(eligible) >= MIN_MONITORING_OBSERVATIONS),
    }

    if df.empty:
        return df, summary

    latency = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    summary.update({
        "predictions": len(df),
        "average_risk": float(df["probability"].mean()),
        "high_risk_rate": float((df["probability"] >= df["threshold"]).mean()),
        "intervention_rate": float((df["recommendation"] == "Hold for Verification").mean()),
        "expected_net_savings_total": float(
            df.loc[df["recommendation"] == "Hold for Verification", "net_expected_savings"].fillna(0).sum()
        ),
        "latency_mean_ms": float(latency.mean()) if len(latency) else None,
        "latency_p95_ms": float(latency.quantile(0.95)) if len(latency) else None,
        "latency_max_ms": float(latency.max()) if len(latency) else None,
    })
    return df, summary


def pipeline_reliability():
    from src.database.duckdb_manager import dataframe
    batches = dataframe("SELECT status, COUNT(*) AS n FROM data_batches GROUP BY status")
    events = dataframe(
        """
        SELECT event_type, status, COUNT(*) AS n
        FROM system_events
        WHERE event_type IN ('DATA_INGESTION', 'MODEL_CANDIDATE', 'MODEL_PROMOTION', 'BATCH_SCORING')
        GROUP BY event_type, status
        """
    )
    rows = []
    if not batches.empty:
        total = int(batches["n"].sum())
        success = int(batches.loc[batches["status"].isin(["SUCCESS", "SEED"]), "n"].sum())
        rows.append({
            "Pipeline": "DataOps batches",
            "Runs": total,
            "Successful": success,
            "Success Rate": success / total if total else None,
        })
    for event_type, label in [
        ("MODEL_CANDIDATE", "Model training"),
        ("MODEL_PROMOTION", "Model promotion"),
        ("BATCH_SCORING", "Batch inference"),
    ]:
        x = events[events["event_type"] == event_type] if not events.empty else pd.DataFrame()
        if len(x):
            total = int(x["n"].sum())
            success = int(x.loc[x["status"] == "SUCCESS", "n"].sum())
            rows.append({
                "Pipeline": label,
                "Runs": total,
                "Successful": success,
                "Success Rate": success / total if total else None,
            })
    return pd.DataFrame(rows)


def _runtime_feature_frame(predictions):
    rows = []
    if predictions.empty or "feature_json" not in predictions:
        return pd.DataFrame()
    for raw in predictions["feature_json"].dropna():
        try:
            rows.append(json.loads(raw))
        except Exception:
            continue
    return pd.DataFrame(rows)


def feature_drift(reference: pd.DataFrame, predictions: pd.DataFrame):
    current = _runtime_feature_frame(predictions)
    if current.empty:
        return pd.DataFrame(columns=["Feature", "Metric", "Drift Score", "Status"])
    rows = []
    for feature in NUMERIC_FEATURES:
        if feature in reference.columns and feature in current.columns:
            value = population_stability_index(reference[feature], current[feature])
            rows.append({
                "Feature": feature,
                "Metric": "PSI",
                "Drift Score": value,
                "Status": drift_label(value, "psi"),
            })
    for feature in CATEGORICAL_FEATURES:
        if feature in reference.columns and feature in current.columns:
            value = categorical_total_variation(reference[feature], current[feature])
            rows.append({
                "Feature": feature,
                "Metric": "Total variation",
                "Drift Score": value,
                "Status": drift_label(value, "tv"),
            })
    return pd.DataFrame(rows)


def labeled_runtime_performance(model_version=None):
    """Performance on production-like predictions whose final outcomes are known."""
    from src.database.duckdb_manager import dataframe
    from src.models.evaluate import metrics as classification_metrics
    from src.business_evaluation import evaluate_business_policy

    where = "" if model_version is None else " AND model_version = ?"
    params = [] if model_version is None else [str(model_version)]
    try:
        df = dataframe(
            f"""
            SELECT probability, threshold, actual_outcome, scoring_mode
            FROM predictions
            WHERE actual_outcome IS NOT NULL{where}
            ORDER BY predicted_at
            """,
            params,
        )
    except Exception:
        return {"labeled_observations": 0}
    if df.empty:
        return {"labeled_observations": 0}
    df["monitoring_mode"] = df["scoring_mode"].map(normalize_scoring_mode)
    df = df[df["monitoring_mode"].isin(PRODUCTION_SCORING_MODES)].copy()
    mapping = {"canceled": 1, "cancelled": 1, "completed": 0, "complete": 0}
    y = df["actual_outcome"].astype(str).str.strip().str.lower().map(mapping)
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int)
    if len(df) < 2 or y.nunique() < 2:
        return {"labeled_observations": int(len(df))}
    probs = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0).to_numpy()
    threshold = float(pd.to_numeric(df["threshold"], errors="coerce").dropna().median()) if df["threshold"].notna().any() else 0.5
    out = classification_metrics(y.to_numpy(), probs, threshold)
    try:
        business = evaluate_business_policy(y.to_numpy(), probs)
        out["net_savings_per_1000_orders"] = business["net_savings_per_1000_orders"]
    except Exception:
        out["net_savings_per_1000_orders"] = None
    out["labeled_observations"] = int(len(df))
    return out
