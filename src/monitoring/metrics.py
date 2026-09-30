import json

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


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


def runtime_prediction_summary(model_version=None):
    from src.database.duckdb_manager import dataframe

    where = "" if model_version is None else " WHERE model_version = ?"
    params = [] if model_version is None else [str(model_version)]
    # New deployments include telemetry columns. During a rolling upgrade, an
    # older runtime database/module may not have them yet, so fall back to the
    # core prediction fields instead of crashing the monitoring page.
    try:
        df = dataframe(
            f"""
            SELECT predicted_at, probability, threshold, recommendation, latency_ms,
                   net_expected_savings, scoring_mode, feature_json
            FROM predictions{where}
            ORDER BY predicted_at DESC
            """,
            params,
        )
    except Exception:
        df = dataframe(
            f"""
            SELECT predicted_at, probability, threshold, recommendation, net_expected_savings
            FROM predictions{where}
            ORDER BY predicted_at DESC
            """,
            params,
        )
        for col in ["latency_ms", "scoring_mode", "feature_json"]:
            df[col] = np.nan

    if df.empty:
        return df, {}
    latency = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    summary = {
        "predictions": len(df),
        "average_risk": float(df["probability"].mean()),
        "high_risk_rate": float((df["probability"] >= df["threshold"]).mean()),
        "intervention_rate": float((df["recommendation"] == "Hold for Verification").mean()),
        "expected_net_savings_total": float(df.loc[df["recommendation"] == "Hold for Verification", "net_expected_savings"].fillna(0).sum()),
        "latency_mean_ms": float(latency.mean()) if len(latency) else None,
        "latency_p95_ms": float(latency.quantile(0.95)) if len(latency) else None,
        "latency_max_ms": float(latency.max()) if len(latency) else None,
    }
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
