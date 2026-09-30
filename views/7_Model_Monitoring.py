from pathlib import Path

import pandas as pd
import streamlit as st

from src.auth import require_role
from src.business import load_policy
try:
    from src.business import evaluate_business_policy
except ImportError:
    # Compatibility with a partially upgraded deployment that still has the
    # earlier live-scoring-only src.business module.
    from src.business_evaluation import evaluate_business_policy
from src.config import ARTIFACT_DIR, ROOT
from src.models.registry import active_metadata
from src.models.train import list_candidates
# Import the monitoring module as a module instead of importing every symbol
# eagerly. This keeps the page compatible with a rolling/partial deployment in
# which an older metrics.py is still present. Newer helpers are resolved with
# getattr and get a local fallback instead of crashing at import time.
from src.monitoring import metrics as monitoring_metrics

MIN_MONITORING_OBSERVATIONS = getattr(monitoring_metrics, "MIN_MONITORING_OBSERVATIONS", 100)
drift_label = monitoring_metrics.drift_label
feature_drift = monitoring_metrics.feature_drift
pipeline_reliability = monitoring_metrics.pipeline_reliability
population_stability_index = monitoring_metrics.population_stability_index
runtime_prediction_summary = monitoring_metrics.runtime_prediction_summary
from src.ui import common as ui_common
from src.currency import fetch_pkr_to_usd_rate, format_usd, pkr_to_usd, rate_summary, usd_to_pkr

FEATURE_LABELS = getattr(ui_common, "FEATURE_LABELS", {
    "customer_cancel_rate": "Prior customer cancellation rate",
    "is_cod": "Cash-on-delivery payment",
    "grand_total": "Order value",
    "discount_ratio": "Discount ratio",
    "price": "Average item price",
    "qty_ordered": "Quantity",
    "payment_method": "Payment method",
    "category_name_1": "Product category",
    "discount_amount": "Discount amount",
    "hour": "Order hour",
    "day_of_week": "Day of week",
    "has_discount": "Discount presence",
})


def _empty_business_summary():
    return {
        "scoring_records": 0,
        "orders_scored": 0,
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


def _fallback_operations_business_summary(model_version=None):
    """Compatibility implementation for deployments with an older metrics.py."""
    try:
        from src.database.duckdb_manager import dataframe

        where = "" if model_version is None else " WHERE model_version = ?"
        params = [] if model_version is None else [str(model_version)]
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
    except Exception:
        return pd.DataFrame(), _empty_business_summary()

    if predictions.empty or "scoring_mode" not in predictions.columns:
        return pd.DataFrame(), _empty_business_summary()

    allowed = {"manual", "batch", "production_manual", "production_batch"}
    df = predictions.copy()
    df["monitoring_mode"] = (
        df["scoring_mode"].fillna("").astype(str).str.strip().str.lower()
    )
    df = df[df["monitoring_mode"].isin(allowed)].copy()
    if df.empty:
        return df, _empty_business_summary()

    scoring_records = len(df)
    df["predicted_at"] = pd.to_datetime(df.get("predicted_at"), errors="coerce")
    df = df.sort_values("predicted_at")
    if "order_id" in df.columns:
        order_key = df["order_id"].astype("string")
        if "prediction_id" in df.columns:
            order_key = order_key.fillna(df["prediction_id"].astype("string"))
        df = df.assign(_order_key=order_key).drop_duplicates("_order_key", keep="last")

    verified = df["recommendation"].eq("Hold for Verification")
    verified_df = df.loc[verified].copy()
    orders = int(len(df))
    verified_n = int(verified.sum())

    def _sum(column):
        if column not in verified_df.columns:
            return 0.0
        return float(pd.to_numeric(verified_df[column], errors="coerce").fillna(0.0).sum())

    probs = pd.to_numeric(
        verified_df.get("probability", pd.Series(index=verified_df.index, dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    expected_cancellations = float(probs.sum())
    cost_prevented = _sum("expected_avoidable_cost")
    verification_cost = _sum("intervention_cost")
    unnecessary_cost = _sum("expected_false_positive_cost")
    net_savings = _sum("net_expected_savings")

    summary = _empty_business_summary()
    summary.update({
        "scoring_records": int(scoring_records),
        "orders_scored": orders,
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
        "first_scored_at": df["predicted_at"].min(),
        "last_scored_at": df["predicted_at"].max(),
    })
    return df.drop(columns=["_order_key"], errors="ignore"), summary


operations_business_summary = getattr(
    monitoring_metrics, "operations_business_summary", _fallback_operations_business_summary
)


@st.cache_data(ttl=3600, show_spinner=False)
def _fallback_currency_context():
    return fetch_pkr_to_usd_rate()


def currency_caption():
    helper = getattr(ui_common, "currency_caption", None)
    if callable(helper):
        return helper()
    fx = _fallback_currency_context()
    if fx.get("is_live"):
        st.caption(rate_summary(fx) + ". Refreshed automatically up to once per hour. Model features remain in PKR internally; only user-facing money is shown in USD.")
    else:
        st.warning(rate_summary(fx) + ". Live FX lookup is unavailable, so the packaged fallback rate is being used.")
    return fx


def load_reference():
    helper = getattr(ui_common, "load_reference", None)
    if callable(helper):
        return helper()
    path = ARTIFACT_DIR / "reference_orders.csv.gz"
    return pd.read_csv(path, low_memory=False, parse_dates=["created_at"]) if path.exists() else pd.DataFrame()


def business_policy_controls(key_prefix="policy"):
    shared = getattr(ui_common, "business_policy_controls", None)
    shared_fx = getattr(ui_common, "get_currency_context", None)
    if callable(shared) and callable(shared_fx):
        return shared(key_prefix)

    defaults = load_policy()
    fx = _fallback_currency_context()
    rate = float(fx["rate"])
    with st.expander("Business assumptions for verification", expanded=False):
        st.caption("These assumptions do not change cancellation risk. They only decide whether verifying an order is expected to save money.")
        st.caption(rate_summary(fx))
        c1, c2 = st.columns(2)
        avoidable_usd = c1.number_input(
            "Loss if a canceled order reaches fulfillment ($)", min_value=0.0,
            value=round(float(pkr_to_usd(defaults["avoidable_fulfillment_cost"], rate)), 2), step=0.50,
            key=f"{key_prefix}_avoidable_usd",
        )
        prevention_pct = c2.slider(
            "Loss prevented by verification (%)", 0, 100,
            int(round(float(defaults["intervention_effectiveness"]) * 100)), 5,
            key=f"{key_prefix}_effectiveness_pct",
        )
        c3, c4 = st.columns(2)
        intervention_usd = c3.number_input(
            "Cost to verify one order ($)", min_value=0.0,
            value=round(float(pkr_to_usd(defaults["intervention_cost"], rate)), 2), step=0.25,
            key=f"{key_prefix}_intervention_usd",
        )
        unnecessary_usd = c4.number_input(
            "Extra cost if a good order is verified ($)", min_value=0.0,
            value=round(float(pkr_to_usd(defaults["false_positive_friction_cost"], rate)), 2), step=0.25,
            key=f"{key_prefix}_unnecessary_usd",
        )
    return {
        "avoidable_fulfillment_cost": float(usd_to_pkr(avoidable_usd, rate)),
        "intervention_effectiveness": prevention_pct / 100.0,
        "intervention_cost": float(usd_to_pkr(intervention_usd, rate)),
        "false_positive_friction_cost": float(usd_to_pkr(unnecessary_usd, rate)),
    }


require_role("developer")
st.title("📈 Model Monitoring")
st.caption(
    "Monitor production behavior, inference latency, pipeline reliability, drift, technical performance, and estimated business value."
)
fx = currency_caption()
fx_rate = float(fx["rate"])

meta = active_metadata()
if not meta:
    st.error("No production model metadata found.")
    st.stop()

m = meta.get("test_metrics", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Production Model", f"LightGBM {meta.get('model_version')}")
c2.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
c3.metric("PR-AUC", f"{m.get('pr_auc', float('nan')):.3f}")
c4.metric("Brier Score", f"{m.get('brier', float('nan')):.3f}")
c5.metric("Recall @ 90% Precision", f"{m.get('recall_at_90_precision', float('nan')):.1%}")
st.caption(
    f"Dataset {meta.get('dataset_version')} • technical threshold {float(meta.get('threshold', .5)):.1%} • "
    f"source {meta.get('source', 'packaged')}"
)

prev = meta.get("split_cancellation_prevalence", {})
if prev:
    st.info(
        f"Temporal cancellation prevalence changed from {prev.get('train', 0):.1%} in train to "
        f"{prev.get('validation', 0):.1%} in validation and {prev.get('test', 0):.1%} in test. "
        "This is distribution-shift context, so holdout metrics should be read alongside drift and calibration."
    )

eval_dir = Path(meta.get("evaluation_dir", "artifacts/production_evaluation"))
if not eval_dir.is_absolute():
    eval_dir = ROOT / eval_dir
predictions_path = eval_dir / "test_predictions.csv.gz"

runtime_tab, drift_tab, business_tab, evaluation_tab, candidate_tab = st.tabs([
    "Runtime Health",
    "Drift",
    "Business Impact",
    "Model Evaluation",
    "Candidate History",
])

MODE_LABELS = {
    "manual": "Manual orders",
    "batch": "Batch orders",
    "production": "Production/API",
    "production_manual": "Production manual",
    "production_batch": "Production batch",
    "api": "API orders",
    "simulation_live": "Live simulation",
    "simulation_historical": "Historical evaluation",
    "simulation": "Legacy simulation",
    "historical_evaluation": "Legacy historical evaluation",
    "single": "Legacy / unclassified single",
    "legacy_or_unknown": "Legacy / unknown",
}
PRODUCTION_MODES = {"manual", "batch", "production", "production_manual", "production_batch", "api"}


def _traffic_population_table(mode_counts):
    rows = []
    for mode, count in sorted(mode_counts.items(), key=lambda x: (-x[1], x[0])):
        rows.append({
            "Traffic type": MODE_LABELS.get(mode, mode),
            "Scoring mode": mode,
            "Predictions": int(count),
            "Used for production monitoring": "Yes" if mode in PRODUCTION_MODES else "No",
        })
    return pd.DataFrame(rows)


with runtime_tab:
    st.subheader("Production inference")
    runtime_predictions, runtime = runtime_prediction_summary(meta.get("model_version"))

    total = int(runtime.get("total_predictions", 0))
    eligible = int(runtime.get("eligible_predictions", 0))
    excluded = int(runtime.get("excluded_predictions", 0))
    if total == 0:
        st.info("No runtime predictions have been recorded yet. Score manual or batch orders in the Operations workspace to populate production monitoring.")
    else:
        st.markdown("#### Monitoring population")
        a, b, c = st.columns(3)
        a.metric("All Logged Predictions", f"{total:,}")
        b.metric("Production-Monitoring Eligible", f"{eligible:,}")
        c.metric("Demo / Evaluation Excluded", f"{excluded:,}")
        st.caption(
            "Only explicitly production-like traffic (manual, batch, or API scoring) contributes to drift, runtime risk, verification rate, latency, and live savings statistics. "
            "Live simulation and historical evaluation remain logged for auditability but are excluded from those calculations."
        )
        population_table = _traffic_population_table(runtime.get("mode_counts", {}))
        if not population_table.empty:
            st.dataframe(population_table, use_container_width=True, hide_index=True)

        if eligible == 0:
            st.info("No production-like observations are available yet. Simulation traffic is intentionally not used as a substitute for production monitoring data.")
        else:
            st.markdown("#### Eligible production traffic")
            a, b, c, d = st.columns(4)
            a.metric("Predictions", f"{runtime['predictions']:,}")
            b.metric("Average Risk", f"{runtime['average_risk']:.1%}")
            c.metric("High-Risk Rate", f"{runtime['high_risk_rate']:.1%}")
            d.metric("Verification Rate", f"{runtime['intervention_rate']:.1%}")
            e, f, g, h = st.columns(4)
            e.metric("Mean Model Latency", f"{runtime['latency_mean_ms']:.1f} ms" if runtime['latency_mean_ms'] is not None else "n/a")
            f.metric("P95 Model Latency", f"{runtime['latency_p95_ms']:.1f} ms" if runtime['latency_p95_ms'] is not None else "n/a")
            g.metric("Max Model Latency", f"{runtime['latency_max_ms']:.1f} ms" if runtime['latency_max_ms'] is not None else "n/a")
            h.metric("Estimated Net Savings", format_usd(runtime["expected_net_savings_total"], fx_rate))
            st.caption("Latency measures the core production model probability call. Risk explanations are calculated separately and do not inflate this value.")

            trend = runtime_predictions.sort_values("predicted_at").set_index("predicted_at")[["probability"]]
            if len(trend):
                st.markdown("#### Runtime cancellation-risk trend")
                st.line_chart(trend)

    st.markdown("#### Pipeline reliability")
    reliability = pipeline_reliability()
    if reliability.empty:
        st.caption("No pipeline executions have been recorded yet.")
    else:
        display = reliability.copy()
        display["Success Rate"] = display["Success Rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a")
        st.dataframe(display, use_container_width=True, hide_index=True)

with drift_tab:
    st.subheader("Prediction drift")
    runtime_predictions, runtime = runtime_prediction_summary(meta.get("model_version"))
    eligible = int(runtime.get("eligible_predictions", 0))
    excluded = int(runtime.get("excluded_predictions", 0))

    st.caption(
        f"Drift uses production-like traffic only. {excluded:,} simulation/evaluation/legacy observations are currently excluded from the monitoring population."
    )

    if not predictions_path.exists():
        st.warning("The packaged reference prediction distribution is unavailable for this model version.")
    elif eligible < MIN_MONITORING_OBSERVATIONS:
        historical = pd.read_csv(predictions_path)
        a, b, c = st.columns(3)
        a.metric("Reference Mean Risk", f"{historical['probability'].mean():.1%}")
        b.metric("Eligible Runtime Scores", f"{eligible:,}")
        c.metric("Minimum Required", f"{MIN_MONITORING_OBSERVATIONS:,}")
        st.info(
            f"Monitoring status: INSUFFICIENT RUNTIME DATA. At least {MIN_MONITORING_OBSERVATIONS} production-like observations are required before the app reports STABLE, WATCH, or DRIFT. "
            "Demo simulation and historical-evaluation scores do not count toward this threshold."
        )
    else:
        historical = pd.read_csv(predictions_path)
        psi = population_stability_index(historical["probability"], runtime_predictions["probability"])
        status = drift_label(psi, "psi")
        a, b, c = st.columns(3)
        a.metric("Reference Mean Risk", f"{historical['probability'].mean():.1%}")
        b.metric("Production Runtime Mean Risk", f"{runtime_predictions['probability'].mean():.1%}")
        c.metric("Prediction PSI", f"{psi:.3f}" if psi is not None else "Unavailable")
        if status == "STABLE":
            st.success("Prediction distribution status: STABLE")
        elif status == "WATCH":
            st.warning("Prediction distribution status: WATCH")
        elif status == "DRIFT":
            st.error("Prediction distribution status: DRIFT")
        else:
            st.info("Prediction drift could not be estimated from the available production observations.")
        st.caption("Project monitoring thresholds: PSI < 0.10 stable, 0.10–0.25 watch, > 0.25 drift.")

    st.markdown("#### Input feature drift")
    reference = load_reference()
    if reference.empty:
        st.caption("Packaged reference data is unavailable.")
    elif eligible < MIN_MONITORING_OBSERVATIONS:
        st.info(
            f"Feature drift is held until at least {MIN_MONITORING_OBSERVATIONS} production-like observations are available. Current eligible count: {eligible:,}."
        )
    else:
        fd = feature_drift(reference, runtime_predictions)
        if fd.empty or fd["Drift Score"].notna().sum() == 0:
            st.info("Production feature observations are unavailable or incomplete for drift estimation.")
        else:
            fd["Feature"] = fd["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            st.dataframe(fd, use_container_width=True, hide_index=True)
            st.caption("Numeric features use PSI. Categorical features use total-variation distance. These thresholds are monitoring heuristics for this prototype, not universal statistical cutoffs.")

with business_tab:
    st.subheader("Operations business impact")
    st.caption(
        "This tab uses only orders scored through the Operations Manager production workflow (manual and batch scoring). "
        "Live simulation, historical evaluation, and the model holdout are excluded."
    )

    operations_orders, live_impact = operations_business_summary(meta.get("model_version"))
    orders_scored = int(live_impact.get("orders_scored", 0))

    if orders_scored == 0:
        st.info(
            "No Operations Manager production orders have been scored yet. Use Manual Order or Batch Scoring in the Operations workspace to populate this tab. "
            "Simulation orders intentionally do not count toward business impact."
        )
        st.markdown("#### What will appear here")
        st.caption(
            "Once production-like orders are scored, this tab will accumulate expected verification volume, expected cancellations reached, expected cost prevented, "
            "verification cost, and expected net savings. Each order uses the business assumptions that were stored when it was scored."
        )
    else:
        if live_impact.get("scoring_records", orders_scored) > orders_scored:
            st.caption(
                f"{live_impact['scoring_records']:,} scoring events were logged for {orders_scored:,} unique orders. "
                "Only the latest production-like score per order is counted so rescoring does not double-count expected value."
            )
        else:
            st.caption(
                "Values below are expected, not realized, because newly scored orders do not yet have final outcomes. "
                "Underlying business calculations are stored in PKR and displayed using the current USD conversion rate."
            )

        a, b, c, d = st.columns(4)
        a.metric("Orders Scored", f"{orders_scored:,}")
        b.metric(
            "Orders Recommended for Verification",
            f"{live_impact['orders_verified']:,}",
            f"{live_impact['verification_rate']:.1%} of scored orders",
        )
        c.metric(
            "Expected Cancellations Reached",
            f"{live_impact['expected_cancellations_reached']:.1f}",
            help="Sum of predicted cancellation probabilities for orders recommended for verification. This is an expectation, not an observed outcome count.",
        )
        d.metric(
            "Expected Net Savings",
            format_usd(live_impact["expected_net_savings"], fx_rate),
        )

        e, f, g, h = st.columns(4)
        e.metric("Expected Cost Prevented", format_usd(live_impact["expected_cost_prevented"], fx_rate))
        f.metric("Verification Cost", format_usd(live_impact["verification_cost"], fx_rate))
        g.metric(
            "Expected Cost of Unnecessary Checks",
            format_usd(live_impact["expected_unnecessary_check_cost"], fx_rate),
        )
        h.metric(
            "Expected Net Savings / 1,000 Orders",
            format_usd(live_impact["net_savings_per_1000_orders"], fx_rate),
        )

        st.caption(
            "Expected cost prevented = predicted cancellation risk × loss if cancellation reaches fulfillment × preventable share, summed over orders recommended for verification. "
            "Expected net savings subtracts verification cost and the expected cost of unnecessary checks."
        )

        mode_counts = live_impact.get("mode_counts", {})
        if mode_counts:
            rows = []
            for mode, count in sorted(mode_counts.items(), key=lambda x: (-x[1], x[0])):
                rows.append({
                    "Operations scoring type": MODE_LABELS.get(mode, mode),
                    "Unique orders counted": int(count),
                })
            st.markdown("#### Operations scoring mix")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if live_impact.get("first_scored_at") is not None and live_impact.get("last_scored_at") is not None:
            st.caption(
                f"Business-impact window: {pd.Timestamp(live_impact['first_scored_at']).strftime('%Y-%m-%d %H:%M')} "
                f"to {pd.Timestamp(live_impact['last_scored_at']).strftime('%Y-%m-%d %H:%M')} (runtime database time)."
            )


with evaluation_tab:
    st.subheader("Packaged production evaluation")

    st.markdown("#### Historical business backtest")
    st.caption(
        "This is retrospective model evaluation on the final labeled test period. It is intentionally separate from live Operations Business Impact."
    )
    if not predictions_path.exists():
        st.warning("The production holdout probability artifact is unavailable for this model version.")
    else:
        holdout = pd.read_csv(predictions_path)
        policy = business_policy_controls("historical_backtest")
        impact = evaluate_business_policy(holdout["is_canceled"], holdout["probability"], policy)

        a, b, c, d = st.columns(4)
        a.metric("Historical Test Orders", f"{impact['orders']:,}")
        b.metric("Orders That Would Be Verified", f"{impact['interventions']:,}", f"{impact['intervention_rate']:.1%} of orders")
        c.metric("Actual Cancellations Reached", f"{impact['cancellations_intervened']:,}", f"{impact['cancellation_capture_rate']:.1%} of cancellations")
        d.metric("Unnecessary Verifications", f"{impact['false_interventions']:,}", f"{impact['false_intervention_rate']:.1%} of verifications")

        e, f, g, h = st.columns(4)
        e.metric("Estimated Cost Prevented", format_usd(impact["estimated_avoided_cost"], fx_rate))
        f.metric("Verification Cost", format_usd(impact["intervention_cost_total"], fx_rate))
        g.metric("Extra Cost of Unnecessary Checks", format_usd(impact["false_intervention_cost_total"], fx_rate))
        h.metric("Estimated Net Savings / 1,000 Orders", format_usd(impact["net_savings_per_1000_orders"], fx_rate))
        st.metric("Estimated Net Savings on Historical Test Orders", format_usd(impact["net_savings"], fx_rate))
        st.caption(
            "This backtest uses known historical outcomes, so it can count actual canceled/completed orders in the test period. Savings are still estimated because the source data does not contain the retailer's real warehouse or verification costs."
        )

        with st.expander("Simple historical what-if scenarios", expanded=False):
            st.caption(
                "Compare three assumption sets on the same historical test orders. This changes the business policy only; it does not retrain the model."
            )
            base = policy
            scenarios = [
                (
                    "Conservative",
                    {
                        "avoidable_fulfillment_cost": base["avoidable_fulfillment_cost"] * 0.75,
                        "intervention_effectiveness": max(0.05, base["intervention_effectiveness"] - 0.15),
                        "intervention_cost": base["intervention_cost"] * 1.25,
                        "false_positive_friction_cost": base["false_positive_friction_cost"] * 1.25,
                    },
                ),
                ("Current assumptions", dict(base)),
                (
                    "Favorable",
                    {
                        "avoidable_fulfillment_cost": base["avoidable_fulfillment_cost"] * 1.25,
                        "intervention_effectiveness": min(0.95, base["intervention_effectiveness"] + 0.15),
                        "intervention_cost": base["intervention_cost"] * 0.75,
                        "false_positive_friction_cost": base["false_positive_friction_cost"] * 0.75,
                    },
                ),
            ]
            rows = []
            for name, scenario_policy in scenarios:
                result = evaluate_business_policy(holdout["is_canceled"], holdout["probability"], scenario_policy)
                rows.append({
                    "Scenario": name,
                    "Loss per late cancellation": format_usd(scenario_policy["avoidable_fulfillment_cost"], fx_rate),
                    "Loss prevented by verification": f"{scenario_policy['intervention_effectiveness']:.0%}",
                    "Cost per verification": format_usd(scenario_policy["intervention_cost"], fx_rate),
                    "Extra cost if unnecessary": format_usd(scenario_policy["false_positive_friction_cost"], fx_rate),
                    "Orders verified": f"{result['intervention_rate']:.1%}",
                    "Cancellations reached": f"{result['cancellation_capture_rate']:.1%}",
                    "Net savings / 1,000 orders": format_usd(result["net_savings_per_1000_orders"], fx_rate),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    comparison_path = eval_dir / "baseline_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.markdown("#### Logistic Regression baseline vs LightGBM")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    for title, name in [
        ("ROC Curve", "test_roc.png"),
        ("Precision-Recall Curve", "test_pr.png"),
        ("Calibration Plot", "test_calibration.png"),
    ]:
        p = eval_dir / name
        if p.exists():
            st.markdown(f"#### {title}")
            st.image(str(p))

    fi = eval_dir / "feature_importance.csv"
    if fi.exists():
        st.markdown("#### LightGBM feature importance")
        x = pd.read_csv(fi).head(15).set_index("feature")
        st.bar_chart(x)

with candidate_tab:
    candidates = list_candidates()
    if candidates:
        rows = []
        for c in candidates:
            baseline = c.get("baseline") or {}
            rows.append({
                "candidate_id": c["candidate_id"],
                "created_at": c["created_at"],
                "dataset_version": c["dataset_version"],
                "threshold": c["threshold"],
                "baseline_roc_auc": baseline.get("test_metrics", {}).get("roc_auc"),
                "baseline_pr_auc": baseline.get("test_metrics", {}).get("pr_auc"),
                **{f"candidate_{k}": v for k, v in c.get("test_metrics", {}).items()},
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No runtime candidates have been trained yet.")
