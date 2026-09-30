from pathlib import Path

import pandas as pd
import streamlit as st

from src.auth import require_role
try:
    from src.business import evaluate_business_policy, sensitivity_analysis
except ImportError:
    # Compatibility with a partially upgraded deployment that still has the
    # earlier live-scoring-only src.business module.
    from src.business_evaluation import evaluate_business_policy, sensitivity_analysis
from src.config import ARTIFACT_DIR, ROOT
from src.models.registry import active_metadata
from src.models.train import list_candidates
from src.monitoring.metrics import (
    drift_label,
    feature_drift,
    pipeline_reliability,
    population_stability_index,
    runtime_prediction_summary,
)
from src.ui.common import FEATURE_LABELS, business_policy_controls, load_reference

require_role("developer")
st.title("📈 Model Monitoring")
st.caption(
    "Monitor production behavior, inference latency, pipeline reliability, drift, technical performance, and estimated business value."
)

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

with runtime_tab:
    st.subheader("Production inference")
    runtime_predictions, runtime = runtime_prediction_summary(meta.get("model_version"))
    if not runtime:
        st.info("No runtime single-order predictions have been recorded yet. Score orders in the Operations workspace to populate live monitoring.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Predictions", f"{runtime['predictions']:,}")
        b.metric("Average Risk", f"{runtime['average_risk']:.1%}")
        c.metric("High-Risk Rate", f"{runtime['high_risk_rate']:.1%}")
        d.metric("Intervention Rate", f"{runtime['intervention_rate']:.1%}")
        e, f, g, h = st.columns(4)
        e.metric("Mean Model Latency", f"{runtime['latency_mean_ms']:.1f} ms" if runtime['latency_mean_ms'] is not None else "n/a")
        f.metric("P95 Model Latency", f"{runtime['latency_p95_ms']:.1f} ms" if runtime['latency_p95_ms'] is not None else "n/a")
        g.metric("Max Model Latency", f"{runtime['latency_max_ms']:.1f} ms" if runtime['latency_max_ms'] is not None else "n/a")
        h.metric("Expected Net Savings", f"Rs. {runtime['expected_net_savings_total']:,.0f}")
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
    runtime_predictions, _ = runtime_prediction_summary(meta.get("model_version"))
    if predictions_path.exists() and len(runtime_predictions):
        historical = pd.read_csv(predictions_path)
        psi = population_stability_index(historical["probability"], runtime_predictions["probability"])
        status = drift_label(psi, "psi")
        a, b, c = st.columns(3)
        a.metric("Reference Mean Risk", f"{historical['probability'].mean():.1%}")
        b.metric("Runtime Mean Risk", f"{runtime_predictions['probability'].mean():.1%}")
        c.metric("Prediction PSI", f"{psi:.3f}" if psi is not None else "Need ≥20 runtime scores")
        if status == "STABLE":
            st.success("Prediction distribution status: STABLE")
        elif status == "WATCH":
            st.warning("Prediction distribution status: WATCH")
        elif status == "DRIFT":
            st.error("Prediction distribution status: DRIFT")
        else:
            st.info("At least 20 runtime predictions are required before PSI is reported.")
        st.caption("Project monitoring thresholds: PSI < 0.10 stable, 0.10–0.25 watch, > 0.25 drift.")
    else:
        st.info("Runtime prediction drift becomes available after Operations has scored orders.")

    st.markdown("#### Input feature drift")
    reference = load_reference()
    if reference.empty:
        st.caption("Packaged reference data is unavailable.")
    elif runtime_predictions.empty:
        st.caption("No runtime feature observations are available yet.")
    else:
        fd = feature_drift(reference, runtime_predictions)
        if fd.empty or fd["Drift Score"].notna().sum() == 0:
            st.info("At least 20 logged runtime orders are required for stable feature-drift estimates.")
        else:
            fd["Feature"] = fd["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            st.dataframe(fd, use_container_width=True, hide_index=True)
            st.caption("Numeric features use PSI. Categorical features use total-variation distance. These thresholds are monitoring heuristics for this prototype, not universal statistical cutoffs.")

with business_tab:
    st.subheader("Historical business-policy evaluation")
    if not predictions_path.exists():
        st.warning("The production holdout probability artifact is unavailable for this model version.")
    else:
        holdout = pd.read_csv(predictions_path)
        policy = business_policy_controls("monitoring")
        impact = evaluate_business_policy(holdout["is_canceled"], holdout["probability"], policy)
        a, b, c, d = st.columns(4)
        a.metric("Holdout Orders", f"{impact['orders']:,}")
        b.metric("Orders Intervened", f"{impact['interventions']:,}", f"{impact['intervention_rate']:.1%}")
        c.metric("Canceled Orders Captured", f"{impact['cancellations_intervened']:,}", f"{impact['cancellation_capture_rate']:.1%}")
        d.metric("False Interventions", f"{impact['false_interventions']:,}", f"{impact['false_intervention_rate']:.1%} of interventions")
        e, f, g, h = st.columns(4)
        e.metric("Estimated Avoided Cost", f"Rs. {impact['estimated_avoided_cost']:,.0f}")
        f.metric("Intervention Cost", f"Rs. {impact['intervention_cost_total']:,.0f}")
        g.metric("False-Intervention Cost", f"Rs. {impact['false_intervention_cost_total']:,.0f}")
        h.metric("Net Savings / 1,000 Orders", f"Rs. {impact['net_savings_per_1000_orders']:,.0f}")
        st.metric("Estimated Net Savings on Holdout", f"Rs. {impact['net_savings']:,.0f}")
        st.caption(
            "This is a transparent counterfactual estimate, not observed warehouse savings. The dataset does not contain internal fulfillment or intervention costs, so the configured assumptions drive the economic result."
        )

        st.markdown("#### Sensitivity analysis")
        defaults = policy
        fulfillment_costs = sorted({
            max(0.0, defaults["avoidable_fulfillment_cost"] * 0.5),
            defaults["avoidable_fulfillment_cost"],
            defaults["avoidable_fulfillment_cost"] * 1.5,
        })
        effectiveness_values = sorted({
            max(0.05, defaults["intervention_effectiveness"] - 0.20),
            defaults["intervention_effectiveness"],
            min(0.95, defaults["intervention_effectiveness"] + 0.20),
        })
        intervention_costs = sorted({
            max(0.0, defaults["intervention_cost"] * 0.5),
            defaults["intervention_cost"],
            defaults["intervention_cost"] * 1.5,
        })
        sensitivity = sensitivity_analysis(
            holdout["is_canceled"],
            holdout["probability"],
            fulfillment_costs,
            effectiveness_values,
            intervention_costs,
            defaults["false_positive_friction_cost"],
        )
        sensitivity = sensitivity.sort_values(
            ["avoidable_fulfillment_cost", "intervention_effectiveness", "intervention_cost"]
        )
        display = sensitivity.copy()
        display["intervention_effectiveness"] = display["intervention_effectiveness"].map(lambda x: f"{x:.0%}")
        display["intervention_rate"] = display["intervention_rate"].map(lambda x: f"{x:.1%}")
        display["cancellation_capture_rate"] = display["cancellation_capture_rate"].map(lambda x: f"{x:.1%}")
        display["net_savings"] = display["net_savings"].map(lambda x: f"Rs. {x:,.0f}")
        display["net_savings_per_1000_orders"] = display["net_savings_per_1000_orders"].map(lambda x: f"Rs. {x:,.0f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

with evaluation_tab:
    st.subheader("Packaged production evaluation")
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
