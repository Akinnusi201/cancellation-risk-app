import streamlit as st

from src.auth import require_role
from src.business import load_policy
from src.database.duckdb_manager import dataframe
from src.models.registry import active_metadata
from src.ui import common as ui_common
from src.currency import fetch_pkr_to_usd_rate, format_usd, rate_summary

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


require_role("manager")
st.title("📊 Operations Dashboard")
st.caption("Production scoring is ready at login. Data preparation and model training are isolated from day-to-day operations.")
fx = currency_caption()
fx_rate = float(fx["rate"])

meta = active_metadata()
if not meta:
    st.error("The packaged production model is missing. Contact the developer team.")
    st.stop()

m = meta.get("test_metrics", {})
cols = st.columns(5)
production_name = meta.get('model_display_name') or str(meta.get('model_name', 'Unknown')).replace('_', ' ').title()
cols[0].metric("Production Model", production_name)
cols[1].metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
cols[2].metric("PR-AUC", f"{m.get('pr_auc', float('nan')):.3f}")
cols[3].metric("Brier Score", f"{m.get('brier', float('nan')):.3f}")
cols[4].metric("Risk Threshold", f"{float(meta.get('threshold', .5)):.1%}")
st.caption(f"Production dataset: **{meta.get('dataset_version', 'unknown')}**. The model is loaded from a packaged artifact and does not retrain when the app starts.")

policy = load_policy()
st.subheader("Business Rules Used for Recommendations")
a, b, c, d = st.columns(4)
a.metric("Loss from a late cancellation", format_usd(policy["avoidable_fulfillment_cost"], fx_rate))
b.metric("Loss prevented by verification", f"{policy['intervention_effectiveness']:.0%}")
c.metric("Cost to verify an order", format_usd(policy["intervention_cost"], fx_rate))
d.metric("Cost of an unnecessary verification", format_usd(policy["false_positive_friction_cost"], fx_rate))
prevented = policy['avoidable_fulfillment_cost'] * policy['intervention_effectiveness']
st.info(
    f"Simple rule: verify an order only when the expected money saved is greater than the expected cost of checking it. "
    f"Under the current assumptions, a successful verification can prevent about {format_usd(prevented, fx_rate)} of a cancellation-related loss."
)


st.subheader("Incoming Order Review Queue")
try:
    queue = dataframe(
        """
        SELECT po.created_at, po.order_id, po.status, po.customer_message,
               p.probability, p.recommendation, p.scoring_mode
        FROM prototype_orders po
        LEFT JOIN predictions p ON p.prediction_id = po.prediction_id
        ORDER BY po.created_at DESC
        LIMIT 100
        """
    )
except Exception:
    queue = None

if queue is None or queue.empty:
    st.info("No interactive orders are waiting yet. Manual single-order scoring and the optional Incoming Order Demo can both create orders for Operations review.")
else:
    pending = int((queue["status"] == "AWAITING_OPERATIONS_REVIEW").sum())
    released = int((queue["status"] == "RELEASED_TO_FULFILLMENT").sum())
    verify = int((queue["status"] == "VERIFICATION_REQUIRED").sum())
    q1, q2, q3 = st.columns(3)
    q1.metric("Awaiting Review", pending)
    q2.metric("Released to Fulfillment", released)
    q3.metric("Verification Required", verify)
    st.caption(
        "Order flow: order received → automated risk screening → Operations review → manager releases the order or keeps it for verification. Manual orders are production-like; simulated orders are clearly tagged as demo traffic."
    )
    pending_rows = queue[queue["status"] == "AWAITING_OPERATIONS_REVIEW"].copy()
    if not pending_rows.empty:
        st.markdown("#### Orders waiting for Operations")
        pending_rows = pending_rows[["created_at", "order_id", "probability", "recommendation", "scoring_mode"]].rename(columns={
            "created_at": "Placed at",
            "order_id": "Order ID",
            "probability": "Cancellation risk",
            "recommendation": "Model recommendation",
            "scoring_mode": "Source",
        })
        pending_rows["Source"] = pending_rows["Source"].replace({"manual": "Manual order", "simulation_live": "Demo simulation", "production_manual": "Production manual"})
        pending_rows["Cancellation risk"] = pending_rows["Cancellation risk"].map(lambda x: f"{x:.1%}" if x is not None else "")
        pending_rows["Model recommendation"] = pending_rows["Model recommendation"].replace({
            "Hold for Verification": "Keep for verification",
            "Approve for Fulfillment": "Release to fulfillment",
        })
        st.dataframe(pending_rows, use_container_width=True, hide_index=True)
        st.info("Open **Score Order** to review and decide on the incoming order. Manual orders and demo simulations use the same review workflow.")

st.subheader("Recent Decisions")
hist = dataframe(
    "SELECT decided_at, order_id, probability, recommendation, manager_decision, net_expected_savings "
    "FROM manager_decisions ORDER BY decided_at DESC LIMIT 10"
)
if len(hist):
    hist = hist.rename(columns={
        "decided_at": "Decision time",
        "order_id": "Order ID",
        "probability": "Cancellation risk",
        "recommendation": "System recommendation",
        "manager_decision": "Manager decision",
        "net_expected_savings": "Estimated net value ($)",
    })
    if "Cancellation risk" in hist.columns:
        hist["Cancellation risk"] = hist["Cancellation risk"].map(lambda x: f"{x:.1%}" if x is not None else "")
    if "Estimated net value ($)" in hist.columns:
        hist["Estimated net value ($)"] = hist["Estimated net value ($)"].map(lambda x: f"{'-' if float(x) * fx_rate < 0 else ''}${abs(float(x) * fx_rate):,.2f}" if x is not None else "")
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.caption("No manager decisions have been recorded yet. Open Score Order to begin.")
