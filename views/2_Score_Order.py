import io
import inspect
from datetime import datetime

import pandas as pd
import streamlit as st

from src.auth import require_role
from src.features.inference_features import prepare_order_features
from src.models.predict import record_decision, score_batch, score_order
from src.config import ARTIFACT_DIR
from src.models.registry import active_metadata
from src.ui import common as ui_common

# Import the stable helpers through the module instead of a direct symbol list.
# This keeps the page usable during rolling/partial upgrades where an older
# src/ui/common.py may still be present in the deployed checkout.
FEATURE_LABELS = ui_common.FEATURE_LABELS
business_policy_controls = ui_common.business_policy_controls
load_demo_orders = ui_common.load_demo_orders
load_reference = ui_common.load_reference


def load_historical_demo_orders():
    """Load the natural holdout queue with backward compatibility."""
    loader = getattr(ui_common, "load_historical_demo_orders", None)
    if callable(loader):
        return loader()

    historical_path = ARTIFACT_DIR / "historical_demo_orders.csv.gz"
    if historical_path.exists():
        return pd.read_csv(historical_path, low_memory=False, parse_dates=["created_at"])

    # Last-resort compatibility for very old deployments. This keeps the page
    # operational, while the caption below makes clear that the historical
    # queue is unavailable until the full upgrade is copied.
    return load_demo_orders()

require_role("manager")
st.title("🛒 Score Order")
st.caption("Use the packaged production model immediately. No DataOps or training step is required.")

meta = active_metadata()
reference = load_reference()
if not meta or reference.empty:
    st.error("Production inference artifacts are incomplete. Contact the developer team.")
    st.stop()

policy = business_policy_controls("score")
tabs = st.tabs(["Incoming Order Simulation", "Manual Order", "Batch CSV"])


def _score_order_compat(row, reference, policy, scoring_mode):
    """Call the newest scoring API while tolerating an older predict.py during upgrades."""
    kwargs = {"policy": policy}
    try:
        if "scoring_mode" in inspect.signature(score_order).parameters:
            kwargs["scoring_mode"] = scoring_mode
    except (TypeError, ValueError):
        pass
    return score_order(row, reference, **kwargs)


def show_score(sc, row):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cancellation Risk", f"{sc['probability']:.1%}")
    c2.metric("Risk Level", sc["risk_level"])
    c3.metric("Expected Avoidable Cost", f"Rs. {sc['expected_avoidable_cost']:,.0f}")
    c4.metric("Net Expected Savings", f"Rs. {sc['net_expected_savings']:,.0f}")
    if sc["recommendation"] == "Hold for Verification":
        st.warning(f"Recommended action: **{sc['recommendation']}**")
    else:
        st.success(f"Recommended action: **{sc['recommendation']}**")
    latency = sc.get("latency_ms")
    latency_text = f" Core model latency: {latency:.1f} ms." if latency is not None else ""
    st.caption(
        f"Technical risk threshold: {sc['threshold']:.1%}. Economic recommendation uses the configurable cost policy above."
        f"{latency_text}"
    )
    st.markdown("#### Key risk drivers")
    if sc["reasons"]:
        for feature, impact in sc["reasons"]:
            st.write(f"• **{FEATURE_LABELS.get(feature, feature)}** increased estimated cancellation risk by about {impact:.1%} versus the reference baseline.")
    else:
        st.write("No single feature materially increased risk versus the reference baseline.")


with tabs[0]:
    mode = st.radio(
        "Simulation mode",
        ["Live Operations Simulation", "Historical Evaluation"],
        horizontal=True,
        help="Live mode uses a risk-stratified demo queue and hides outcomes. Historical Evaluation uses a natural sample of the final temporal holdout and reveals the outcome only after a decision.",
    )

    if mode == "Live Operations Simulation":
        demo = load_demo_orders()
        st.info(
            "Live simulation is deliberately stratified across low, medium, and high model-risk orders so managers see a useful mix of decisions. "
            "It is a demo queue, not an estimate of the historical cancellation prevalence."
        )
        index_key = "live_queue_index"
        scoring_mode = "simulation_live"
    else:
        demo = load_historical_demo_orders()
        st.caption(
            "Historical Evaluation uses a natural sample from the final temporal holdout. Its cancellation prevalence is intentionally left unchanged."
        )
        index_key = "historical_queue_index"
        scoring_mode = "simulation_historical"

    if demo.empty:
        st.warning("The packaged simulation queue is unavailable.")
    else:
        if index_key not in st.session_state:
            st.session_state[index_key] = 0
        idx = st.session_state[index_key] % len(demo)
        row = demo.iloc[[idx]].copy()
        score_key = f"sim_score_{idx}_{mode}_{policy}"
        if st.session_state.get("sim_score_key") != score_key:
            st.session_state.sim_score_key = score_key
            st.session_state.current_score = _score_order_compat(row, reference, policy, scoring_mode)
            st.session_state.decision_made = False
        sc = st.session_state.current_score

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Order ID", str(row.iloc[0]["order_id"]))
        c2.metric("Order Value", f"Rs. {row.iloc[0]['grand_total']:,.0f}")
        c3.metric("Quantity", f"{row.iloc[0]['qty_ordered']:,.0f}")
        c4.metric("Payment", str(row.iloc[0]["payment_method"]))
        st.caption(f"Category: {row.iloc[0]['category_name_1']} • Discount: Rs. {row.iloc[0]['discount_amount']:,.0f} • Prior cancellation rate: {row.iloc[0]['customer_cancel_rate']:.1%}")
        show_score(sc, row)

        b1, b2 = st.columns(2)
        decision_payload = dict(sc)
        if mode == "Live Operations Simulation":
            decision_payload["actual_outcome"] = None
        if b1.button("✅ Approve for Fulfillment", use_container_width=True, disabled=st.session_state.get("decision_made", False)):
            record_decision(decision_payload, row.iloc[0]["order_id"], "Approve for Fulfillment")
            st.session_state.decision_made = True
            st.rerun()
        if b2.button("🟠 Hold for Verification", use_container_width=True, disabled=st.session_state.get("decision_made", False)):
            record_decision(decision_payload, row.iloc[0]["order_id"], "Hold for Verification")
            st.session_state.decision_made = True
            st.rerun()

        if st.session_state.get("decision_made"):
            if mode == "Historical Evaluation":
                st.info(f"Historical outcome: **{sc.get('actual_outcome', 'Unavailable')}**")
            else:
                st.info("Decision recorded. The outcome remains hidden in Live Operations Simulation mode.")
            if st.button("Next Incoming Order →", type="primary"):
                st.session_state[index_key] += 1
                st.session_state.current_score = None
                st.session_state.decision_made = False
                st.rerun()

with tabs[1]:
    st.subheader("Manual single-order scoring")
    categories = sorted(reference["category_name_1"].dropna().astype(str).unique().tolist())
    payments = sorted(reference["payment_method"].dropna().astype(str).unique().tolist())
    with st.form("manual_order"):
        a, b, c = st.columns(3)
        order_id = a.text_input("Order ID", value=f"manual_{datetime.now().strftime('%H%M%S')}")
        price = b.number_input("Average item price (Rs.)", min_value=0.0, value=1500.0, step=100.0)
        qty = c.number_input("Quantity", min_value=1.0, value=1.0, step=1.0)
        d, e, f = st.columns(3)
        grand_total = d.number_input("Grand total (Rs.)", min_value=0.0, value=1500.0, step=100.0)
        discount = e.number_input("Discount amount (Rs.)", min_value=0.0, value=0.0, step=50.0)
        prior = f.slider("Customer prior cancellation rate", 0.0, 1.0, 0.0, 0.05)
        g, h = st.columns(2)
        payment = g.selectbox("Payment method", payments)
        category = h.selectbox("Product category", categories)
        submit = st.form_submit_button("Score Order", type="primary", use_container_width=True)
    if submit:
        manual = prepare_order_features(pd.DataFrame([{
            "order_id": order_id,
            "created_at": datetime.now(),
            "price": price,
            "qty_ordered": qty,
            "grand_total": grand_total,
            "discount_amount": discount,
            "payment_method": payment,
            "category_name_1": category,
            "customer_cancel_rate": prior,
        }]))
        sc = _score_order_compat(manual, reference, policy, 'manual')
        show_score(sc, manual)

with tabs[2]:
    st.subheader("Batch scoring")
    st.caption("Upload one row per order. Outcome/status fields are not required and are ignored for inference.")
    template = pd.DataFrame([{
        "order_id": "NEW-1001", "created_at": "2026-09-29 14:30:00", "price": 1500,
        "qty_ordered": 1, "grand_total": 1500, "discount_amount": 0, "payment_method": "cod",
        "category_name_1": "Men's Fashion", "customer_cancel_rate": 0.10,
    }])
    st.download_button("Download batch template", template.to_csv(index=False).encode(), "prediction_batch_template.csv", "text/csv")
    batch_file = st.file_uploader("Upload order-level CSV", type=["csv"], key="prediction_batch")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            ready = prepare_order_features(batch)
            scored = score_batch(ready, policy=policy)
            show_cols = [
                "order_id", "cancellation_probability", "risk_level", "expected_avoidable_cost",
                "net_expected_savings", "recommendation",
            ]
            st.dataframe(scored[show_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download scored batch",
                scored.to_csv(index=False).encode(),
                "scored_orders.csv",
                "text/csv",
                type="primary",
            )
        except Exception as exc:
            st.error(str(exc))
