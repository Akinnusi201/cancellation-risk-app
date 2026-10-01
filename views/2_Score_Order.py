import io
import inspect
from datetime import datetime

import pandas as pd
import streamlit as st

from src.auth import require_role
from src.business import load_policy
from src.features.inference_features import prepare_order_features
from src.models.predict import record_decision, score_batch, score_order
from src.models import predict as predict_module
from src.config import ARTIFACT_DIR
from src.models.registry import active_metadata
from src.ui import common as ui_common
from src.currency import fetch_pkr_to_usd_rate, format_usd, pkr_to_usd, rate_summary, usd_to_pkr

# Import stable helpers through the module, but never assume that optional USD
# helpers exist. Streamlit Cloud can briefly serve a mixed checkout during an
# upgrade, so this page owns safe fallbacks for every new presentation helper.
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


def _load_csv_artifact(filename):
    path = ARTIFACT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, parse_dates=["created_at"])


def load_reference():
    helper = getattr(ui_common, "load_reference", None)
    return helper() if callable(helper) else _load_csv_artifact("reference_orders.csv.gz")


def load_demo_orders():
    helper = getattr(ui_common, "load_demo_orders", None)
    return helper() if callable(helper) else _load_csv_artifact("demo_orders.csv.gz")


def business_policy_controls(key_prefix="policy"):
    # Use the shared USD control only when the deployed common module is new
    # enough to include its currency context. Otherwise render the same controls
    # locally so an older common.py cannot break or revert the page to PKR.
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
        prevented = avoidable_usd * (prevention_pct / 100.0)
        st.info(
            f"If the order would cancel, verification can prevent about **${prevented:,.2f}** of loss. "
            f"Every verification costs **${intervention_usd:,.2f}**, and an unnecessary verification adds **${unnecessary_usd:,.2f}**."
        )
    return {
        "avoidable_fulfillment_cost": float(usd_to_pkr(avoidable_usd, rate)),
        "intervention_effectiveness": prevention_pct / 100.0,
        "intervention_cost": float(usd_to_pkr(intervention_usd, rate)),
        "false_positive_friction_cost": float(usd_to_pkr(unnecessary_usd, rate)),
    }


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

def get_order_status(prediction_id):
    loader = getattr(predict_module, "operations_order_status", None) or getattr(predict_module, "prototype_order_status", None)
    if callable(loader):
        try:
            return loader(prediction_id)
        except Exception:
            return None
    return None


def show_customer_order_status(sc, decision_made=False):
    """Render the customer-facing state for an interactive incoming order."""
    state = get_order_status(sc.get("prediction_id")) or {}
    status = state.get("status")
    message = state.get("customer_message")

    st.markdown("### Customer order status")
    if status == "RELEASED_TO_FULFILLMENT":
        st.success(message or "Order approved. Your order has been released to fulfillment.")
        st.caption("Flow: Order placed ✓  →  Risk screening ✓  →  Operations review ✓  →  Released to fulfillment ✓")
    elif status == "VERIFICATION_REQUIRED":
        st.warning(message or "Additional verification is required before fulfillment.")
        st.caption("Flow: Order placed ✓  →  Risk screening ✓  →  Operations review ✓  →  Verification required")
    else:
        st.info(message or "Order received. Please wait while our operations team completes a short verification review.")
        st.caption("Flow: Order placed ✓  →  Risk screening ✓  →  Awaiting Operations review …")

require_role("manager")
st.title("🛒 Score Order")
st.caption("Use the packaged production model immediately. No DataOps or training step is required.")
fx = currency_caption()
fx_rate = float(fx["rate"])

meta = active_metadata()
reference = load_reference()
if not meta or reference.empty:
    st.error("Production inference artifacts are incomplete. Contact the developer team.")
    st.stop()

policy = business_policy_controls("score")
tabs = st.tabs(["Incoming Order Demo", "Manual Order", "Batch CSV"])
st.caption("**Manual Order** and **Batch CSV** are production-like scoring paths. **Incoming Order Demo** is optional and exists to demonstrate the customer-to-Operations review flow.")


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
    verify = sc["recommendation"] == "Hold for Verification"
    action = "Verify before fulfillment" if verify else "Release to fulfillment"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cancellation Risk", f"{sc['probability']:.1%}")
    c2.metric("Risk Level", sc["risk_level"])
    c3.metric("Expected Money Saved", format_usd(sc["expected_avoidable_cost"], fx_rate))
    c4.metric("Expected Net Savings", format_usd(sc["net_expected_savings"], fx_rate))

    if verify:
        st.warning(f"Recommended action: **{action}**")
    else:
        st.success(f"Recommended action: **{action}**")

    policy_used = sc.get("policy", {})
    verify_cost = float(policy_used.get("intervention_cost", 0.0))
    unnecessary_cost = float(policy_used.get("false_positive_friction_cost", 0.0))
    st.caption(
        f"Why: the app compares the expected cost prevented with the expected cost of verification. "
        f"A verification costs {format_usd(verify_cost, fx_rate)}; if the order would have completed normally, the model also allows {format_usd(unnecessary_cost, fx_rate)} for unnecessary delay/service effort."
    )

    latency = sc.get("latency_ms")
    if latency is not None:
        st.caption(f"Model scoring latency: {latency:.1f} ms. The {sc['threshold']:.1%} technical threshold labels risk as High/Low; it does not by itself decide the business action.")

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
            "Demo customer flow: a simulated customer places an order, the model screens it, and the customer temporarily sees **Awaiting Operations Review**. "
            "The Operations Manager then releases the order or keeps it for verification. Demo traffic may appear in Business Impact for presentation purposes but remains excluded from drift monitoring."
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
        # Streamlit may preserve some session keys across a hot redeploy while
        # dropping others. Never assume ``current_score`` exists just because
        # ``sim_score_key`` survived from an earlier rerun/version.
        policy_key = tuple(sorted((str(k), float(v)) for k, v in policy.items()))
        score_key = f"sim_score_{idx}_{mode}_{policy_key}"
        cached_score = st.session_state.get("current_score")
        needs_score = (
            st.session_state.get("sim_score_key") != score_key
            or not isinstance(cached_score, dict)
            or "probability" not in cached_score
        )
        if needs_score:
            st.session_state["sim_score_key"] = score_key
            st.session_state["current_score"] = _score_order_compat(
                row, reference, policy, scoring_mode
            )
            st.session_state["decision_made"] = False

        sc = st.session_state.get("current_score")
        if not isinstance(sc, dict):
            st.error("The order score could not be initialized. Please refresh the page and try again.")
            st.stop()

        if mode == "Live Operations Simulation":
            show_customer_order_status(sc, st.session_state.get("decision_made", False))
            st.markdown("### Operations review")
            st.caption("The order has reached the Operations queue. Review the model signal and choose what happens next.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Order ID", str(row.iloc[0]["order_id"]))
        c2.metric("Order Value", format_usd(row.iloc[0]["grand_total"], fx_rate))
        c3.metric("Quantity", f"{row.iloc[0]['qty_ordered']:,.0f}")
        c4.metric("Payment", str(row.iloc[0]["payment_method"]))
        st.caption(f"Category: {row.iloc[0]['category_name_1']} • Discount: {format_usd(row.iloc[0]['discount_amount'], fx_rate)} • Prior cancellation rate: {row.iloc[0]['customer_cancel_rate']:.1%}")
        show_score(sc, row)

        b1, b2 = st.columns(2)
        decision_payload = dict(sc)
        if mode == "Live Operations Simulation":
            decision_payload["actual_outcome"] = None
        if b1.button("✅ Release to Fulfillment", use_container_width=True, disabled=st.session_state.get("decision_made", False)):
            record_decision(decision_payload, row.iloc[0]["order_id"], "Approve for Fulfillment")
            st.session_state.decision_made = True
            st.rerun()
        if b2.button("🟠 Keep for Verification", use_container_width=True, disabled=st.session_state.get("decision_made", False)):
            record_decision(decision_payload, row.iloc[0]["order_id"], "Hold for Verification")
            st.session_state.decision_made = True
            st.rerun()

        if st.session_state.get("decision_made"):
            if mode == "Historical Evaluation":
                st.info(f"Historical outcome: **{sc.get('actual_outcome', 'Unavailable')}**")
            else:
                status = get_order_status(sc.get("prediction_id")) or {}
                if status.get("status") == "RELEASED_TO_FULFILLMENT":
                    st.success("Operations decision recorded: the customer order is released to fulfillment.")
                else:
                    st.warning("Operations decision recorded: the customer order remains on hold for verification.")
                st.caption("The true historical outcome remains hidden in Live Operations Simulation mode.")
            next_label = "Place Next Simulated Customer Order →" if mode == "Live Operations Simulation" else "Next Historical Order →"
            if st.button(next_label, type="primary"):
                st.session_state[index_key] += 1
                st.session_state["current_score"] = None
                st.session_state["decision_made"] = False
                st.rerun()

with tabs[1]:
    st.subheader("Manual single-order scoring")
    categories = sorted(reference["category_name_1"].dropna().astype(str).unique().tolist())
    payments = sorted(reference["payment_method"].dropna().astype(str).unique().tolist())
    with st.form("manual_order"):
        a, b, c = st.columns(3)
        order_id = a.text_input("Order ID", value=f"manual_{datetime.now().strftime('%H%M%S')}")
        default_order_usd = round(float(pkr_to_usd(1500.0, fx_rate)), 2)
        price_usd = b.number_input("Average item price ($)", min_value=0.0, value=default_order_usd, step=0.50)
        qty = c.number_input("Quantity", min_value=1.0, value=1.0, step=1.0)
        d, e, f = st.columns(3)
        grand_total_usd = d.number_input("Grand total ($)", min_value=0.0, value=default_order_usd, step=0.50)
        discount_usd = e.number_input("Discount amount ($)", min_value=0.0, value=0.0, step=0.25)
        prior = f.slider("Customer prior cancellation rate", 0.0, 1.0, 0.0, 0.05)
        g, h = st.columns(2)
        payment = g.selectbox("Payment method", payments)
        category = h.selectbox("Product category", categories)
        submit = st.form_submit_button("Score Order", type="primary", use_container_width=True)
    if submit:
        manual = prepare_order_features(pd.DataFrame([{
            "order_id": order_id,
            "created_at": datetime.now(),
            "price": float(usd_to_pkr(price_usd, fx_rate)),
            "qty_ordered": qty,
            "grand_total": float(usd_to_pkr(grand_total_usd, fx_rate)),
            "discount_amount": float(usd_to_pkr(discount_usd, fx_rate)),
            "payment_method": payment,
            "category_name_1": category,
            "customer_cancel_rate": prior,
        }]))
        st.session_state["manual_order_row"] = manual
        st.session_state["manual_order_score"] = _score_order_compat(manual, reference, policy, "manual")
        st.session_state["manual_decision_made"] = False

    manual = st.session_state.get("manual_order_row")
    sc = st.session_state.get("manual_order_score")
    if isinstance(sc, dict) and isinstance(manual, pd.DataFrame) and not manual.empty:
        show_customer_order_status(sc, st.session_state.get("manual_decision_made", False))
        show_score(sc, manual)
        st.markdown("#### Operations decision")
        st.caption("The order stays in review until you release it or keep it for verification.")
        m1, m2 = st.columns(2)
        if m1.button("✅ Release Manual Order", use_container_width=True, disabled=st.session_state.get("manual_decision_made", False)):
            record_decision(sc, manual.iloc[0]["order_id"], "Approve for Fulfillment")
            st.session_state["manual_decision_made"] = True
            st.rerun()
        if m2.button("🟠 Keep Manual Order for Verification", use_container_width=True, disabled=st.session_state.get("manual_decision_made", False)):
            record_decision(sc, manual.iloc[0]["order_id"], "Hold for Verification")
            st.session_state["manual_decision_made"] = True
            st.rerun()
        if st.session_state.get("manual_decision_made"):
            state = get_order_status(sc.get("prediction_id")) or {}
            if state.get("status") == "RELEASED_TO_FULFILLMENT":
                st.success("Decision recorded. The order has been released to fulfillment.")
            elif state.get("status") == "VERIFICATION_REQUIRED":
                st.warning("Decision recorded. The order remains on hold for verification.")

with tabs[2]:
    st.subheader("Batch scoring")
    st.caption("Upload one row per order. Outcome/status fields are not required and are ignored for inference. Dollar inputs are converted to PKR internally before model scoring.")
    template = pd.DataFrame([{
        "order_id": "NEW-1001", "created_at": "2026-09-29 14:30:00", "price": round(float(pkr_to_usd(1500, fx_rate)), 2),
        "qty_ordered": 1, "grand_total": round(float(pkr_to_usd(1500, fx_rate)), 2), "discount_amount": 0, "payment_method": "cod",
        "category_name_1": "Men's Fashion", "customer_cancel_rate": 0.10,
    }])
    st.download_button("Download USD batch template", template.to_csv(index=False).encode(), "prediction_batch_template_usd.csv", "text/csv")
    batch_currency = st.selectbox("Currency used by price, grand_total, and discount_amount", ["USD", "PKR"], index=0, help="Choose USD for the new template. PKR keeps compatibility with older source-format files.")
    batch_file = st.file_uploader("Upload order-level CSV", type=["csv"], key="prediction_batch")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            if batch_currency == "USD":
                for money_col in ["price", "grand_total", "discount_amount"]:
                    if money_col in batch.columns:
                        batch[money_col] = pd.to_numeric(batch[money_col], errors="coerce") / fx_rate
            ready = prepare_order_features(batch)
            scored = score_batch(ready, policy=policy)
            show_cols = [
                "order_id", "cancellation_probability", "risk_level", "expected_avoidable_cost",
                "net_expected_savings", "recommendation",
            ]
            display = scored[show_cols].copy().rename(columns={
                "order_id": "Order ID",
                "cancellation_probability": "Cancellation risk",
                "risk_level": "Risk level",
                "expected_avoidable_cost": "Expected money saved ($)",
                "net_expected_savings": "Expected net savings ($)",
                "recommendation": "Recommended action",
            })
            display["Cancellation risk"] = display["Cancellation risk"].map(lambda x: f"{x:.1%}")
            display["Expected money saved ($)"] = display["Expected money saved ($)"].map(lambda x: format_usd(x, fx_rate))
            display["Expected net savings ($)"] = display["Expected net savings ($)"].map(lambda x: format_usd(x, fx_rate))
            display["Recommended action"] = display["Recommended action"].replace({
                "Hold for Verification": "Verify before fulfillment",
                "Approve for Fulfillment": "Release to fulfillment",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)
            export = scored.copy()
            for money_col in ["price", "grand_total", "discount_amount", "expected_avoidable_cost", "expected_false_positive_cost", "net_expected_savings"]:
                if money_col in export.columns:
                    export[money_col] = export[money_col].map(lambda x: float(pkr_to_usd(x, fx_rate)) if pd.notna(x) else x)
            export = export.rename(columns={
                "price": "price_usd",
                "grand_total": "grand_total_usd",
                "discount_amount": "discount_amount_usd",
                "expected_avoidable_cost": "expected_money_saved_usd",
                "expected_false_positive_cost": "expected_unnecessary_check_cost_usd",
                "net_expected_savings": "expected_net_savings_usd",
            })
            export["fx_pkr_to_usd"] = fx_rate
            st.download_button(
                "Download scored batch (USD)",
                export.to_csv(index=False).encode(),
                "scored_orders_usd.csv",
                "text/csv",
                type="primary",
            )
        except Exception as exc:
            st.error(str(exc))
