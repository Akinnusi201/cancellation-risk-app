from pathlib import Path
import pandas as pd
import streamlit as st
from src.config import ARTIFACT_DIR, FEATURES
from src.models.registry import active_metadata
from src.models.predict import score_order, record_decision
from src.database.duckdb_manager import dataframe

st.set_page_config(page_title="Order Risk Monitor", page_icon="🛒", layout="wide")
st.title("🛒 Order Risk Monitor")
st.caption("Held-out historical orders are presented as simulated incoming orders.")
meta=active_metadata()
if not meta:
    st.warning("No production model is active yet. Upload a dataset on the Data Pipeline page first.")
    st.stop()

holdout=ARTIFACT_DIR/"latest_holdout.parquet"
if not holdout.exists():
    st.warning("No holdout queue is available yet.")
    st.stop()
df=pd.read_parquet(holdout).reset_index(drop=True)
if "queue_index" not in st.session_state: st.session_state.queue_index=0
if "current_score" not in st.session_state: st.session_state.current_score=None
if "decision_made" not in st.session_state: st.session_state.decision_made=False
idx=st.session_state.queue_index % len(df); row=df.iloc[[idx]].copy()

c1,c2,c3,c4=st.columns(4)
c1.metric("Order ID", str(row.iloc[0]["order_id"]))
c2.metric("Order Value", f"Rs. {row.iloc[0]['grand_total']:,.0f}")
c3.metric("Quantity", f"{row.iloc[0]['qty_ordered']:,.0f}")
c4.metric("Payment", str(row.iloc[0]["payment_method"]))
st.write(f"**Category:** {row.iloc[0]['category_name_1']}  |  **Discount:** Rs. {row.iloc[0]['discount_amount']:,.0f}  |  **Customer prior cancellation rate:** {row.iloc[0]['customer_cancel_rate']:.1%}")

if st.session_state.current_score is None:
    st.session_state.current_score=score_order(row,df)
sc=st.session_state.current_score
risk="HIGH" if sc["probability"]>=sc["threshold"] else "LOW"
col1,col2,col3=st.columns(3)
col1.metric("Cancellation Risk", f"{sc['probability']:.1%}")
col2.metric("Active Threshold", f"{sc['threshold']:.1%}")
col3.metric("Risk Level", risk)
st.subheader(f"System recommendation: {sc['recommendation']}")

st.markdown("#### Why this prediction?")
labels={"customer_cancel_rate":"Prior customer cancellation rate","is_cod":"Cash-on-delivery payment","grand_total":"Order value","discount_ratio":"Discount ratio","price":"Item price","qty_ordered":"Quantity","payment_method":"Payment method","category_name_1":"Product category","discount_amount":"Discount amount","hour":"Order hour","day_of_week":"Day of week","has_discount":"Discount presence"}
if sc["reasons"]:
    for f,impact in sc["reasons"]: st.write(f"• **{labels.get(f,f)}** increased estimated risk by about {impact:.1%} versus a typical baseline value.")
else: st.write("No single feature materially increased risk versus the baseline order.")

b1,b2=st.columns(2)
if b1.button("✅ Approve for Fulfillment", use_container_width=True, disabled=st.session_state.decision_made):
    record_decision(sc,row.iloc[0]["order_id"],"Approve for Fulfillment"); st.session_state.decision_made=True; st.rerun()
if b2.button("🟠 Hold for Verification", use_container_width=True, disabled=st.session_state.decision_made):
    record_decision(sc,row.iloc[0]["order_id"],"Hold for Verification"); st.session_state.decision_made=True; st.rerun()

if st.session_state.decision_made:
    st.success(f"Historical outcome revealed: **{sc['actual_outcome']}**")
    if st.button("Next Incoming Order →"):
        st.session_state.queue_index += 1; st.session_state.current_score=None; st.session_state.decision_made=False; st.rerun()

st.divider(); st.subheader("Decision History")
hist=dataframe("SELECT decided_at, order_id, probability, threshold, recommendation, manager_decision, actual_outcome, model_name, model_version FROM manager_decisions ORDER BY decided_at DESC LIMIT 50")
st.dataframe(hist,use_container_width=True,hide_index=True)
