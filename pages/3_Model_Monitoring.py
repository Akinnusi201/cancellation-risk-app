from pathlib import Path
import pandas as pd
import streamlit as st
import mlflow
from src.config import ARTIFACT_DIR, MLFLOW_TRACKING_URI
from src.models.registry import active_metadata

st.set_page_config(page_title="Model Monitoring", page_icon="🤖", layout="wide")
st.title("🤖 Model Monitoring")
st.caption("The prototype currently deploys one model family, LightGBM. Monitoring compares performance across versioned training runs instead of across model families.")

meta = active_metadata()
if not meta:
    st.warning("No active production model yet.")
    st.stop()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
run = mlflow.get_run(meta["run_id"])
m = run.data.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Production Model", f"LightGBM v{meta['model_version']}")
c2.metric("ROC-AUC", f"{m.get('test_roc_auc', float('nan')):.3f}")
c3.metric("PR-AUC", f"{m.get('test_pr_auc', float('nan')):.3f}")
c4.metric("Brier Score", f"{m.get('test_brier', float('nan')):.3f}")
st.caption(f"Dataset {meta['dataset_version']} • F1-selected validation threshold {meta['threshold']:.1%}")

exp = mlflow.get_experiment_by_name("cancellation-risk")
if exp:
    runs = mlflow.search_runs([exp.experiment_id], max_results=25, order_by=["start_time DESC"])
    auto = runs[runs.get("params.run_source", pd.Series(index=runs.index, dtype=str)).eq("automatic_batch")] if len(runs) else runs
    keep = [c for c in [
        "run_id", "start_time", "params.dataset_version", "params.model__n_estimators",
        "params.model__learning_rate", "params.model__num_leaves", "params.decision_threshold",
        "metrics.val_roc_auc", "metrics.val_pr_auc", "metrics.val_brier",
        "metrics.test_roc_auc", "metrics.test_pr_auc", "metrics.test_brier",
    ] if c in auto.columns]
    if len(auto):
        st.subheader("LightGBM Version History")
        st.dataframe(auto[keep], use_container_width=True, hide_index=True)

run_dir = ARTIFACT_DIR / meta["dataset_version"] / "lightgbm" / meta["run_id"][:8]
st.subheader("Evaluation Visualizations")
for title, name in [
    ("ROC Curve", "test_roc.png"),
    ("Precision-Recall Curve", "test_pr.png"),
    ("Calibration Plot", "test_calibration.png"),
]:
    p = run_dir / name
    if p.exists():
        st.markdown(f"#### {title}")
        st.image(str(p))

fi = run_dir / "feature_importance.csv"
if fi.exists():
    st.markdown("#### Feature Importance")
    x = pd.read_csv(fi).head(15).set_index("feature")
    st.bar_chart(x)
