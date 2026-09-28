import pandas as pd
import streamlit as st
import mlflow
from src.config import MLFLOW_TRACKING_URI
from src.database.duckdb_manager import dataframe
from src.models.train import run_manual_experiment

st.set_page_config(page_title="MLflow Experiments", page_icon="🧪", layout="wide")
st.title("🧪 MLflow Experiments")
st.caption("For now the project uses one model family: LightGBM. You can still create versioned experiments by changing its hyperparameters. Manual experiments never replace production automatically.")

versions = dataframe("SELECT dataset_version, processed_path FROM dataset_versions ORDER BY created_at DESC")
if not len(versions):
    st.warning("Create a dataset version first.")
    st.stop()

ver = st.selectbox("Dataset Version", versions["dataset_version"].tolist())
name = st.text_input("Experiment Name", value="manual_lightgbm")

c1, c2, c3 = st.columns(3)
n_estimators = c1.number_input("Trees (n_estimators)", min_value=50, max_value=800, value=220, step=25)
learning_rate = c2.number_input("Learning rate", min_value=0.01, max_value=0.30, value=0.05, step=0.01, format="%.2f")
num_leaves = c3.number_input("Num leaves", min_value=7, max_value=127, value=31, step=4)

if st.button("Run LightGBM Experiment", type="primary"):
    p = versions[versions.dataset_version == ver].iloc[0].processed_path
    progress_bar = st.progress(0, text="0% • Preparing experiment")
    stage_box = st.empty()

    def update_progress(percent, message):
        percent = max(0, min(100, int(percent)))
        progress_bar.progress(percent, text=f"{percent}% • {message}")
        stage_box.info(f"Current step: **{message}**")

    try:
        update_progress(5, "Loading dataset version")
        snapshot = pd.read_parquet(p)
        r = run_manual_experiment(
            snapshot, ver, name,
            progress_callback=update_progress,
            n_estimators=int(n_estimators), learning_rate=float(learning_rate), num_leaves=int(num_leaves),
        )
        progress_bar.progress(100, text="100% • Experiment complete")
        stage_box.success(f"Experiment logged. MLflow run ID: {r['run_id']}")
        st.json({"threshold": r["threshold"], "validation": r["val"], "test": r["test"]})
    except Exception as exc:
        progress_bar.progress(100, text="Stopped • Experiment failed")
        stage_box.error(f"Experiment failed: {exc}")
        st.exception(exc)

st.divider(); st.subheader("Recent MLflow Runs")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
exp = mlflow.get_experiment_by_name("cancellation-risk")
if exp:
    runs = mlflow.search_runs([exp.experiment_id], max_results=50, order_by=["start_time DESC"])
    keep = [c for c in [
        "run_id", "tags.mlflow.runName", "params.run_source", "params.dataset_version",
        "params.model__n_estimators", "params.model__learning_rate", "params.model__num_leaves",
        "metrics.val_roc_auc", "metrics.val_pr_auc", "metrics.val_brier",
        "metrics.test_roc_auc", "metrics.test_pr_auc", "metrics.test_brier",
    ] if c in runs.columns]
    st.dataframe(runs[keep], use_container_width=True, hide_index=True)
else:
    st.info("No MLflow runs yet.")
