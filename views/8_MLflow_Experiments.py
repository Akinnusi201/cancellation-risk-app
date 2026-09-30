from pathlib import Path
import pandas as pd

from src.data.io import read_snapshot
import streamlit as st
import mlflow

from src.auth import require_role
from src.config import MLFLOW_TRACKING_URI
from src.database.duckdb_manager import dataframe
from src.models.train import run_manual_experiment

require_role("developer")
st.title("🧪 MLflow Experiments")
st.caption("Run parameter experiments without changing production. Model promotion happens only on the ModelOps page.")

versions = dataframe("SELECT dataset_version, processed_path FROM dataset_versions ORDER BY created_at DESC")
if not len(versions):
    st.warning("Create or initialize a dataset version first.")
    st.stop()

ver = st.selectbox("Dataset Version", versions["dataset_version"].tolist())
name = st.text_input("Experiment Name", value="manual_lightgbm")
c1, c2, c3 = st.columns(3)
n_estimators = c1.number_input("Trees", min_value=50, max_value=800, value=220, step=25)
learning_rate = c2.number_input("Learning rate", min_value=0.01, max_value=0.30, value=0.05, step=0.01, format="%.2f")
num_leaves = c3.number_input("Leaves", min_value=7, max_value=127, value=31, step=4)

if st.button("Run MLflow Experiment", type="primary", use_container_width=True):
    p = Path(versions[versions.dataset_version == ver].iloc[0].processed_path)
    bar = st.progress(0, text="Preparing experiment...")
    stage = st.empty()

    def update_progress(percent, message):
        bar.progress(max(0, min(100, int(percent))), text=f"{int(percent)}% • {message}")
        stage.info(message)

    try:
        snapshot = read_snapshot(p)
        r = run_manual_experiment(
            snapshot, ver, name,
            progress_callback=update_progress,
            n_estimators=int(n_estimators), learning_rate=float(learning_rate), num_leaves=int(num_leaves),
        )
        stage.success(f"Experiment logged. MLflow run ID: {r['run_id']}")
        st.json({"threshold": r["threshold"], "validation": r["val"], "test": r["test"]})
    except Exception as exc:
        stage.error(f"Experiment failed: {exc}")
        st.exception(exc)

st.divider()
st.subheader("Recent Runtime MLflow Runs")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    exp = mlflow.get_experiment_by_name("cancellation-risk")
    if exp:
        runs = mlflow.search_runs([exp.experiment_id], max_results=50, order_by=["start_time DESC"])
        keep = [c for c in [
            "run_id", "tags.mlflow.runName", "params.run_source", "params.model_type", "params.dataset_version",
            "params.model__n_estimators", "params.model__learning_rate", "params.model__num_leaves",
            "metrics.val_roc_auc", "metrics.val_pr_auc", "metrics.val_brier",
            "metrics.test_roc_auc", "metrics.test_pr_auc", "metrics.test_brier",
            "metrics.test_recall_at_90_precision",
        ] if c in runs.columns]
        st.dataframe(runs[keep], use_container_width=True, hide_index=True)
    else:
        st.info("No runtime MLflow runs yet. The packaged seed production model is intentionally independent of runtime MLflow state.")
except Exception as exc:
    st.warning(f"MLflow history is unavailable: {exc}")
