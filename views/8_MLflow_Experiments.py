from pathlib import Path
import time

import mlflow
import numpy as np
import pandas as pd
import streamlit as st

from src.auth import require_role
from src.config import MLFLOW_TRACKING_URI
from src.data.io import read_snapshot
from src.database.duckdb_manager import dataframe
from src.models.train import run_manual_experiment

require_role("developer")
st.title("🧪 MLflow Experiments")
st.caption(
    "Run parameter experiments without changing production. Manual experiments log metrics and evaluation artifacts, "
    "but skip full model serialization for speed. Promotable models are created only on the ModelOps page."
)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_snapshot(path_text: str, modified_ns: int):
    """Cache immutable dataset versions between experiment runs."""
    _ = modified_ns  # included in the cache key so changed files invalidate automatically
    return read_snapshot(Path(path_text))


def _temporal_demo_sample(snapshot: pd.DataFrame, max_rows: int = 90_000) -> pd.DataFrame:
    """Deterministically thin the timeline for a responsive live experiment.

    We sample evenly across the complete time-ordered dataset instead of taking only
    the earliest/latest rows, then the normal temporal train/validation/test split is
    applied. Full-dataset ModelOps training remains unchanged.
    """
    if len(snapshot) <= max_rows:
        return snapshot.copy()
    ordered = snapshot.sort_values("created_at").reset_index(drop=True)
    idx = np.linspace(0, len(ordered) - 1, num=max_rows, dtype=np.int64)
    return ordered.iloc[idx].reset_index(drop=True)


versions = dataframe("SELECT dataset_version, processed_path, order_count FROM dataset_versions ORDER BY created_at DESC")
if not len(versions):
    st.warning("Create or initialize a dataset version first.")
    st.stop()

ver = st.selectbox("Dataset Version", versions["dataset_version"].tolist())
selected = versions[versions.dataset_version == ver].iloc[0]
name = st.text_input("Experiment Name", value="manual_lightgbm")

scope = st.radio(
    "Experiment scope",
    ["Fast demo", "Full dataset"],
    horizontal=True,
    help=(
        "Fast demo uses a deterministic 90,000-order sample spread across the full timeline. "
        "Full dataset uses every order in the selected version. Both are labeled in MLflow."
    ),
)
if scope == "Fast demo":
    st.info(
        "Fast demo mode is intended for live experimentation. It keeps the temporal structure but uses up to "
        "90,000 orders. Use **Full dataset** when you want metrics for formal reporting."
    )
else:
    st.caption(f"Full-dataset experiment: {int(selected.order_count):,} order-level records.")

c1, c2, c3 = st.columns(3)
n_estimators = c1.number_input("Trees", min_value=30, max_value=800, value=70, step=10)
learning_rate = c2.number_input("Learning rate", min_value=0.01, max_value=0.30, value=0.05, step=0.01, format="%.2f")
num_leaves = c3.number_input("Leaves", min_value=7, max_value=127, value=31, step=4)

if st.button("Run MLflow Experiment", type="primary", use_container_width=True):
    p = Path(selected.processed_path)
    if not p.exists():
        st.error(f"Dataset artifact not found: {p}")
        st.stop()

    # One stage is visible at a time. When a stage changes, the previous bar is
    # explicitly removed and a new stage bar appears in the same location.
    title_slot = st.empty()
    description_slot = st.empty()
    bar_slot = st.empty()
    completed_slot = st.empty()

    stage_state = {
        "key": None,
        "title": None,
        "started": None,
        "completed": [],
    }

    def _render_completed():
        if stage_state["completed"]:
            text = "  ·  ".join(
                f"✓ {title} ({duration:.1f}s)" for title, duration in stage_state["completed"][-6:]
            )
            completed_slot.caption(text)

    def update_stage(stage_key, fraction, title, description):
        now = time.perf_counter()
        if stage_key != stage_state["key"]:
            if stage_state["key"] is not None and stage_state["started"] is not None:
                stage_state["completed"].append(
                    (stage_state["title"], now - stage_state["started"])
                )
                _render_completed()

            # Remove the prior step before drawing the next one.
            title_slot.empty()
            description_slot.empty()
            bar_slot.empty()
            stage_state["key"] = stage_key
            stage_state["title"] = title
            stage_state["started"] = now

        pct = max(0, min(100, int(round(float(fraction) * 100))))
        title_slot.markdown(f"**{title}**")
        description_slot.caption(description)
        bar_slot.progress(pct, text=f"{pct}%")

    def finish_current_stage():
        if stage_state["key"] is not None and stage_state["started"] is not None:
            stage_state["completed"].append(
                (stage_state["title"], time.perf_counter() - stage_state["started"])
            )
            _render_completed()
        title_slot.empty()
        description_slot.empty()
        bar_slot.empty()
        stage_state["key"] = None
        stage_state["started"] = None

    started = time.perf_counter()
    try:
        update_stage("load", 0.05, "Load dataset version", "Reading the immutable order-level dataset artifact. Cached versions load almost instantly on later runs.")
        snapshot = _cached_snapshot(str(p), p.stat().st_mtime_ns)
        update_stage("load", 0.72, "Load dataset version", f"Loaded {len(snapshot):,} order-level records. Preparing the requested experiment scope.")

        source_rows = len(snapshot)
        if scope == "Fast demo":
            snapshot = _temporal_demo_sample(snapshot, max_rows=90_000)
        update_stage("load", 1.0, "Load dataset version", f"Experiment input contains {len(snapshot):,} orders from {source_rows:,} available records.")

        result = run_manual_experiment(
            snapshot,
            ver,
            name,
            stage_callback=update_stage,
            run_metadata={
                "experiment_scope": "fast_demo" if scope == "Fast demo" else "full_dataset",
                "source_dataset_rows": int(source_rows),
                "experiment_rows": int(len(snapshot)),
            },
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
        )
        finish_current_stage()
        elapsed = time.perf_counter() - started
        st.success(
            f"Experiment complete in **{elapsed:.1f} seconds**. MLflow run ID: `{result['run_id']}`"
        )
        if scope == "Fast demo":
            st.caption("This run is labeled `fast_demo` in MLflow. Use Full dataset for final report metrics.")
        st.json({"threshold": result["threshold"], "validation": result["val"], "test": result["test"]})
    except Exception as exc:
        finish_current_stage()
        st.error(f"Experiment failed: {exc}")
        st.exception(exc)

st.divider()
st.subheader("Recent Runtime MLflow Runs")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    exp = mlflow.get_experiment_by_name("cancellation-risk")
    if exp:
        runs = mlflow.search_runs([exp.experiment_id], max_results=50, order_by=["start_time DESC"])
        keep = [c for c in [
            "run_id", "tags.mlflow.runName", "params.run_source", "params.experiment_scope",
            "params.experiment_rows", "params.source_dataset_rows", "params.model_type", "params.dataset_version",
            "params.model_artifact_logged", "params.model__n_estimators", "params.model__learning_rate", "params.model__num_leaves",
            "metrics.val_roc_auc", "metrics.val_pr_auc", "metrics.val_brier",
            "metrics.test_roc_auc", "metrics.test_pr_auc", "metrics.test_brier",
            "metrics.test_recall_at_90_precision",
        ] if c in runs.columns]
        st.dataframe(runs[keep], use_container_width=True, hide_index=True)
    else:
        st.info("No runtime MLflow runs yet. The packaged seed production model is intentionally independent of runtime MLflow state.")
except Exception as exc:
    st.warning(f"MLflow history is unavailable: {exc}")
