from pathlib import Path

import pandas as pd
import streamlit as st

from src.auth import require_role
from src.data.io import read_snapshot
from src.database.duckdb_manager import dataframe
from src.models.registry import active_metadata
from src.models.train import list_candidates, promote_candidate, train_candidate

require_role("developer")
st.title("🤖 ModelOps")
st.caption(
    "Run a Logistic Regression baseline and a LightGBM candidate on the same temporal split. "
    "Production remains unchanged until explicit promotion."
)

versions = dataframe("SELECT dataset_version, processed_path, order_count FROM dataset_versions ORDER BY created_at DESC")
if not len(versions):
    st.warning("No dataset versions are available.")
    st.stop()

ver = st.selectbox("Training dataset version", versions["dataset_version"].tolist())
row = versions[versions.dataset_version == ver].iloc[0]
st.caption(f"{int(row.order_count):,} order-level records")

with st.expander("Experiment parameters", expanded=False):
    include_baseline = st.checkbox(
        "Train Logistic Regression baseline",
        value=True,
        help="The baseline uses the exact same temporal split and features as the LightGBM candidate.",
    )
    c1, c2, c3 = st.columns(3)
    n_estimators = c1.number_input("LightGBM trees", min_value=50, max_value=800, value=150, step=25)
    learning_rate = c2.number_input("Learning rate", min_value=0.01, max_value=0.30, value=0.05, step=0.01, format="%.2f")
    num_leaves = c3.number_input("Leaves", min_value=7, max_value=127, value=31, step=4)

if st.button("Run Experiment Set", type="primary", use_container_width=True):
    p = Path(row.processed_path)
    if not p.exists():
        st.error(f"Dataset artifact not found: {p}")
    else:
        bar = st.progress(0, text="Preparing experiment...")
        stage = st.empty()

        def update_progress(percent, message):
            bar.progress(max(0, min(100, int(percent))), text=f"{int(percent)}% • {message}")
            stage.info(message)

        try:
            snapshot = read_snapshot(p)
            cand = train_candidate(
                snapshot,
                ver,
                progress_callback=update_progress,
                include_baseline=include_baseline,
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                num_leaves=int(num_leaves),
            )
            stage.success(
                f"Experiment complete. Candidate **{cand['candidate_id']}** is ready. Production was not changed."
            )
            st.session_state.latest_candidate = cand["candidate_id"]
        except Exception as exc:
            stage.error(f"Experiment failed: {exc}")
            st.exception(exc)

st.divider()
st.subheader("Experiment Comparison")
candidates = list_candidates()
if not candidates:
    st.info("No runtime candidate models have been trained yet. The packaged production evaluation remains available under Model Monitoring.")
else:
    ids = [x["candidate_id"] for x in candidates]
    default_idx = ids.index(st.session_state.latest_candidate) if st.session_state.get("latest_candidate") in ids else 0
    selected_id = st.selectbox("Candidate", ids, index=default_idx)
    cand = next(x for x in candidates if x["candidate_id"] == selected_id)
    prod = active_metadata() or {}
    baseline = cand.get("baseline") or {}

    metrics = [
        "roc_auc",
        "pr_auc",
        "brier",
        "f1",
        "precision",
        "recall",
        "recall_at_80_precision",
        "recall_at_90_precision",
    ]
    compare = pd.DataFrame({
        "Metric": metrics,
        "Production": [prod.get("test_metrics", {}).get(m) for m in metrics],
        "Logistic Regression": [baseline.get("test_metrics", {}).get(m) for m in metrics],
        "LightGBM Candidate": [cand.get("test_metrics", {}).get(m) for m in metrics],
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

    if baseline:
        st.caption(
            f"Baseline run: {baseline.get('run_id', 'n/a')[:12]} • "
            f"Candidate run: {cand.get('run_id', 'n/a')[:12]} • Dataset: {cand['dataset_version']}"
        )
    else:
        st.caption(f"Candidate dataset: {cand['dataset_version']} • Candidate threshold: {cand['threshold']:.1%}")

    st.info(
        "Recall at fixed precision answers how many canceled orders the model can identify while enforcing a minimum precision. "
        "The 90% line is especially useful here because the final temporal holdout has cancellation prevalence above 80%."
    )
    st.warning("Promotion is an explicit governance action. Review the metrics and dataset version before replacing production.")

    confirm = st.checkbox(f"I reviewed {selected_id} and want to make it the production model.")
    if st.button("Promote Candidate to Production", disabled=not confirm, type="primary", use_container_width=True):
        try:
            active = promote_candidate(selected_id)
            st.success(
                f"Production updated to model **{active['model_version']}** using dataset **{active['dataset_version']}**."
            )
        except Exception as exc:
            st.error(f"Promotion failed: {exc}")
            st.exception(exc)
