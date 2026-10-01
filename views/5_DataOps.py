from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

from src.auth import require_role
from src.data.ingestion import ingest_batch
from src.database.duckdb_manager import dataframe
from src.config import ROOT
from src.models.registry import active_metadata
import src.retraining as retraining

require_role("developer")
st.title("📦 DataOps")
st.caption("Ingest → validate → quarantine → aggregate → version → optional retraining handoff. DataOps never changes Production directly.")
st.info("The Pakistan baseline is already packaged as an order-level dataset version. Upload only new batches here.")

STEPS = {
    1: ("Fingerprint & duplicate check", "Prevent exact batches from being ingested twice."),
    2: ("Read source batch", "Load the item-level CSV while preserving the immutable raw upload."),
    3: ("Validate data quality", "Check schema, dates, ranges, identifiers, target values, and duplicates."),
    4: ("Check order consistency", "Quarantine inconsistent statuses and previously ingested order IDs."),
    5: ("Aggregate to one row per order", "Create the prediction unit deterministically without outcome leakage in features."),
    6: ("Build cumulative dataset", "Append new orders and recompute prior-customer history across the full timeline."),
    7: ("Persist dataset version", "Store cleaned Parquet, quarantine CSV, validation report, and lineage metadata."),
}


def step_for_message(message):
    m = message.lower()
    if any(x in m for x in ["fingerprint", "duplicate upload", "reserving new batch"]): return 1
    if any(x in m for x in ["saving raw", "reading csv"]): return 2
    if any(x in m for x in ["validating", "validation complete"]): return 3
    if any(x in m for x in ["order consistency", "previously ingested"]): return 4
    if "aggregat" in m: return 5
    if any(x in m for x in ["cumulative", "next dataset version"]): return 6
    if any(x in m for x in ["saving cleaned", "dataset version stored"]): return 7
    return 1

with st.expander("DataOps controls and checks"):
    for n, (title, explanation) in STEPS.items():
        st.markdown(f"**{n}. {title}**  \n{explanation}")

sample_path = ROOT / "demo_batch.csv"
if sample_path.exists():
    st.download_button("Download safe demo batch", sample_path.read_bytes(), "demo_batch.csv", "text/csv")

up = st.file_uploader("Upload a new item-level CSV batch", type=["csv"])
if "last_upload_key" not in st.session_state:
    st.session_state.last_upload_key = None
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

if up is not None:
    raw = up.getvalue()
    upload_key = hashlib.sha256(raw).hexdigest()
    if upload_key != st.session_state.last_upload_key:
        st.session_state.last_upload_key = upload_key
        counter = st.empty()
        progress = st.progress(0, text="Preparing DataOps pipeline...")
        stage = st.empty()

        def update_progress(percent, message):
            step = step_for_message(message)
            title, explanation = STEPS[step]
            counter.markdown(f"### Step {step} of 7: {title}")
            # ingestion's old percentages include room for former training stages, so rescale 0-72 to 0-100
            scaled = min(100, int((max(0, min(72, percent)) / 72) * 100))
            progress.progress(scaled, text=f"{scaled}% • {message}")
            stage.info(f"**Running:** {message}  \n{explanation}")

        try:
            result = ingest_batch(raw, up.name, progress_callback=update_progress, run_mode="full")
            st.session_state.last_pipeline_result = result
            if result["status"] == "success":
                progress.progress(100, text="100% • Dataset version created")
                meta = active_metadata() or {}
                request = retraining.create_retraining_request(
                    meta.get("model_version", "unknown"),
                    result["dataset_version"],
                    "NEW_DATA_VERSION",
                    {"dataset_version": result["dataset_version"], "snapshot_orders": result.get("snapshot_orders")},
                    source="dataops",
                )
                settings = retraining.load_retraining_settings()
                if settings.get("automatic_retraining_enabled", True):
                    stage.success(
                        f"DataOps complete. **{result['dataset_version']}** is versioned. Automatic prototype retraining is enabled, so the five-model suite will run now. "
                        "Production remains unchanged during training."
                    )
                    train_title = st.empty()
                    train_bar = st.empty()
                    train_detail = st.empty()
                    current_training_stage = {"key": None}

                    def training_progress(stage_key, fraction, message):
                        if stage_key != current_training_stage["key"]:
                            train_bar.empty()
                            train_detail.empty()
                            current_training_stage["key"] = stage_key
                        train_title.markdown(f"### Retraining: {stage_key.replace('_', ' ').title()}")
                        train_bar.progress(max(0, min(100, int(float(fraction) * 100))), text=message)
                        train_detail.caption("All five models are tracked in MLflow. The best qualified model becomes Candidate only; deployment still requires developer approval.")

                    try:
                        request = retraining.maybe_run_automatic_retraining(request, progress=training_progress)
                        train_bar.empty()
                        train_detail.empty()
                        if request and request.get("status") == "CANDIDATE_READY":
                            train_title.success("Automatic retraining finished. A Candidate is ready for developer review in Model Registry & Deployment.")
                        else:
                            train_title.info("The dataset is ready, but no Candidate was selected. Review the retraining request in Model Registry & Deployment.")
                    except Exception as trigger_exc:
                        train_bar.empty()
                        train_detail.empty()
                        train_title.error(f"DataOps succeeded, but automatic retraining failed: {trigger_exc}")
                else:
                    stage.success(
                        f"DataOps complete. **{result['dataset_version']}** is versioned and retraining request **{request['request_id']}** is queued. "
                        "Automatic retraining is disabled in Model Registry settings, so Production remains unchanged."
                    )
                st.toast("Dataset version created and retraining request recorded.", icon="✅")
            elif result["status"] == "duplicate":
                progress.progress(100, text="Complete • No new data")
                stage.warning(result["message"])
            else:
                stage.error(result["message"])
        except Exception as exc:
            stage.error(f"DataOps failed: {exc}")
            st.exception(exc)

result = st.session_state.last_pipeline_result
if result and result.get("status") == "success":
    st.divider()
    st.subheader("Latest DataOps Result")
    a, b, c, d = st.columns(4)
    a.metric("Source rows", f"{result['raw_rows']:,}")
    b.metric("Valid item rows", f"{result['valid_rows']:,}")
    c.metric("Quarantined", f"{result['quarantined_rows']:,}")
    d.metric("Orders in snapshot", f"{result['snapshot_orders']:,}")
    st.dataframe(pd.DataFrame(result["checks"], columns=["Check", "Status", "Affected Rows", "Details"]), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Dataset Version History")
versions = dataframe(
    "SELECT dataset_version, created_at, row_count, order_count, processed_path, quarantine_path, report_path, active "
    "FROM dataset_versions ORDER BY created_at DESC"
)
if len(versions):
    st.dataframe(versions[["dataset_version", "created_at", "row_count", "order_count", "active"]], use_container_width=True, hide_index=True)
    chosen = st.selectbox("Download artifacts for version", versions["dataset_version"].tolist())
    r = versions[versions["dataset_version"] == chosen].iloc[0]
    cols = st.columns(3)
    for col, path_col, label, mime in [
        (cols[0], "processed_path", "Download cleaned Parquet", "application/octet-stream"),
        (cols[1], "quarantine_path", "Download quarantine CSV", "text/csv"),
        (cols[2], "report_path", "Download validation JSON", "application/json"),
    ]:
        p = Path(r[path_col])
        if p.exists():
            col.download_button(label, p.read_bytes(), file_name=p.name, mime=mime, use_container_width=True)
