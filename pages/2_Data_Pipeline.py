from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st
from src.data.ingestion import ingest_batch
from src.models.train import train_all
from src.database.duckdb_manager import dataframe

st.set_page_config(page_title="Automated Data Pipeline", page_icon="📦", layout="wide")
st.title("📦 Automated Data Pipeline")
st.caption("Upload a CSV batch once. The app validates, cleans, quarantines, versions, retrains one LightGBM model, logs it to MLflow, and promotes the new model automatically.")
up = st.file_uploader("Upload new CSV batch", type=["csv"])

if "last_upload_hash" not in st.session_state:
    st.session_state.last_upload_hash = None
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

if up is not None:
    raw = up.getvalue(); h = hashlib.sha256(raw).hexdigest()
    if h != st.session_state.last_upload_hash:
        st.session_state.last_upload_hash = h
        with st.status("Running automated DataOps pipeline...", expanded=True) as status:
            result = ingest_batch(raw, up.name)
            st.session_state.last_pipeline_result = result
            if result["status"] == "duplicate":
                st.warning(result["message"])
                status.update(label="Duplicate detected. Nothing changed.", state="complete")
            elif result["status"] == "failed":
                st.error(result["message"])
                status.update(label="Batch rejected", state="error")
            else:
                st.write("✓ Validation and cleaning complete")
                st.write(f"✓ Dataset version created: **{result['dataset_version']}**")
                st.write("Training LightGBM and logging the run to MLflow...")
                snap = pd.read_parquet(result["processed_path"])
                trained = train_all(snap, result["dataset_version"], promote=True, run_source="automatic_batch")
                result["trained"] = {
                    "model": "LightGBM",
                    "model_version": trained["model_version"],
                    "comparison": trained["comparison"],
                }
                st.session_state.last_pipeline_result = result
                st.write("✓ LightGBM training run logged to MLflow")
                st.write(f"✓ Production model promoted: **LightGBM v{trained['model_version']}**")
                status.update(label="Pipeline complete", state="complete")

result = st.session_state.last_pipeline_result
if result and result.get("status") == "success":
    a, b, c, d = st.columns(4)
    a.metric("Raw rows", result["raw_rows"]); b.metric("Valid rows", result["valid_rows"])
    c.metric("Quarantined", result["quarantined_rows"]); d.metric("Orders in snapshot", result["snapshot_orders"])
    st.subheader("Validation Results")
    st.dataframe(pd.DataFrame(result["checks"], columns=["Check", "Status", "Affected Rows", "Details"]), use_container_width=True, hide_index=True)
    if result.get("trained"):
        st.subheader("Current Training Run")
        st.dataframe(result["trained"]["comparison"], use_container_width=True, hide_index=True)

st.divider(); st.subheader("Dataset Version History")
versions = dataframe("SELECT dataset_version, created_at, row_count, order_count, processed_path, quarantine_path, report_path, active FROM dataset_versions ORDER BY created_at DESC")
if len(versions):
    st.dataframe(versions[[c for c in versions.columns if c not in ["processed_path", "quarantine_path", "report_path"]]], use_container_width=True, hide_index=True)
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
else:
    st.info("No dataset versions yet.")
