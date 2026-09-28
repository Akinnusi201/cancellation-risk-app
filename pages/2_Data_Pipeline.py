from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st

from src.data.ingestion import ingest_batch
from src.models.train import train_all
from src.database.duckdb_manager import dataframe

st.set_page_config(page_title="Automated Data Pipeline", page_icon="📦", layout="wide")
st.title("📦 Automated Data + Model Pipeline")
st.caption(
    "Choose a fast isolated demo sample or process the full dataset. The app validates, cleans, "
    "aggregates item rows to orders, versions the data, trains one LightGBM model, logs the run to "
    "MLflow, and activates the new model."
)

TOTAL_STEPS = 14
STEP_DEFINITIONS = [
    (1, "Fingerprint & duplicate check", "A SHA-256 fingerprint prevents the same batch from being ingested twice."),
    (2, "Load data / create demo sample", "The source file is item-level. Demo mode samples complete orders, not random rows, so baskets are not cut in half."),
    (3, "Validate data quality", "Schema, missing values, numeric ranges, dates, identifiers, and target values are checked before modeling. Outcome fields such as final status are kept out of the feature matrix to avoid leakage."),
    (4, "Check order consistency", "All item rows belonging to an order should agree on the final status. Full mode also excludes orders already stored in earlier batches."),
    (5, "Aggregate items to orders", "The notebook starts from transaction rows, but the prediction unit is one order. This step deterministically creates one row per order."),
    (6, "Version and store clean data", "The cleaned snapshot, quarantine file, and validation report are stored together so a training run can be reproduced later."),
    (7, "Load versioned training snapshot", "The model always trains from a named dataset version instead of an untracked in-memory dataframe."),
    (8, "Create temporal train/validation/test sets", "Your notebook used a time-based split to mimic predicting future cancellations. This app improves that idea with separate train, validation, and final test periods."),
    (9, "Build model features", "Features follow the notebook: price, quantity, grand total, discount, payment method, category, purchase timing, and customer history. Customer cancellation rate is computed from prior orders only."),
    (10, "Train LightGBM", "As in the notebook, LightGBM is used because boosted trees can capture nonlinear interactions among order value, payment, discount, category, and customer behavior."),
    (11, "Choose decision threshold", "The validation set selects the probability cutoff that maximizes F1, balancing precision and recall. The test set is not used to choose this threshold."),
    (12, "Evaluate held-out performance", "ROC-AUC measures ranking quality, PR-AUC focuses on identifying cancellations, and Brier score checks probability quality because the app uses predicted risk as a probability."),
    (13, "Create and log MLflow artifacts", "ROC, precision-recall, calibration, feature-importance artifacts, parameters, metrics, and the dataset version are logged to the experiment run."),
    (14, "Register and activate model", "The completed LightGBM pipeline is versioned in MLflow and becomes the active model used by the incoming-order screen."),
]
STEP_HELP = {n: (title, help_text) for n, title, help_text in STEP_DEFINITIONS}


def step_for_message(message: str) -> int:
    m = message.lower()
    rules = [
        (1, ["fingerprint", "duplicate upload", "reserving new batch", "starting automated"]),
        (2, ["saving raw", "reading csv", "sampling complete", "demo sample ready"]),
        (3, ["validating schema", "validation complete"]),
        (4, ["order consistency", "previously ingested", "isolated sample"]),
        (5, ["aggregating item", "order aggregation"]),
        (6, ["dataset version", "cumulative training snapshot", "saving cleaned", "dataset version stored"]),
        (7, ["loading versioned training snapshot", "preparing mlflow experiment"]),
        (8, ["temporal train", "train / validation / test"]),
        (9, ["feature matrices", "building model feature"]),
        (10, ["training lightgbm", "lightgbm training complete"]),
        (11, ["f1 threshold"]),
        (12, ["held-out test"]),
        (13, ["logging parameters", "evaluation artifacts", "logged to mlflow"]),
        (14, ["registering", "activating", "production model activated"]),
    ]
    for step, phrases in rules:
        if any(x in m for x in phrases):
            return step
    return 1

with st.expander("What happens in this pipeline?", expanded=False):
    for n, title, explanation in STEP_DEFINITIONS:
        st.markdown(f"**{n}. {title}**  \n{explanation}")

mode_col, sample_col = st.columns([2, 1])
with mode_col:
    mode_label = st.radio(
        "Run mode",
        ["⚡ Demo Sample", "🗄️ Full Dataset"],
        horizontal=True,
        help="Demo Sample trains on an isolated sample of complete orders for a fast presentation. Full Dataset uses cumulative historical data.",
    )
run_mode = "demo" if mode_label.startswith("⚡") else "full"

with sample_col:
    if run_mode == "demo":
        sample_orders = st.selectbox("Orders to sample", [2000, 5000, 10000, 20000], index=1)
    else:
        sample_orders = 5000
        st.metric("Training scope", "All valid orders")

if run_mode == "demo":
    st.info(
        f"**Fast demo mode:** up to **{sample_orders:,} complete orders** will be sampled with a fixed random seed. "
        "The sample is isolated from older full snapshots, and LightGBM uses fewer trees so the demo finishes much faster."
    )
else:
    st.warning(
        "**Full Dataset mode:** the pipeline builds the cumulative historical snapshot and retrains on all available valid orders. "
        "This can take several minutes on Streamlit Cloud."
    )

up = st.file_uploader("Upload new CSV batch", type=["csv"])

if "last_upload_key" not in st.session_state:
    st.session_state.last_upload_key = None
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

if up is not None:
    raw = up.getvalue()
    file_hash = hashlib.sha256(raw).hexdigest()
    upload_key = f"{file_hash}:{run_mode}:{sample_orders if run_mode == 'demo' else 'all'}"

    if upload_key != st.session_state.last_upload_key:
        st.session_state.last_upload_key = upload_key

        step_counter = st.empty()
        progress_bar = st.progress(0, text="Preparing pipeline...")
        stage_box = st.empty()
        explanation_box = st.empty()
        log_box = st.container(border=True)
        completed_steps = []
        last_step = {"value": 0}

        def update_progress(percent, message):
            percent = max(0, min(100, int(percent)))
            current_step = step_for_message(message)
            current_step = max(current_step, last_step["value"] or current_step)
            last_step["value"] = current_step
            title, help_text = STEP_HELP.get(current_step, (message, ""))

            step_counter.markdown(f"### Step {current_step} of {TOTAL_STEPS}: {title}")
            progress_bar.progress(percent, text=f"{percent}% • {message}")
            stage_box.info(f"**Running now:** {message}")
            explanation_box.caption(f"Why this matters: {help_text}")

            if not completed_steps or completed_steps[-1] != message:
                completed_steps.append(message)
                with log_box:
                    st.write(f"✓ {message}")

        try:
            update_progress(1, "Starting automated pipeline")
            result = ingest_batch(
                raw,
                up.name,
                progress_callback=update_progress,
                run_mode=run_mode,
                sample_orders=int(sample_orders),
            )
            st.session_state.last_pipeline_result = result

            if result["status"] == "duplicate":
                progress_bar.progress(100, text="100% • Duplicate upload detected")
                stage_box.warning(result["message"])
                st.toast("Duplicate upload detected. No retraining was triggered.", icon="⚠️")

            elif result["status"] == "failed":
                progress_bar.progress(100, text="Stopped • Batch rejected")
                stage_box.error(result["message"])
                st.toast("Batch validation failed.", icon="❌")

            else:
                update_progress(73, "Loading versioned training snapshot")
                snap = pd.read_parquet(result["processed_path"])
                trained = train_all(
                    snap,
                    result["dataset_version"],
                    promote=True,
                    run_source=f"automatic_{run_mode}",
                    progress_callback=update_progress,
                    fast_mode=(run_mode == "demo"),
                )
                result["trained"] = {
                    "model": "LightGBM",
                    "model_version": trained["model_version"],
                    "comparison": trained["comparison"],
                }
                st.session_state.last_pipeline_result = result
                step_counter.markdown(f"### Step {TOTAL_STEPS} of {TOTAL_STEPS}: Complete")
                progress_bar.progress(100, text="100% • Pipeline complete")
                stage_box.success(
                    f"Pipeline complete. Dataset **{result['dataset_version']}** is stored and "
                    f"LightGBM **v{trained['model_version']}** is active."
                )
                explanation_box.caption(
                    "The active model, its F1-selected threshold, dataset version, metrics, and artifacts are now linked through MLflow."
                )
                st.toast("Data pipeline and model retraining completed.", icon="✅")

        except Exception as exc:
            stage_box.error(f"Pipeline failed: {exc}")
            progress_bar.progress(100, text="Stopped • Pipeline error")
            st.exception(exc)

result = st.session_state.last_pipeline_result
if result and result.get("status") == "success":
    st.divider()
    st.subheader("Latest Pipeline Result")
    if result.get("run_mode") == "demo":
        st.caption(
            f"Demo sample processed from {result.get('source_rows', result['raw_rows']):,} source rows. "
            f"The model trained on an isolated sample snapshot."
        )
    a, b, c, d = st.columns(4)
    a.metric("Processed source rows", f"{result['raw_rows']:,}")
    b.metric("Valid item rows", f"{result['valid_rows']:,}")
    c.metric("Quarantined", f"{result['quarantined_rows']:,}")
    d.metric("Orders in training snapshot", f"{result['snapshot_orders']:,}")

    st.subheader("Validation Results")
    st.dataframe(
        pd.DataFrame(result["checks"], columns=["Check", "Status", "Affected Rows", "Details"]),
        use_container_width=True,
        hide_index=True,
    )
    if result.get("trained"):
        st.subheader("Current Training Run")
        st.dataframe(result["trained"]["comparison"], use_container_width=True, hide_index=True)

st.divider()
st.subheader("Dataset Version History")
versions = dataframe(
    "SELECT dataset_version, created_at, row_count, order_count, processed_path, quarantine_path, report_path, active "
    "FROM dataset_versions ORDER BY created_at DESC"
)
if len(versions):
    st.dataframe(
        versions[[c for c in versions.columns if c not in ["processed_path", "quarantine_path", "report_path"]]],
        use_container_width=True,
        hide_index=True,
    )
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
