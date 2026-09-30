import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.auth import require_role
from src.config import MLFLOW_TRACKING_URI
from src.models.registry import list_registered_models
from src.retraining import ordinary_colab_url

require_role("developer")
st.title("🧪 Experiments")
st.caption(
    "A simplified MLflow-style view of model runs. Training happens in Colab; this page is for tracking, comparing, and understanding results."
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "end_to_end_ml_workflow.ipynb"
models = list_registered_models()

runs_tab, compare_tab, tracking_tab = st.tabs(["Experiment Runs", "Compare Models", "MLflow Tracking"])

with runs_tab:
    st.subheader("Registered experiment runs")
    st.caption("Each row is one reproducible trained model. Production and Candidate are statuses, not separate model types.")
    if not models:
        st.info("No model runs are registered yet.")
    else:
        rows = []
        for m in models:
            tm = m.get("test_metrics", {})
            rows.append({
                "Run name": m.get("run_name", m.get("model_id")),
                "Model": m.get("display_name"),
                "Status": m.get("status", "READY"),
                "Dataset": m.get("dataset_version"),
                "Device": m.get("training_device"),
                "ROC-AUC": tm.get("roc_auc"),
                "PR-AUC": tm.get("pr_auc"),
                "Brier": tm.get("brier"),
                "Recall @ 90% precision": tm.get("recall_at_90_precision"),
                "Trained": m.get("trained_at"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        selected = st.selectbox("Run details", [m["model_id"] for m in models], format_func=lambda mid: next((m["display_name"] for m in models if m["model_id"] == mid), mid))
        record = next(m for m in models if m["model_id"] == selected)
        c1, c2, c3 = st.columns(3)
        c1.metric("Random seed", str(record.get("random_seed", 42)))
        c2.metric("Training rows", f"{int(record.get('training_rows', 0) or 0):,}")
        c3.metric("Duration", f"{record.get('duration_seconds'):.1f}s" if record.get("duration_seconds") else "Packaged production")
        st.caption(f"Experiment: `{record.get('experiment_name', 'n/a')}`")
        with st.expander("Reproducibility metadata"):
            st.json(record.get("reproducibility", {}))

    st.markdown("#### Run a new experiment")
    st.caption("Use the Colab notebook for heavy training. The Streamlit app stays responsive and never retrains on startup.")
    url = ordinary_colab_url()
    if url:
        st.link_button("Open End-to-End Training Notebook", url, use_container_width=True)
    elif NOTEBOOK.exists():
        st.download_button("Download End-to-End Training Notebook", NOTEBOOK.read_bytes(), file_name=NOTEBOOK.name, mime="application/x-ipynb+json", use_container_width=True)

with compare_tab:
    st.subheader("Model comparison")
    if len(models) < 2:
        st.info("At least two registered models are required for comparison.")
    else:
        names = {m["model_id"]: f"{m['display_name']} · {m.get('status','READY')}" for m in models}
        choices = st.multiselect("Choose models", list(names), default=list(names)[: min(5, len(names))], format_func=lambda x: names[x])
        metrics = [
            ("ROC-AUC", "roc_auc", True),
            ("PR-AUC", "pr_auc", True),
            ("Brier score", "brier", False),
            ("F1", "f1", True),
            ("Precision", "precision", True),
            ("Recall", "recall", True),
            ("Recall @ 90% precision", "recall_at_90_precision", True),
        ]
        rows = []
        for model_id in choices:
            m = next(x for x in models if x["model_id"] == model_id)
            row = {"Model": m["display_name"], "Status": m.get("status", "READY")}
            for label, key, _ in metrics:
                row[label] = m.get("test_metrics", {}).get(key)
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.info("Higher ROC-AUC, PR-AUC, F1, precision, recall, and fixed-precision recall are better. Lower Brier score means better probability calibration.")

with tracking_tab:
    st.subheader("MLflow tracking store")
    st.caption(f"Configured tracking URI: `{MLFLOW_TRACKING_URI}`")
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiments = mlflow.search_experiments(max_results=50)
        if not experiments:
            st.info("No runtime MLflow experiments are stored in this deployment yet. Packaged starter runs remain visible in the other tabs.")
        else:
            exp_rows = [{"Experiment": e.name, "Experiment ID": e.experiment_id, "Lifecycle": e.lifecycle_stage} for e in experiments]
            st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)
            selected_exp = st.selectbox("View MLflow experiment", experiments, format_func=lambda e: e.name)
            runs = mlflow.search_runs([selected_exp.experiment_id], max_results=100, order_by=["start_time DESC"])
            keep = [c for c in [
                "run_id", "tags.mlflow.runName", "params.model_family", "params.dataset_version", "params.training_device",
                "metrics.test_roc_auc", "metrics.test_pr_auc", "metrics.test_brier", "metrics.test_recall_at_90_precision",
                "metrics.business_net_savings_per_1000_orders",
            ] if c in runs.columns]
            st.dataframe(runs[keep], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"The MLflow tracking store is not available in this runtime: {exc}")
        st.caption("This does not affect production scoring. Colab stores full experiment history in the configured MLflow location, such as Google Drive.")
