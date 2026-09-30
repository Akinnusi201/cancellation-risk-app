from pathlib import Path

import streamlit as st

from src.auth import require_role
from src.config import DB_PATH, MLFLOW_TRACKING_URI, ROOT
from src.database.duckdb_manager import dataframe
from src.models.registry import PRODUCTION_MODEL_PATH, active_metadata, list_registered_models
from src.retraining import colab_enterprise_configuration

require_role("developer")
st.title("⚙️ System Status")
meta = active_metadata()
latest = dataframe("SELECT dataset_version, created_at FROM dataset_versions ORDER BY created_at DESC LIMIT 1")
last_event = dataframe("SELECT created_at,event_type,status,message FROM system_events ORDER BY created_at DESC LIMIT 10")

workflow_dir = ROOT / ".github" / "workflows"
workflow_files = []
if workflow_dir.exists():
    workflow_files = sorted(list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml")))
dockerfile = ROOT / "Dockerfile"
ci_template = ROOT / "GITHUB_ACTIONS_CI.yml"

registry_models = list_registered_models()
items = [
    ("Data Store", "Connected" if DB_PATH.exists() else "Ready on first use"),
    ("Production Model", (meta.get('model_display_name') or str(meta.get('model_name', 'Unknown')).replace('_', ' ').title()) if meta else "Missing"),
    ("Registered Models", str(len(registry_models))),
    ("Production Dataset", meta.get("dataset_version") if meta else (latest.iloc[0].dataset_version if len(latest) else "None")),
    ("Inference", "Ready" if meta and PRODUCTION_MODEL_PATH.exists() else "Blocked"),
]
cols = st.columns(len(items))
for c, (k, v) in zip(cols, items):
    c.metric(k, v)

st.success("App startup loads the packaged production artifact. It does not retrain the model.")
st.caption(f"Runtime MLflow backend: {MLFLOW_TRACKING_URI}")

st.subheader("DevOps Readiness")
a, b, c = st.columns(3)
a.metric("Automated CI", "Configured" if workflow_files else ("Packaged, not installed" if ci_template.exists() else "Missing"))
b.metric("Docker Image", "Configured" if dockerfile.exists() else "Missing")
c.metric("Packaged Inference Test", "Enabled" if (ROOT / "tests" / "test_inference_artifact.py").exists() else "Missing")
if workflow_files:
    st.caption("GitHub Actions workflow detected: " + ", ".join(p.name for p in workflow_files) + ". It runs tests, verifies packaged inference, then builds the Docker image.")
else:
    st.warning(
        "The GitHub Actions definition is packaged with the project, but it is not installed under `.github/workflows/` in this deployed checkout. GitHub only runs CI workflows from that directory."
    )
    st.code(
        "mkdir -p .github/workflows\n"
        "cp GITHUB_ACTIONS_CI.yml .github/workflows/ci.yml\n"
        "git add .github/workflows/ci.yml\n"
        "git commit -m \"Add GitHub Actions CI\"\n"
        "git push",
        language="bash",
    )
    if ci_template.exists():
        st.download_button(
            "Download GitHub Actions workflow",
            data=ci_template.read_bytes(),
            file_name="ci.yml",
            mime="text/yaml",
        )


st.subheader("Training & Automation")
auto_cfg = colab_enterprise_configuration()
a1, a2, a3 = st.columns(3)
a1.metric("Colab Notebook", "Packaged" if (ROOT / "notebooks" / "end_to_end_ml_workflow.ipynb").exists() else "Missing")
a2.metric("Starter Model Suite", f"{len(registry_models)} models" if registry_models else "Missing")
a3.metric("Colab Enterprise", "Configured" if auto_cfg.get("configured") else "Optional / not configured")
st.caption("Ordinary Colab is the default classroom training path. Colab Enterprise automation activates only when cloud settings and credentials are configured.")

st.subheader("Recent System Events")
if len(last_event):
    st.dataframe(last_event, use_container_width=True, hide_index=True)
else:
    st.caption("No runtime events yet.")
