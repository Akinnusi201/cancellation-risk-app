from pathlib import Path

import streamlit as st

from src.auth import require_role
from src.config import DB_PATH, MLFLOW_TRACKING_URI, ROOT
from src.database.duckdb_manager import dataframe
from src.models.registry import PRODUCTION_MODEL_PATH, active_metadata, list_registered_models
from src.models.benchmark import load_final_benchmark, load_final_benchmark_table
import src.retraining as retraining

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

st.success("App startup loads the packaged Production model. It does not retrain on startup.")
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
settings = retraining.load_retraining_settings()
a1, a2, a3, a4, a5 = st.columns(5)
a1.metric("Starter Model Suite", f"{len(registry_models)} models" if registry_models else "Missing")
a2.metric("Automatic Retraining", "Enabled" if settings.get("automatic_retraining_enabled", True) else "Manual")
a3.metric("Training Scope", "Quick sampled run" if settings.get("training_scope") in {"quick_sample", "fast_prototype"} else "Full dataset")
a4.metric("GPU Preference", "On" if settings.get("prefer_gpu_if_available", True) else "Off")
a5.metric("Deployment", "Manual approval")
st.caption(
    "Retraining runs directly in the application. New data or sustained degradation can trigger the five-model workflow automatically when enabled. "
    "Full-dataset training is the standard path; GPU preference applies only where supported. The selected model becomes Candidate only and still requires developer promotion to Production."
)
st.caption(
    "Business Impact demo inclusion: " + ("enabled" if settings.get("include_demo_business_impact", True) else "disabled") + ". Demo traffic is always excluded from technical drift monitoring."
)
if settings.get("training_scope") in {"quick_sample", "fast_prototype"}:
    st.caption(f"Quick-run limit: up to {int(settings.get('max_training_rows', 60000)):,} deterministic time-spanning orders per retraining run.")

requests = retraining.list_retraining_requests()
if requests:
    latest_request = requests[-1]
    st.caption(
        f"Latest retraining request: {latest_request.get('request_id')} · {latest_request.get('status')} · "
        f"{latest_request.get('dataset_version')}"
    )

st.subheader("Final Computational Evidence")
benchmark = load_final_benchmark()
benchmark_table = load_final_benchmark_table()
if benchmark:
    manifest = benchmark.get("split_manifest", {})
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Fair Five-Model Benchmark", "Ready")
    b2.metric("Benchmark Rows", f"{int(benchmark.get('source_rows', 0) or 0):,}")
    b3.metric("Split ID", manifest.get("split_id", "n/a"))
    b4.metric("Candidate Selection", "Validation only")
    st.caption(
        "All five benchmark models use the same dataset fingerprint, temporal split, feature schema, random-seed policy, and business assumptions. "
        "The test holdout is reserved for final unbiased evaluation and is not used to choose the Candidate."
    )
    if not benchmark_table.empty:
        st.caption(f"Benchmark includes {len(benchmark_table):,} model families and is available under Experiments & MLflow → Final Benchmark.")
else:
    st.warning("The report-ready full-data five-model benchmark has not been generated yet. Run it from Experiments & MLflow → Final Benchmark.")

try:
    import mlflow  # noqa: F401
    mlflow_state = "Installed"
except Exception:
    mlflow_state = "Dependency missing"
st.caption(f"MLflow runtime dependency: **{mlflow_state}** · configured backend: `{MLFLOW_TRACKING_URI}`")


st.subheader("Recent System Events")
if len(last_event):
    st.dataframe(last_event, use_container_width=True, hide_index=True)
else:
    st.caption("No runtime events yet.")
