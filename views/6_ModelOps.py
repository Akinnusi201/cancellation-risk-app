from pathlib import Path

import pandas as pd
import streamlit as st

from src.auth import require_role
from src.currency import fetch_pkr_to_usd_rate, format_usd
from src.models.registry import (
    active_metadata,
    candidate_metadata,
    import_candidate_package,
    list_registered_models,
    promote_registered_model,
    set_candidate,
)
import src.retraining as retraining

require_role("developer")
st.title("🤖 Model Registry & Deployment")
st.caption(
    "Compare trained models, retrain the full five-model suite, review the selected Candidate, and control production deployment. "
    "For this prototype, retraining can run directly inside the app; deployment always remains a separate developer approval."
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "end_to_end_ml_workflow.ipynb"


def plain_health(model):
    m = model.get("test_metrics", {})
    if not m:
        return "No metrics"
    if m.get("roc_auc", 0) >= 0.90 and m.get("brier", 1) <= 0.12:
        return "Strong"
    if m.get("roc_auc", 0) >= 0.82 and m.get("brier", 1) <= 0.20:
        return "Good"
    return "Needs review"


def run_training_with_ui(request=None, dataset_version=None):
    stage_slot = st.empty()
    progress_slot = st.empty()
    detail_slot = st.empty()
    completed_slot = st.empty()
    current = {"stage": None}
    completed = []

    def progress(stage_key, fraction, message):
        if stage_key != current["stage"]:
            if current["stage"] is not None:
                completed.append(current["stage"].replace("_", " ").title())
            progress_slot.empty()
            detail_slot.empty()
            current["stage"] = stage_key
        stage_slot.markdown(f"#### {stage_key.replace('_', ' ').title()}")
        progress_slot.progress(max(0, min(100, int(float(fraction) * 100))), text=message)
        detail_slot.caption(
            "The prototype is training and evaluating one reproducible stage at a time. "
            "Production remains available and will not change during this run."
        )
        if completed:
            completed_slot.caption("Completed: " + " · ".join(completed[-5:]))

    try:
        result = retraining.run_local_retraining(
            request=request,
            dataset_version=dataset_version,
            progress=progress,
        )
        progress_slot.empty()
        detail_slot.empty()
        stage_slot.success("Retraining complete. The best qualified model is registered as Candidate; Production is unchanged.")
        return result
    except Exception as exc:
        progress_slot.empty()
        detail_slot.empty()
        stage_slot.error(f"Retraining failed: {exc}")
        st.exception(exc)
        return None


models = list_registered_models()
active = active_metadata() or {}
candidate = candidate_metadata()
fx = fetch_pkr_to_usd_rate()

c1, c2, c3 = st.columns(3)
c1.metric("Production", active.get("model_display_name") or str(active.get("model_name", "Unknown")).replace("_", " ").title())
c2.metric("Production Version", str(active.get("model_version", "n/a")))
c3.metric("Candidate", candidate.get("model_id") if candidate else "None")

st.markdown("### Model registry")
st.caption(
    "**Production** is scoring orders now. **Candidate** is the proposed replacement. "
    "**Ready** models are trained comparison models. Retraining never changes Production automatically."
)
if not models:
    st.warning("No registered models are packaged yet.")
else:
    rows = []
    for model in models:
        tm = model.get("test_metrics", {})
        bm = model.get("business_metrics", {})
        rows.append({
            "Model": model.get("display_name"),
            "Status": model.get("status", "READY"),
            "Health": plain_health(model),
            "ROC-AUC": tm.get("roc_auc"),
            "PR-AUC": tm.get("pr_auc"),
            "Brier": tm.get("brier"),
            "Recall @ 90% precision": tm.get("recall_at_90_precision"),
            "Est. savings / 1,000": format_usd(bm.get("net_savings_per_1000_orders", 0.0), float(fx["rate"])) if bm.get("net_savings_per_1000_orders") is not None else "n/a",
            "Training": model.get("training_device", "CPU"),
            "Dataset": model.get("dataset_version"),
            "Model ID": model.get("model_id"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected_id = st.selectbox(
        "Inspect a model",
        [m["model_id"] for m in models],
        format_func=lambda mid: next((f"{m['display_name']} · {m.get('status','READY')}" for m in models if m["model_id"] == mid), mid),
    )
    selected = next(m for m in models if m["model_id"] == selected_id)
    a, b, c, d = st.columns(4)
    tm = selected.get("test_metrics", {})
    a.metric("ROC-AUC", f"{tm.get('roc_auc', float('nan')):.3f}")
    b.metric("PR-AUC", f"{tm.get('pr_auc', float('nan')):.3f}")
    c.metric("Brier", f"{tm.get('brier', float('nan')):.3f}")
    d.metric("Recall @ 90% precision", f"{tm.get('recall_at_90_precision', float('nan')):.1%}")
    st.caption(
        f"Experiment: `{selected.get('experiment_name', 'packaged starter')}` · Run: `{selected.get('run_name', selected_id)}` · "
        f"Seed: {selected.get('random_seed', 42)} · Scope: {selected.get('training_scope', 'packaged / formal workflow')}"
    )

    if selected.get("status") != "PRODUCTION":
        if st.button("Mark selected model as Candidate", use_container_width=True):
            set_candidate(selected_id, "Developer selected model from registry")
            st.success(f"{selected.get('display_name')} is now the Candidate. Production has not changed.")
            st.rerun()

st.divider()
st.markdown("### Candidate review and deployment")
candidate = candidate_metadata()
if not candidate:
    st.info("There is no active Candidate. Retraining can select one automatically, or you can mark a Ready model as Candidate above.")
else:
    candidate_model = next((m for m in list_registered_models() if m.get("model_id") == candidate.get("model_id")), None)
    if not candidate_model:
        st.warning("Candidate metadata exists, but its model artifact is not registered.")
    else:
        st.success(f"Candidate: **{candidate_model.get('display_name')}** · `{candidate_model.get('model_id')}`")
        selection = candidate_model.get("selection") or candidate.get("selection") or {}
        if selection:
            st.caption("Candidate selection: qualification gates first, then a weighted comparison of discrimination, calibration, high-precision recall, and business value.")
        st.warning("Deployment is never automatic in this prototype. A developer must explicitly approve the Candidate.")
        confirm = st.checkbox(f"I reviewed {candidate_model.get('display_name')} and approve deployment to Production.")
        if st.button("Promote Candidate to Production", type="primary", disabled=not confirm, use_container_width=True):
            try:
                deployed = promote_registered_model(candidate_model["model_id"])
                st.success(f"Production is now **{deployed.get('model_display_name')}** · `{deployed.get('model_version')}`.")
                st.rerun()
            except Exception as exc:
                st.error(f"Deployment failed: {exc}")
                st.exception(exc)

st.divider()
st.markdown("### Prototype retraining settings")
st.caption(
    "These settings control whether a new dataset or sustained model degradation starts retraining directly inside the app. "
    "The prototype trains Logistic Regression, Random Forest, Extra Trees, LightGBM, and XGBoost, tracks them in MLflow, and registers only the best qualified model as Candidate."
)
settings = retraining.load_retraining_settings()
with st.form("retraining_settings_form"):
    automatic = st.toggle(
        "Automatically retrain when new data or sustained degradation creates a retraining request",
        value=bool(settings.get("automatic_retraining_enabled", True)),
    )
    scope_label = st.radio(
        "Training workload",
        ["Fast prototype", "Full active dataset"],
        index=0 if settings.get("training_scope") == "fast_prototype" else 1,
        help="Fast prototype samples across the full timeline to keep the five-model workflow practical on Streamlit CPU. Full active dataset can take considerably longer.",
    )
    max_rows = st.number_input(
        "Maximum orders in Fast prototype mode",
        min_value=10_000,
        max_value=150_000,
        value=int(settings.get("max_training_rows", 60_000)),
        step=10_000,
        disabled=scope_label == "Full active dataset",
    )
    st.caption("Deployment policy is fixed: retraining may create a Candidate automatically, but only a developer can promote it to Production.")
    if st.form_submit_button("Save retraining settings", use_container_width=True):
        settings = retraining.save_retraining_settings({
            "automatic_retraining_enabled": automatic,
            "training_scope": "fast_prototype" if scope_label == "Fast prototype" else "full_dataset",
            "max_training_rows": int(max_rows),
        })
        st.success("Retraining settings saved.")

st.divider()
st.markdown("### Run retraining now")
try:
    versions = retraining.list_dataset_versions()
except Exception as exc:
    versions = []
    st.warning(f"Dataset versions could not be loaded: {exc}")

if versions:
    version_ids = [v["dataset_version"] for v in versions]
    selected_version = st.selectbox(
        "Dataset version",
        version_ids,
        format_func=lambda vid: next(
            (
                f"{v['dataset_version']} · {v['order_count']:,} orders" + (" · active" if v.get("active") else "")
                for v in versions if v["dataset_version"] == vid
            ),
            vid,
        ),
    )
    active_settings = retraining.load_retraining_settings()
    if active_settings.get("training_scope") == "fast_prototype":
        st.info(
            f"This run will train all five models directly in the app using up to {int(active_settings['max_training_rows']):,} time-spanning orders. "
            "This is the recommended prototype mode for Streamlit CPU."
        )
    else:
        st.warning("Full-dataset mode is selected. Training all five models can take several minutes on a CPU-only Streamlit deployment.")

    if st.button("Run Five-Model Retraining Now", type="primary", use_container_width=True):
        result = run_training_with_ui(dataset_version=selected_version)
        if result and result.get("status") == "CANDIDATE_READY":
            st.success(f"Candidate ready: `{result.get('metadata', {}).get('candidate_model_id', 'selected model')}`. Review it above before deployment.")
            st.rerun()
else:
    st.info("No versioned dataset is currently available for retraining.")

with st.expander("Optional heavy-compute notebook", expanded=False):
    st.caption(
        "The main prototype no longer requires Colab. This notebook is kept as an optional utility if you later want to run the same five-model workflow on a separate compute environment."
    )
    colab_url = retraining.ordinary_colab_url()
    if colab_url:
        st.link_button("Open optional training notebook in Colab", colab_url, use_container_width=True)
    elif NOTEBOOK_PATH.exists():
        st.download_button(
            "Download optional Colab notebook",
            NOTEBOOK_PATH.read_bytes(),
            file_name="end_to_end_ml_workflow.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
    uploaded = st.file_uploader(
        "Import Candidate package from the optional notebook",
        type=["zip"],
        help="Importing a candidate package never deploys it automatically.",
    )
    if uploaded and st.button("Register Imported Candidate", use_container_width=True):
        try:
            imported = import_candidate_package(uploaded.getvalue())
            st.success(f"Registered **{imported.get('display_name', imported.get('model_family'))}** as Candidate. Production is unchanged.")
            st.rerun()
        except Exception as exc:
            st.error(f"Candidate import failed: {exc}")
            st.exception(exc)

st.markdown("### Retraining requests")
requests = retraining.list_retraining_requests()
if not requests:
    st.caption("No retraining requests have been created yet.")
else:
    req = pd.DataFrame(requests)
    keep = [c for c in ["created_at", "request_id", "source", "trigger_type", "model_version", "dataset_version", "status", "training_backend"] if c in req.columns]
    st.dataframe(req[keep].sort_values("created_at", ascending=False), use_container_width=True, hide_index=True)
    queued = next((r for r in reversed(requests) if r.get("status") == "RETRAINING_REQUIRED"), None)
    if queued:
        st.info(f"Queued request `{queued['request_id']}` is waiting for retraining on dataset `{queued.get('dataset_version')}`.")
        if st.button("Run Latest Queued Request Now", use_container_width=True):
            result = run_training_with_ui(request=queued)
            if result and result.get("status") == "CANDIDATE_READY":
                st.success("The queued request produced a Candidate. Review it above before deployment.")
                st.rerun()
