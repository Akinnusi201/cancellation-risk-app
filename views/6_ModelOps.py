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
from src.retraining import (
    colab_enterprise_configuration,
    list_retraining_requests,
    ordinary_colab_url,
    trigger_colab_enterprise,
)

require_role("developer")
st.title("🤖 Model Registry & Deployment")
st.caption(
    "Compare trained models, review the current candidate, and control production deployment. "
    "Training is intentionally separated from the web app and runs in Google Colab or Colab Enterprise."
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
    "**Production** is the model currently scoring orders. **Candidate** is the proposed replacement. "
    "**Ready** models are trained and available for comparison but are not deployed."
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
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)

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
        f"Seed: {selected.get('random_seed', 42)} · Scope: {selected.get('training_scope', 'full training workflow')}"
    )

    if selected.get("status") != "PRODUCTION":
        if st.button("Mark selected model as Candidate", use_container_width=True):
            set_candidate(selected_id, "Developer selected model from registry")
            st.success(f"{selected.get('display_name')} is now the candidate. Production has not changed.")
            st.rerun()

st.divider()
st.markdown("### Candidate review and deployment")
candidate = candidate_metadata()
if not candidate:
    st.info("There is no active candidate. A Colab experiment can select one automatically, or you can mark a registered model as Candidate above.")
else:
    candidate_model = next((m for m in list_registered_models() if m.get("model_id") == candidate.get("model_id")), None)
    if not candidate_model:
        st.warning("Candidate metadata exists, but its model artifact is not registered.")
    else:
        st.success(f"Candidate: **{candidate_model.get('display_name')}** · `{candidate_model.get('model_id')}`")
        selection = candidate_model.get("selection") or candidate.get("selection") or {}
        if selection:
            st.caption("Candidate selection: passed qualification gates, then ranked using discrimination, calibration, fixed-precision recall, and business value.")
        st.warning("Deployment is never automatic in this project. A developer must explicitly approve the candidate.")
        confirm = st.checkbox(f"I reviewed {candidate_model.get('display_name')} and approve deployment to production.")
        if st.button("Promote Candidate to Production", type="primary", disabled=not confirm, use_container_width=True):
            try:
                active = promote_registered_model(candidate_model["model_id"])
                st.success(f"Production is now **{active.get('model_display_name')}** · `{active.get('model_version')}`.")
                st.rerun()
            except Exception as exc:
                st.error(f"Deployment failed: {exc}")
                st.exception(exc)

st.divider()
st.markdown("### Train or retrain in Google Colab")
st.caption(
    "The notebook runs the complete computational workflow: prepare data, train five models, track every run in MLflow, "
    "apply qualification gates, select the best qualified candidate, and export a candidate package. Random Forest and Extra Trees use CPU; "
    "LightGBM and XGBoost prefer the Colab GPU and fall back to CPU if needed."
)

colab_url = ordinary_colab_url()
left, right = st.columns(2)
if colab_url:
    left.link_button("Open Training Workflow in Colab", colab_url, use_container_width=True)
elif NOTEBOOK_PATH.exists():
    left.download_button(
        "Download Colab Training Notebook",
        NOTEBOOK_PATH.read_bytes(),
        file_name="end_to_end_ml_workflow.ipynb",
        mime="application/x-ipynb+json",
        use_container_width=True,
    )
else:
    left.button("Training notebook unavailable", disabled=True, use_container_width=True)

cfg = colab_enterprise_configuration()
if cfg.get("configured"):
    if right.button("Trigger Colab Enterprise Retraining", use_container_width=True):
        try:
            result = trigger_colab_enterprise()
            st.success("Colab Enterprise notebook execution was submitted.")
            st.json(result)
        except Exception as exc:
            st.error(str(exc))
else:
    right.button("Colab Enterprise: not configured", disabled=True, use_container_width=True)
    st.caption("Optional production path: configure the COLAB_ENTERPRISE_* environment variables and Google Cloud credentials to enable automatic notebook execution.")

uploaded = st.file_uploader("Import candidate package produced by Colab", type=["zip"], help="Upload the candidate_package__*.zip created by the notebook. Importing registers the candidate but does not deploy it.")
if uploaded and st.button("Register Imported Candidate", use_container_width=True):
    try:
        imported = import_candidate_package(uploaded.getvalue())
        st.success(f"Registered **{imported.get('display_name', imported.get('model_family'))}** as Candidate. Production is unchanged.")
        st.rerun()
    except Exception as exc:
        st.error(f"Candidate import failed: {exc}")
        st.exception(exc)

st.markdown("### Retraining requests")
requests = list_retraining_requests()
if not requests:
    st.caption("No monitoring-triggered retraining requests are open.")
else:
    req = pd.DataFrame(requests)
    keep = [c for c in ["created_at", "request_id", "trigger_type", "model_version", "dataset_version", "status"] if c in req.columns]
    st.dataframe(req[keep].sort_values("created_at", ascending=False), use_container_width=True, hide_index=True)
