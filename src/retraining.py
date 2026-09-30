"""Hybrid monitoring-driven retraining orchestration.

Ordinary Colab is the default classroom path. Colab Enterprise execution is optional
and only activates when Google Cloud configuration and credentials are present.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import ARTIFACT_DIR

RETRAINING_DIR = ARTIFACT_DIR / "retraining"
REQUESTS_PATH = RETRAINING_DIR / "requests.json"
HEALTH_HISTORY_PATH = RETRAINING_DIR / "health_checks.json"


def _read(path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def list_retraining_requests():
    return _read(REQUESTS_PATH, [])


def create_retraining_request(model_version, dataset_version, trigger_type, signals, source="monitoring"):
    requests = list_retraining_requests()
    open_existing = next((
        r for r in reversed(requests)
        if r.get("status") in {"RETRAINING_REQUIRED", "RUNNING", "CANDIDATE_READY"}
        and r.get("model_version") == str(model_version)
        and r.get("dataset_version") == dataset_version
        and r.get("trigger_type") == trigger_type
    ), None)
    if open_existing:
        return open_existing
    request = {
        "request_id": f"retrain_{uuid.uuid4().hex[:10]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": str(model_version),
        "dataset_version": dataset_version,
        "trigger_type": trigger_type,
        "signals": signals,
        "source": source,
        "status": "RETRAINING_REQUIRED",
        "deployment_policy": "candidate_requires_manual_promotion",
    }
    requests.append(request)
    _write(REQUESTS_PATH, requests)
    return request


def update_retraining_request(request_id, status, metadata=None):
    requests = list_retraining_requests()
    for request in requests:
        if request.get("request_id") == request_id:
            request["status"] = status
            request["updated_at"] = datetime.now(timezone.utc).isoformat()
            if metadata:
                request.setdefault("metadata", {}).update(metadata)
            _write(REQUESTS_PATH, requests)
            return request
    raise FileNotFoundError(request_id)


def record_health_check(model_version, signals, minimum_minutes=30):
    history = _read(HEALTH_HISTORY_PATH, [])
    now = datetime.now(timezone.utc)
    latest = next((h for h in reversed(history) if h.get("model_version") == str(model_version)), None)
    if latest:
        try:
            last_time = datetime.fromisoformat(latest["checked_at"])
            if (now - last_time).total_seconds() < minimum_minutes * 60:
                return latest, history
        except Exception:
            pass
    item = {
        "checked_at": now.isoformat(),
        "model_version": str(model_version),
        **signals,
    }
    history.append(item)
    history = history[-200:]
    _write(HEALTH_HISTORY_PATH, history)
    return item, history


def evaluate_retraining_policy(
    model_version,
    dataset_version,
    prediction_psi=None,
    severe_feature_drift_count=0,
    eligible_runtime_predictions=0,
    labeled_metrics=None,
    production_metrics=None,
):
    """Return a transparent hybrid retraining decision.

    Confirmed labeled degradation triggers immediately. Unlabeled drift must be severe
    and sustained across two recorded checks so one noisy batch does not launch training.
    """
    labeled_metrics = labeled_metrics or {}
    production_metrics = production_metrics or {}
    signals = {
        "prediction_psi": prediction_psi,
        "severe_feature_drift_count": int(severe_feature_drift_count or 0),
        "eligible_runtime_predictions": int(eligible_runtime_predictions or 0),
        "labeled_observations": int(labeled_metrics.get("labeled_observations", 0) or 0),
    }

    confirmed = []
    if signals["labeled_observations"] >= 100:
        current_roc = labeled_metrics.get("roc_auc")
        current_brier = labeled_metrics.get("brier")
        current_business = labeled_metrics.get("net_savings_per_1000_orders")
        base_roc = production_metrics.get("roc_auc")
        base_brier = production_metrics.get("brier")
        if current_roc is not None and base_roc is not None and current_roc < base_roc - 0.05:
            confirmed.append(f"ROC-AUC declined by more than 0.05 ({current_roc:.3f} vs {base_roc:.3f})")
        if current_brier is not None and base_brier is not None and current_brier > base_brier + 0.05:
            confirmed.append(f"Brier score worsened by more than 0.05 ({current_brier:.3f} vs {base_brier:.3f})")
        if current_business is not None and current_business <= 0:
            confirmed.append("Observed business value is no longer positive")

    severe_drift = bool(
        signals["eligible_runtime_predictions"] >= 100
        and ((prediction_psi is not None and prediction_psi >= 0.25) or severe_feature_drift_count >= 2)
    )
    signals["severe_drift"] = severe_drift
    signals["confirmed_degradation"] = confirmed

    item, history = record_health_check(model_version, signals)
    recent_same = [h for h in history if h.get("model_version") == str(model_version)][-2:]
    sustained_drift = len(recent_same) >= 2 and all(bool(h.get("severe_drift")) for h in recent_same)
    signals["sustained_drift"] = sustained_drift

    trigger = None
    if confirmed:
        trigger = "CONFIRMED_PERFORMANCE_DEGRADATION"
    elif sustained_drift:
        trigger = "SUSTAINED_DATA_DRIFT"

    request = None
    if trigger:
        request = create_retraining_request(model_version, dataset_version, trigger, signals)
    return {
        "triggered": bool(trigger),
        "trigger_type": trigger,
        "signals": signals,
        "request": request,
    }


def ordinary_colab_url():
    explicit = os.getenv("COLAB_NOTEBOOK_URL")
    if explicit:
        return explicit
    repo = os.getenv("GITHUB_REPOSITORY") or os.getenv("GITHUB_REPO")
    branch = os.getenv("GITHUB_BRANCH", "main")
    if repo:
        repo = repo.removeprefix("https://github.com/").removesuffix(".git").strip("/")
        return f"https://colab.research.google.com/github/{repo}/blob/{branch}/notebooks/end_to_end_ml_workflow.ipynb"
    return None


def colab_enterprise_configuration():
    keys = {
        "project_id": os.getenv("COLAB_ENTERPRISE_PROJECT_ID"),
        "location": os.getenv("COLAB_ENTERPRISE_LOCATION"),
        "runtime_template_id": os.getenv("COLAB_ENTERPRISE_RUNTIME_TEMPLATE_ID"),
        "notebook_gcs_uri": os.getenv("COLAB_ENTERPRISE_NOTEBOOK_GCS_URI"),
        "output_gcs_uri": os.getenv("COLAB_ENTERPRISE_OUTPUT_GCS_URI"),
        "service_account": os.getenv("COLAB_ENTERPRISE_SERVICE_ACCOUNT"),
        "execution_user": os.getenv("COLAB_ENTERPRISE_EXECUTION_USER"),
    }
    required = ["project_id", "location", "runtime_template_id", "notebook_gcs_uri", "output_gcs_uri"]
    keys["configured"] = all(keys.get(k) for k in required) and bool(keys.get("service_account") or keys.get("execution_user"))
    return keys


def trigger_colab_enterprise(display_name=None):
    cfg = colab_enterprise_configuration()
    if not cfg["configured"]:
        raise RuntimeError("Colab Enterprise is not configured. Set the COLAB_ENTERPRISE_* environment variables first.")
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except Exception as exc:
        raise RuntimeError("Install google-auth to trigger Colab Enterprise from the app.") from exc

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    authed = AuthorizedSession(credentials)
    parent = f"projects/{cfg['project_id']}/locations/{cfg['location']}"
    endpoint = f"https://{cfg['location']}-aiplatform.googleapis.com/v1/{parent}/notebookExecutionJobs"
    body = {
        "displayName": display_name or f"cancellation-risk-retrain-{utc_short()}",
        "notebookRuntimeTemplateResourceName": f"{parent}/notebookRuntimeTemplates/{cfg['runtime_template_id']}",
        "gcsNotebookSource": {"uri": cfg["notebook_gcs_uri"]},
        "gcsOutputUri": cfg["output_gcs_uri"],
    }
    if cfg.get("service_account"):
        body["serviceAccount"] = cfg["service_account"]
    else:
        body["executionUser"] = cfg["execution_user"]
    response = authed.post(endpoint, json=body, timeout=30)
    if response.status_code >= 300:
        raise RuntimeError(f"Colab Enterprise trigger failed ({response.status_code}): {response.text[:500]}")
    return response.json()


def utc_short():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def enterprise_auto_trigger_enabled():
    return os.getenv("AUTO_TRIGGER_COLAB_ENTERPRISE", "false").strip().lower() in {"1", "true", "yes", "on"}


def maybe_auto_trigger_enterprise(request):
    """Submit an open request to Colab Enterprise when explicitly enabled.

    This opt-in guard prevents an accidental cloud-cost side effect in classroom or
    local deployments. Ordinary Colab remains the default manual launch path.
    """
    if not request or request.get("status") != "RETRAINING_REQUIRED":
        return request
    if not enterprise_auto_trigger_enabled() or not colab_enterprise_configuration().get("configured"):
        return request
    result = trigger_colab_enterprise(display_name=f"cancellation-risk-{request['request_id']}")
    job_name = result.get("name") or result.get("metadata", {}).get("target")
    return update_retraining_request(
        request["request_id"],
        "RUNNING",
        {"colab_enterprise_job": job_name, "trigger_response": result},
    )
