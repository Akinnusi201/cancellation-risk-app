"""Retraining orchestration for the deployed ML application.

New versioned data or sustained model degradation can create a retraining request.
When automatic retraining is enabled, the five-model suite runs in the application
environment, MLflow tracks the experiment, and the best qualified model is registered
as Candidate. Production remains unchanged until a developer explicitly promotes it.

Full-dataset training is the default. A deterministic sampled mode is retained only as
an optional faster run for demonstrations, diagnostics, or resource-constrained hosts.
The packaged Colab notebook remains an optional accelerated-compute alternative.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import ARTIFACT_DIR, MLFLOW_TRACKING_URI

RETRAINING_DIR = ARTIFACT_DIR / "retraining"
REQUESTS_PATH = RETRAINING_DIR / "requests.json"
HEALTH_HISTORY_PATH = RETRAINING_DIR / "health_checks.json"
SETTINGS_PATH = RETRAINING_DIR / "settings.json"
RUNS_DIR = RETRAINING_DIR / "runs"

DEFAULT_RETRAINING_SETTINGS = {
    "automatic_retraining_enabled": True,
    "training_scope": "full_dataset",
    "max_training_rows": 60_000,
    "prefer_gpu_if_available": True,
    "include_demo_business_impact": True,
    "deployment_policy": "manual_promotion_required",
    "candidate_registration": "automatic",
}


def _read(path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def load_retraining_settings():
    stored = _read(SETTINGS_PATH, {})
    settings = {**DEFAULT_RETRAINING_SETTINGS, **(stored or {})}
    if settings.get("training_scope") == "fast_prototype":
        settings["training_scope"] = "quick_sample"
    settings["prefer_gpu_if_available"] = bool(settings.get("prefer_gpu_if_available", True))
    settings["include_demo_business_impact"] = bool(settings.get("include_demo_business_impact", True))
    return settings


def save_retraining_settings(settings):
    clean = {**DEFAULT_RETRAINING_SETTINGS, **(settings or {})}
    clean["automatic_retraining_enabled"] = bool(clean.get("automatic_retraining_enabled", True))
    scope = clean.get("training_scope")
    # Backward compatibility with earlier builds. The sampled path remains available,
    # but full-dataset training is the standard operating mode.
    clean["training_scope"] = "quick_sample" if scope in {"quick_sample", "fast_prototype"} else "full_dataset"
    clean["max_training_rows"] = max(5_000, int(clean.get("max_training_rows", 60_000) or 60_000))
    clean["prefer_gpu_if_available"] = bool(clean.get("prefer_gpu_if_available", True))
    clean["include_demo_business_impact"] = bool(clean.get("include_demo_business_impact", True))
    clean["deployment_policy"] = "manual_promotion_required"
    clean["candidate_registration"] = "automatic"
    _write(SETTINGS_PATH, clean)
    return clean


def list_retraining_requests():
    return _read(REQUESTS_PATH, [])


def create_retraining_request(model_version, dataset_version, trigger_type, signals, source="monitoring", force_new=False):
    requests = list_retraining_requests()
    open_existing = None if force_new else next((
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
        "training_backend": "in_app_local",
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

    _, history = record_health_check(model_version, signals)
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


def latest_active_dataset():
    """Return metadata for the active versioned dataset."""
    from src.database.duckdb_manager import connect

    with connect() as con:
        row = con.execute(
            """
            SELECT dataset_version, processed_path, row_count, order_count
            FROM dataset_versions
            WHERE active = TRUE
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        return None
    return {
        "dataset_version": row[0],
        "processed_path": row[1],
        "row_count": int(row[2] or 0),
        "order_count": int(row[3] or 0),
    }


def list_dataset_versions():
    from src.database.duckdb_manager import connect

    with connect() as con:
        rows = con.execute(
            """
            SELECT dataset_version, processed_path, row_count, order_count, active, created_at
            FROM dataset_versions
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [
        {
            "dataset_version": r[0],
            "processed_path": r[1],
            "row_count": int(r[2] or 0),
            "order_count": int(r[3] or 0),
            "active": bool(r[4]),
            "created_at": r[5],
        }
        for r in rows
    ]


def _quick_sample(snapshot, max_rows):
    """Create a deterministic time-spanning sample for optional quick training runs."""
    if len(snapshot) <= max_rows:
        return snapshot.copy()
    ordered = snapshot.sort_values(["created_at", "order_id"]).reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, num=max_rows, dtype=int)
    return ordered.iloc[np.unique(indices)].reset_index(drop=True)


def load_training_snapshot(dataset_version=None, settings=None):
    """Load a versioned snapshot and apply the configured full/quick training scope."""
    from src.data.io import read_snapshot

    settings = settings or load_retraining_settings()
    versions = list_dataset_versions()
    selected = next((v for v in versions if v["dataset_version"] == dataset_version), None)
    if selected is None:
        selected = next((v for v in versions if v.get("active")), None)
    if selected is None:
        raise FileNotFoundError("No versioned dataset is available for retraining.")

    path = Path(selected["processed_path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset artifact does not exist: {path}")
    snapshot = read_snapshot(path)
    original_rows = len(snapshot)
    if settings.get("training_scope") in {"quick_sample", "fast_prototype"}:
        snapshot = _quick_sample(snapshot, int(settings.get("max_training_rows", 60_000)))
    return snapshot, selected, {
        "source_rows": int(original_rows),
        "training_rows": int(len(snapshot)),
        "training_scope": settings.get("training_scope"),
    }


def _new_manual_request(dataset_version, trigger_type="MANUAL_RETRAIN"):
    from src.models.registry import active_metadata

    active = active_metadata() or {}
    return create_retraining_request(
        active.get("model_version", "unknown"),
        dataset_version,
        trigger_type,
        {"requested_by": "developer", "dataset_version": dataset_version},
        source="modelops",
        force_new=True,
    )


def run_local_retraining(request=None, dataset_version=None, progress=None, settings=None):
    """Run the complete five-model retraining workflow directly in the app process.

    All five runs are tracked in MLflow. The best qualified model is automatically
    registered as Candidate. Production never changes here; promotion remains a separate
    developer action.
    """
    from src.database.duckdb_manager import log_event
    from src.models.registry import import_candidate_package
    from src.models.suite import SuiteConfig, train_model_suite

    settings = settings or load_retraining_settings()
    if request is None:
        dataset_version = dataset_version or (latest_active_dataset() or {}).get("dataset_version")
        if not dataset_version:
            raise FileNotFoundError("No active dataset is available for retraining.")
        request = _new_manual_request(dataset_version)
    request_id = request["request_id"]
    if request.get("status") == "CANDIDATE_READY":
        return request

    update_retraining_request(request_id, "RUNNING", {
        "training_backend": "in_app_local",
        "training_scope": settings.get("training_scope"),
    })
    log_event(
        f"retrain_start_{request_id}",
        "MODEL_RETRAINING",
        "STARTED",
        f"In-app retraining started for {request.get('dataset_version')}",
        {"request_id": request_id, "trigger_type": request.get("trigger_type")},
    )

    try:
        if progress:
            progress("load_data", 0.05, "Loading the selected versioned dataset")
        snapshot, selected, scope_meta = load_training_snapshot(
            request.get("dataset_version") or dataset_version,
            settings=settings,
        )
        if progress:
            progress(
                "load_data",
                1.0,
                f"Loaded {scope_meta['training_rows']:,} of {scope_meta['source_rows']:,} orders for {selected['dataset_version']}",
            )

        output_dir = RUNS_DIR / request_id
        experiment_name = f"cancellation-risk__{selected['dataset_version']}__{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        config = SuiteConfig(
            dataset_version=selected["dataset_version"],
            output_dir=output_dir,
            use_gpu=bool(settings.get("prefer_gpu_if_available", True)),
            experiment_name=experiment_name,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            training_scope=scope_meta["training_scope"],
            source_rows=scope_meta["source_rows"],
        )
        summary = train_model_suite(snapshot, config, progress=progress)
        candidate_id = summary.get("candidate_model_id")
        package_path = summary.get("candidate_package")
        imported = None
        if candidate_id and package_path and Path(package_path).exists():
            if progress:
                progress("register_candidate", 0.35, f"Registering {candidate_id} as Candidate")
            imported = import_candidate_package(Path(package_path).read_bytes())
            if progress:
                progress("register_candidate", 1.0, "Candidate registered; production is unchanged")

        status = "CANDIDATE_READY" if imported else "COMPLETED_NO_CANDIDATE"
        updated = update_retraining_request(request_id, status, {
            "dataset_version": selected["dataset_version"],
            "experiment_name": summary.get("experiment_name"),
            "candidate_model_id": candidate_id,
            "candidate_package": package_path,
            "training_scope": scope_meta["training_scope"],
            "source_rows": scope_meta["source_rows"],
            "training_rows": scope_meta["training_rows"],
            "model_count": len(summary.get("models", [])),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        log_event(
            f"retrain_done_{request_id}",
            "MODEL_RETRAINING",
            "SUCCESS" if imported else "WARNING",
            f"Retraining completed; candidate {candidate_id or 'not selected'}",
            {"request_id": request_id, "candidate_model_id": candidate_id},
        )
        return updated
    except Exception as exc:
        update_retraining_request(request_id, "FAILED", {"error": str(exc)})
        log_event(
            f"retrain_failed_{request_id}",
            "MODEL_RETRAINING",
            "FAILED",
            str(exc),
            {"request_id": request_id},
        )
        raise


def maybe_run_automatic_retraining(request, progress=None):
    """Run an open retraining request in-app when automatic retraining is enabled."""
    if not request or request.get("status") != "RETRAINING_REQUIRED":
        return request
    settings = load_retraining_settings()
    if not settings.get("automatic_retraining_enabled", True):
        return request
    return run_local_retraining(request=request, progress=progress, settings=settings)


def ordinary_colab_url():
    """Optional accelerated-compute notebook link."""
    explicit = os.getenv("COLAB_NOTEBOOK_URL")
    if explicit:
        return explicit
    repo = os.getenv("GITHUB_REPOSITORY") or os.getenv("GITHUB_REPO")
    branch = os.getenv("GITHUB_BRANCH", "main")
    if repo:
        repo = repo.removeprefix("https://github.com/").removesuffix(".git").strip("/")
        return f"https://colab.research.google.com/github/{repo}/blob/{branch}/notebooks/end_to_end_ml_workflow.ipynb"
    return None
