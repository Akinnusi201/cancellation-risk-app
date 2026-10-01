import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cloudpickle
import pandas as pd

from src.config import ARTIFACT_DIR, MLFLOW_TRACKING_URI, RANDOM_STATE
from src.simulation import build_balanced_live_queue

PRODUCTION_MODEL_PATH = ARTIFACT_DIR / "production_model.pkl"
ACTIVE_METADATA_PATH = ARTIFACT_DIR / "active_model.json"
MODEL_REGISTRY_DIR = ARTIFACT_DIR / "model_registry"
MODEL_REGISTRY_INDEX = MODEL_REGISTRY_DIR / "registry.json"
CANDIDATE_STATE_PATH = MODEL_REGISTRY_DIR / "candidate.json"


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


def active_metadata():
    return json.loads(ACTIVE_METADATA_PATH.read_text()) if ACTIVE_METADATA_PATH.exists() else None


@lru_cache(maxsize=12)
def _load_pickle(path: str, mtime_ns: int):
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def _resolve_artifact(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = (ARTIFACT_DIR.parent / path).resolve()
    return path


def load_active_model():
    meta = active_metadata()
    if not meta:
        return None, None

    artifact = _resolve_artifact(meta.get("model_artifact", PRODUCTION_MODEL_PATH))
    if artifact.exists():
        stat = artifact.stat()
        return _load_pickle(str(artifact), stat.st_mtime_ns), meta

    if meta.get("run_id"):
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return mlflow.sklearn.load_model(f"runs:/{meta['run_id']}/model"), meta
    return None, meta


def save_production_model(model, metadata):
    PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRODUCTION_MODEL_PATH, "wb") as f:
        cloudpickle.dump(model, f)
    payload = dict(metadata)
    payload["model_artifact"] = str(PRODUCTION_MODEL_PATH.relative_to(ARTIFACT_DIR.parent))
    ACTIVE_METADATA_PATH.write_text(json.dumps(payload, indent=2, default=str))
    _load_pickle.cache_clear()
    return payload


def load_model_registry():
    if not MODEL_REGISTRY_INDEX.exists():
        return {"registry_version": 1, "created_at": None, "production_model_id": None, "models": []}
    return json.loads(MODEL_REGISTRY_INDEX.read_text())


def save_model_registry(registry):
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    registry = dict(registry)
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    MODEL_REGISTRY_INDEX.write_text(json.dumps(registry, indent=2, default=str))
    return registry


def list_registered_models():
    return list(load_model_registry().get("models", []))


def get_registered_model(model_id):
    return next((m for m in list_registered_models() if m.get("model_id") == model_id), None)


def load_registered_model(model_id):
    meta = get_registered_model(model_id)
    if not meta:
        raise FileNotFoundError(f"Model {model_id} is not registered.")
    artifact = _resolve_artifact(meta["model_artifact"])
    stat = artifact.stat()
    return _load_pickle(str(artifact), stat.st_mtime_ns), meta


def candidate_metadata():
    return json.loads(CANDIDATE_STATE_PATH.read_text()) if CANDIDATE_STATE_PATH.exists() else None


def set_candidate(model_id, reason="Developer selected candidate", selection=None):
    registry = load_model_registry()
    if not any(m.get("model_id") == model_id for m in registry.get("models", [])):
        raise FileNotFoundError(f"Model {model_id} is not registered.")
    for model in registry.get("models", []):
        if model.get("status") == "CANDIDATE":
            model["status"] = "READY"
        if model.get("model_id") == model_id and model.get("status") != "PRODUCTION":
            model["status"] = "CANDIDATE"
    save_model_registry(registry)
    payload = {
        "model_id": model_id,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "selection": selection or {},
        "deployment_policy": "manual_approval_required",
    }
    CANDIDATE_STATE_PATH.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def clear_candidate():
    if CANDIDATE_STATE_PATH.exists():
        CANDIDATE_STATE_PATH.unlink()


def _seed_frame():
    seed = ARTIFACT_DIR.parent / "data" / "seed" / "pakistan_orders_v1.csv.gz"
    if not seed.exists():
        return pd.DataFrame()
    return pd.read_csv(seed, parse_dates=["created_at"], low_memory=False)


def _refresh_production_support_artifacts(model, meta):
    """Install candidate support artifacts, or rebuild them from the packaged seed.

    Colab candidate packages include their own holdout/reference/simulation artifacts.
    Those must win over the packaged seed because a retrained candidate may use newer data.
    """
    from src.config import FEATURES

    model_artifact = _resolve_artifact(meta.get("model_artifact", ""))
    source_dir = model_artifact.parent if model_artifact.exists() else None
    prod_eval = ARTIFACT_DIR / "production_evaluation"
    prod_eval.mkdir(parents=True, exist_ok=True)
    if source_dir:
        test_predictions = source_dir / "test_predictions.csv.gz"
        reference = source_dir / "reference_orders.csv.gz"
        demo = source_dir / "demo_orders.csv.gz"
        historical_demo = source_dir / "historical_demo_orders.csv.gz"
        if test_predictions.exists():
            shutil.copy2(test_predictions, prod_eval / "test_predictions.csv.gz")
        if reference.exists():
            shutil.copy2(reference, ARTIFACT_DIR / "reference_orders.csv.gz")
        if demo.exists():
            shutil.copy2(demo, ARTIFACT_DIR / "demo_orders.csv.gz")
        if historical_demo.exists():
            shutil.copy2(historical_demo, ARTIFACT_DIR / "historical_demo_orders.csv.gz")
        if all(x.exists() for x in [test_predictions, reference, demo, historical_demo]):
            pred = pd.read_csv(test_predictions)
            return {
                "evaluation_dir": str(prod_eval.relative_to(ARTIFACT_DIR.parent)),
                "reference_prediction_summary": {
                    "holdout_rows": int(len(pred)),
                    "mean_probability": float(pred["probability"].mean()),
                    "median_probability": float(pred["probability"].median()),
                    "p95_probability": float(pred["probability"].quantile(0.95)),
                },
                "split_cancellation_prevalence": meta.get("split_cancellation_prevalence", {}),
            }

    seed = _seed_frame()
    if seed.empty:
        return {"evaluation_dir": str(prod_eval.relative_to(ARTIFACT_DIR.parent))}
    ordered = seed.sort_values("created_at").reset_index(drop=True)
    i = int(len(ordered) * 0.70)
    j = int(len(ordered) * 0.85)
    train_df = ordered.iloc[:i].copy()
    test_df = ordered.iloc[j:].copy()
    probs = model.predict_proba(test_df[FEATURES])[:, 1]

    pd.DataFrame({
        "order_id": test_df["order_id"].astype(str).to_numpy(),
        "is_canceled": test_df["is_canceled"].astype(int).to_numpy(),
        "probability": probs,
    }).to_csv(prod_eval / "test_predictions.csv.gz", index=False, compression="gzip")

    train_df.sample(n=min(20_000, len(train_df)), random_state=RANDOM_STATE).to_csv(
        ARTIFACT_DIR / "reference_orders.csv.gz", index=False, compression="gzip"
    )
    historical = test_df.sample(n=min(2_500, len(test_df)), random_state=RANDOM_STATE)
    historical.to_csv(ARTIFACT_DIR / "historical_demo_orders.csv.gz", index=False, compression="gzip")
    live = build_balanced_live_queue(test_df, probs, max_rows=min(2_400, len(test_df)))
    live.to_csv(ARTIFACT_DIR / "demo_orders.csv.gz", index=False, compression="gzip")

    return {
        "evaluation_dir": str(prod_eval.relative_to(ARTIFACT_DIR.parent)),
        "reference_prediction_summary": {
            "holdout_rows": int(len(test_df)),
            "mean_probability": float(pd.Series(probs).mean()),
            "median_probability": float(pd.Series(probs).median()),
            "p95_probability": float(pd.Series(probs).quantile(0.95)),
        },
        "split_cancellation_prevalence": {
            "train": float(train_df["is_canceled"].mean()),
            "validation": float(ordered.iloc[i:j]["is_canceled"].mean()),
            "test": float(test_df["is_canceled"].mean()),
        },
    }


def promote_registered_model(model_id):
    """Deploy one registered model after explicit developer approval."""
    model, meta = load_registered_model(model_id)
    support = _refresh_production_support_artifacts(model, meta)
    payload = {
        "model_name": meta.get("model_family"),
        "model_display_name": meta.get("display_name", MODEL_DISPLAY_NAMES.get(meta.get("model_family"), meta.get("model_family"))),
        "model_version": model_id,
        "run_id": meta.get("run_id"),
        "experiment_name": meta.get("experiment_name"),
        "dataset_version": meta.get("dataset_version"),
        "threshold": meta.get("threshold", 0.5),
        "val_metrics": meta.get("val_metrics", {}),
        "test_metrics": meta.get("test_metrics", {}),
        "business_metrics": meta.get("business_metrics", {}),
        "source": "model_registry_promotion",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "training_device": meta.get("training_device"),
        "training_scope": meta.get("training_scope", "full_dataset"),
        "source_rows": meta.get("source_rows"),
        "reproducibility": meta.get("reproducibility", {}),
        **support,
    }
    active = save_production_model(model, payload)

    registry = load_model_registry()
    for entry in registry.get("models", []):
        if entry.get("status") == "PRODUCTION":
            entry["status"] = "READY"
        if entry.get("model_id") == model_id:
            entry["status"] = "PRODUCTION"
    registry["production_model_id"] = model_id
    save_model_registry(registry)
    clear_candidate()
    return active


def import_candidate_package(package_bytes):
    """Import a Colab-produced candidate package and register it without deploying it."""
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(package_bytes), "r") as zf:
        names = set(zf.namelist())
        required = {"candidate/metadata.json", "candidate/model.pkl"}
        missing = required - names
        if missing:
            raise ValueError("Candidate package is missing: " + ", ".join(sorted(missing)))
        meta = json.loads(zf.read("candidate/metadata.json"))
        model_id = meta.get("model_id") or meta.get("candidate_id")
        if not model_id:
            raise ValueError("Candidate metadata must contain model_id.")
        out = MODEL_REGISTRY_DIR / model_id
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.pkl").write_bytes(zf.read("candidate/model.pkl"))
        for name in [
            "candidate/test_predictions.csv.gz",
            "candidate/reference_orders.csv.gz",
            "candidate/demo_orders.csv.gz",
            "candidate/historical_demo_orders.csv.gz",
        ]:
            if name in names:
                (out / Path(name).name).write_bytes(zf.read(name))

    meta["model_artifact"] = str((out / "model.pkl").relative_to(ARTIFACT_DIR.parent))
    meta["status"] = "CANDIDATE"
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    registry = load_model_registry()
    models = [m for m in registry.get("models", []) if m.get("model_id") != model_id]
    for m in models:
        if m.get("status") == "CANDIDATE":
            m["status"] = "READY"
    models.append(meta)
    registry["models"] = models
    save_model_registry(registry)
    set_candidate(model_id, "Imported best-qualified candidate from Colab experiment", meta.get("selection", {}))
    return meta
