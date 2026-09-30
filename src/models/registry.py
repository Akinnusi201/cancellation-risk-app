import json
from functools import lru_cache
from pathlib import Path

import cloudpickle
from src.config import ARTIFACT_DIR, MLFLOW_TRACKING_URI

PRODUCTION_MODEL_PATH = ARTIFACT_DIR / "production_model.pkl"
ACTIVE_METADATA_PATH = ARTIFACT_DIR / "active_model.json"


def active_metadata():
    return json.loads(ACTIVE_METADATA_PATH.read_text()) if ACTIVE_METADATA_PATH.exists() else None


@lru_cache(maxsize=2)
def _load_pickle(path: str, mtime_ns: int):
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def load_active_model():
    meta = active_metadata()
    if not meta:
        return None, None

    artifact = Path(meta.get("model_artifact", PRODUCTION_MODEL_PATH))
    if not artifact.is_absolute():
        artifact = (ARTIFACT_DIR.parent / artifact).resolve()
    if artifact.exists():
        stat = artifact.stat()
        return _load_pickle(str(artifact), stat.st_mtime_ns), meta

    # Backward-compatible fallback for older deployments that only stored an MLflow run ID.
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
