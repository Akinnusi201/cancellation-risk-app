import json
from pathlib import Path
import mlflow
import mlflow.sklearn
from src.config import ARTIFACT_DIR, MLFLOW_TRACKING_URI


def active_metadata():
    p=ARTIFACT_DIR/"active_model.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_active_model():
    meta=active_metadata()
    if not meta: return None, None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"runs:/{meta['run_id']}/model"), meta
