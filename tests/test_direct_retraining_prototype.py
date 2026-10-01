from pathlib import Path

import pandas as pd

from src import retraining

ROOT = Path(__file__).resolve().parents[1]


def test_direct_retraining_is_primary_application_path():
    modelops = (ROOT / "views/6_ModelOps.py").read_text()
    monitoring = (ROOT / "views/7_Model_Monitoring.py").read_text()
    dataops = (ROOT / "views/5_DataOps.py").read_text()
    retraining_text = (ROOT / "src/retraining.py").read_text()

    assert "Run Five-Model Retraining Now" in modelops
    assert "automatic_retraining_enabled" in modelops
    assert "maybe_run_automatic_retraining" in monitoring
    assert "maybe_run_automatic_retraining" in dataops
    assert "run_local_retraining" in retraining_text
    assert "notebookExecutionJobs" not in retraining_text


def test_default_retraining_settings_use_full_dataset_and_manual_deployment():
    settings = retraining.DEFAULT_RETRAINING_SETTINGS
    assert settings["automatic_retraining_enabled"] is True
    assert settings["training_scope"] == "full_dataset"
    assert settings["max_training_rows"] == 60_000
    assert settings["prefer_gpu_if_available"] is True
    assert settings["deployment_policy"] == "manual_promotion_required"


def test_quick_sample_is_deterministic_and_spans_timeline():
    frame = pd.DataFrame({
        "created_at": pd.date_range("2020-01-01", periods=100, freq="D"),
        "order_id": [str(i) for i in range(100)],
        "is_canceled": [i % 2 for i in range(100)],
    })
    a = retraining._quick_sample(frame, 20)
    b = retraining._quick_sample(frame, 20)
    assert a.equals(b)
    assert len(a) == 20
    assert a.iloc[0]["created_at"] == frame.iloc[0]["created_at"]
    assert a.iloc[-1]["created_at"] == frame.iloc[-1]["created_at"]
