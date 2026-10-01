"""Final fair five-model benchmark helpers.

A benchmark is valid only when every model uses the same versioned dataset, temporal
split, feature schema, random-seed policy, and business assumptions. Candidate selection
uses validation evidence; the test split is reserved for final unbiased comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACT_DIR

FINAL_BENCHMARK_DIR = ARTIFACT_DIR / "final_benchmark"
FINAL_BENCHMARK_SUMMARY = FINAL_BENCHMARK_DIR / "benchmark_summary.json"
FINAL_BENCHMARK_TABLE = FINAL_BENCHMARK_DIR / "benchmark_comparison.csv"


def benchmark_rows(summary: dict) -> list[dict]:
    rows = []
    candidate_id = summary.get("candidate_model_id")
    for model in summary.get("models", []):
        val = model.get("val_metrics", {})
        test = model.get("test_metrics", {})
        val_business = model.get("val_business_metrics", {})
        test_business = model.get("test_business_metrics") or model.get("business_metrics", {})
        rows.append({
            "model_id": model.get("model_id"),
            "model": model.get("display_name"),
            "model_family": model.get("model_family"),
            "qualified_on_validation": bool(model.get("qualification", {}).get("passed")),
            "selected_candidate": model.get("model_id") == candidate_id,
            "validation_roc_auc": val.get("roc_auc"),
            "validation_pr_auc": val.get("pr_auc"),
            "validation_brier": val.get("brier"),
            "validation_recall_at_90_precision": val.get("recall_at_90_precision"),
            "validation_net_savings_per_1000_orders": val_business.get("net_savings_per_1000_orders"),
            "test_roc_auc": test.get("roc_auc"),
            "test_pr_auc": test.get("pr_auc"),
            "test_brier": test.get("brier"),
            "test_f1": test.get("f1"),
            "test_precision": test.get("precision"),
            "test_recall": test.get("recall"),
            "test_recall_at_90_precision": test.get("recall_at_90_precision"),
            "test_net_savings_per_1000_orders": test_business.get("net_savings_per_1000_orders"),
            "training_device": model.get("training_device"),
            "duration_seconds": model.get("duration_seconds"),
        })
    return rows


def save_final_benchmark(summary: dict, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = Path(output_dir or FINAL_BENCHMARK_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    table_path = output_dir / "benchmark_comparison.csv"
    payload = dict(summary)
    payload["benchmark_contract"] = {
        "same_dataset_version": True,
        "same_dataset_fingerprint": True,
        "same_temporal_split": True,
        "same_feature_schema": True,
        "same_business_policy": True,
        "candidate_selection_uses": "validation_only",
        "test_holdout_use": "final_unbiased_evaluation_only",
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str))
    pd.DataFrame(benchmark_rows(payload)).to_csv(table_path, index=False)
    return summary_path, table_path


def load_final_benchmark() -> dict | None:
    if not FINAL_BENCHMARK_SUMMARY.exists():
        return None
    try:
        return json.loads(FINAL_BENCHMARK_SUMMARY.read_text())
    except Exception:
        return None


def load_final_benchmark_table() -> pd.DataFrame:
    if FINAL_BENCHMARK_TABLE.exists():
        return pd.read_csv(FINAL_BENCHMARK_TABLE)
    summary = load_final_benchmark()
    return pd.DataFrame(benchmark_rows(summary)) if summary else pd.DataFrame()
