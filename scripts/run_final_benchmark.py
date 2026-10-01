"""Run a fair full-data five-model benchmark and save report-ready evidence.

Usage:
  python scripts/run_final_benchmark.py

By default this uses the packaged Pakistan order-level seed, CPU execution, and does
not persist five extra model artifacts. Add --track-mlflow to also log every run to
the configured MLflow tracking store.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ROOT, RANDOM_STATE, MLFLOW_TRACKING_URI
from src.models.benchmark import FINAL_BENCHMARK_DIR, save_final_benchmark
from src.models.suite import SuiteConfig, train_model_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data/seed/pakistan_orders_v1.csv.gz"))
    parser.add_argument("--dataset-version", default="pakistan_seed_v1")
    parser.add_argument("--gpu", action="store_true", help="Prefer GPU for LightGBM/XGBoost when available")
    parser.add_argument("--track-mlflow", action="store_true", help="Log the benchmark runs to the configured MLflow store")
    args = parser.parse_args()

    if args.track_mlflow:
        try:
            import mlflow  # noqa: F401
        except Exception as exc:
            raise RuntimeError("--track-mlflow requires MLflow from requirements.txt") from exc

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    print(f"Loading {dataset} ...")
    snapshot = pd.read_csv(dataset, parse_dates=["created_at"], low_memory=False)
    print(f"Loaded {len(snapshot):,} orders")

    work = FINAL_BENCHMARK_DIR / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    def progress(stage, fraction, message):
        print(f"[{stage:>24}] {fraction:6.1%}  {message}", flush=True)

    config = SuiteConfig(
        dataset_version=args.dataset_version,
        output_dir=work,
        use_gpu=bool(args.gpu),
        random_seed=RANDOM_STATE,
        experiment_name=f"cancellation-risk-final-benchmark__{args.dataset_version}",
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        training_scope="full_dataset",
        source_rows=len(snapshot),
        enable_mlflow=bool(args.track_mlflow),
        persist_model_artifacts=False,
        package_candidate=False,
    )
    summary = train_model_suite(snapshot, config, progress=progress)
    summary_path, table_path = save_final_benchmark(summary)
    shutil.rmtree(work, ignore_errors=True)
    print(f"Saved benchmark summary: {summary_path}")
    print(f"Saved comparison table: {table_path}")
    print(f"Validation-selected candidate: {summary.get('candidate_model_id')}")


if __name__ == "__main__":
    main()
