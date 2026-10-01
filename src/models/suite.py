"""Reproducible five-model training used by the application and optional Colab workflow.

The application never trains during normal startup. Training runs only when an explicit
manual or automatic retraining job is created from versioned data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.business import DEFAULT_POLICY
from src.business_evaluation import evaluate_business_policy
from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, RANDOM_STATE

# FEATURE_SCHEMA_VERSION was introduced in v4.4.  Keep the training suite
# import-compatible with older deployed config.py files during rolling/partial
# upgrades so the Experiments page does not crash before Streamlit can render.
try:
    from src.config import FEATURE_SCHEMA_VERSION
except ImportError:  # compatibility with pre-v4.4 config.py
    FEATURE_SCHEMA_VERSION = "order_features_v1"
from src.models.evaluate import best_f1_threshold, metrics
from src.simulation import build_balanced_live_queue

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

DEFAULT_GATES = {
    # Gates are evaluated on the validation split only. The final temporal test
    # holdout is reserved for unbiased reporting after a Candidate is selected.
    "min_roc_auc": 0.80,
    "min_pr_auc": 0.70,
    "max_brier": 0.22,
    "min_recall_at_90_precision": 0.15,
    "min_net_savings_per_1000_orders": 0.0,
}


@dataclass
class SuiteConfig:
    dataset_version: str
    output_dir: Path
    use_gpu: bool = True
    random_seed: int = RANDOM_STATE
    experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
    business_policy: dict | None = None
    gates: dict | None = None
    training_scope: str = "full_dataset"
    source_rows: int | None = None
    enable_mlflow: bool = True
    persist_model_artifacts: bool = True
    package_candidate: bool = True


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.getenv("GIT_COMMIT_SHA", "unknown")


def environment_metadata():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ["sklearn", "lightgbm", "xgboost", "pandas", "numpy"]:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = "not-installed"
    versions["git_commit_sha"] = git_commit_sha()
    return versions


def temporal_split(df, train_frac=0.70, val_frac=0.15):
    ordered = df.sort_values("created_at").reset_index(drop=True)
    n = len(ordered)
    i = int(n * train_frac)
    j = int(n * (train_frac + val_frac))
    return ordered.iloc[:i].copy(), ordered.iloc[i:j].copy(), ordered.iloc[j:].copy()


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable SHA-256 fingerprint for the exact model-ready dataset contents."""
    columns = [c for c in ["order_id", "created_at", "is_canceled", *FEATURES] if c in df.columns]
    ordered = df.sort_values([c for c in ["created_at", "order_id"] if c in df.columns]).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered[columns], index=False, categorize=True).to_numpy(dtype="uint64")
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _split_stats(frame: pd.DataFrame) -> dict:
    created = pd.to_datetime(frame["created_at"], errors="coerce") if "created_at" in frame else pd.Series(dtype="datetime64[ns]")
    return {
        "rows": int(len(frame)),
        "cancellation_rate": float(frame["is_canceled"].mean()) if len(frame) else None,
        "start": created.min().isoformat() if len(created.dropna()) else None,
        "end": created.max().isoformat() if len(created.dropna()) else None,
    }


def split_manifest(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, fingerprint: str) -> dict:
    """Describe the deterministic temporal split used by every model in one experiment."""
    manifest = {
        "strategy": "temporal_70_15_15",
        "dataset_fingerprint": fingerprint,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "train": _split_stats(train_df),
        "validation": _split_stats(val_df),
        "test": _split_stats(test_df),
    }
    split_key = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["split_id"] = hashlib.sha256(split_key.encode("utf-8")).hexdigest()[:16]
    return manifest


def _preprocessor(scale_numeric=False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num = Pipeline(num_steps)
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def _fit_with_gpu_fallback(estimator, X, y, family, use_gpu):
    device = "CPU"
    if family == "lightgbm" and use_gpu:
        try:
            estimator.set_params(device="gpu")
            estimator.fit(X, y)
            return estimator, "GPU"
        except Exception:
            estimator.set_params(device="cpu")
    elif family == "xgboost" and use_gpu:
        try:
            # XGBoost 2+ uses device='cuda' with hist tree method.
            estimator.set_params(device="cuda", tree_method="hist")
            estimator.fit(X, y)
            return estimator, "GPU"
        except Exception:
            estimator.set_params(device="cpu", tree_method="hist")
    estimator.fit(X, y)
    return estimator, device


MODEL_DEFAULT_PARAMS = {
    "logistic_regression": {"max_iter": 600, "C": 1.0, "solver": "lbfgs", "class_weight": None},
    # Defaults are sized so a full 318k-order five-model run is feasible on a CPU
    # host while still providing meaningful nonlinear baselines. Developers can
    # increase the budgets from the manual experiment workbench.
    "random_forest": {"n_estimators": 40, "max_depth": 14, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced_subsample", "max_samples": 0.60},
    "extra_trees": {"n_estimators": 50, "max_depth": 16, "min_samples_leaf": 8, "max_features": "sqrt", "class_weight": "balanced"},
    "lightgbm": {"n_estimators": 160, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9},
    "xgboost": {"n_estimators": 160, "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 2, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
}


def model_parameter_defaults(family):
    if family not in MODEL_DEFAULT_PARAMS:
        raise ValueError(f"Unknown model family: {family}")
    return dict(MODEL_DEFAULT_PARAMS[family])


def build_estimator(family, random_seed=RANDOM_STATE, overrides=None):
    """Build one supported estimator with validated, reproducible overrides."""
    if family == "lightgbm" and LGBMClassifier is None:
        raise RuntimeError("Install LightGBM before training a LightGBM model.")
    if family == "xgboost" and XGBClassifier is None:
        raise RuntimeError("Install XGBoost before training an XGBoost model.")
    params = model_parameter_defaults(family)
    for key, value in (overrides or {}).items():
        if key not in params:
            raise ValueError(f"Unsupported {family} parameter: {key}")
        params[key] = value

    if family == "logistic_regression":
        return LogisticRegression(random_state=random_seed, **params)
    if family == "random_forest":
        return RandomForestClassifier(random_state=random_seed, n_jobs=-1, **params)
    if family == "extra_trees":
        return ExtraTreesClassifier(random_state=random_seed, n_jobs=-1, **params)
    if family == "lightgbm":
        return LGBMClassifier(random_state=random_seed, verbosity=-1, n_jobs=-1, **params)
    if family == "xgboost":
        return XGBClassifier(random_state=random_seed, n_jobs=-1, tree_method="hist", eval_metric="logloss", **params)
    raise ValueError(f"Unknown model family: {family}")


def model_specifications(random_seed=RANDOM_STATE):
    return {family: build_estimator(family, random_seed) for family in DISPLAY_NAMES}

def _selection_evidence(record):
    """Return validation-only evidence used for model selection.

    Test metrics are intentionally excluded from qualification and ranking so the final
    holdout remains an unbiased estimate after a Candidate has been chosen. Older
    packaged records fall back to test evidence only for display/backward compatibility.
    """
    metrics_block = record.get("val_metrics") or record.get("test_metrics") or {}
    business_block = record.get("val_business_metrics") or record.get("business_metrics") or {}
    return metrics_block, business_block


def qualify_model(record, gates=None):
    gates = {**DEFAULT_GATES, **(gates or {})}
    m, b = _selection_evidence(record)
    checks = {
        "ROC-AUC gate": m.get("roc_auc", 0) >= gates["min_roc_auc"],
        "PR-AUC gate": m.get("pr_auc", 0) >= gates["min_pr_auc"],
        "Calibration gate": m.get("brier", 1) <= gates["max_brier"],
        "Recall @ 90% precision gate": m.get("recall_at_90_precision", 0) >= gates["min_recall_at_90_precision"],
        "Business value gate": b.get("net_savings_per_1000_orders", -math.inf) >= gates["min_net_savings_per_1000_orders"],
    }
    return all(checks.values()), checks


def select_candidate(records, gates=None):
    """Select a Candidate using validation evidence only; never rank on the test holdout."""
    qualified = []
    for record in records:
        passed, checks = qualify_model(record, gates)
        record["qualification"] = {
            "passed": passed,
            "checks": checks,
            "evidence_split": "validation" if record.get("val_metrics") else "legacy_test_fallback",
        }
        if passed:
            qualified.append(record)
    if not qualified:
        return None

    business = [_selection_evidence(r)[1].get("net_savings_per_1000_orders", 0.0) for r in qualified]
    lo, hi = min(business), max(business)
    for r in qualified:
        m, b_metrics = _selection_evidence(r)
        b = b_metrics.get("net_savings_per_1000_orders", 0.0)
        b_norm = 1.0 if hi == lo else (b - lo) / (hi - lo)
        r["selection_score"] = float(
            0.25 * m["roc_auc"]
            + 0.25 * m["pr_auc"]
            + 0.20 * (1.0 - m["brier"])
            + 0.15 * m.get("recall_at_90_precision", 0.0)
            + 0.15 * b_norm
        )
        r["selection_evidence"] = {
            "split": "validation" if r.get("val_metrics") else "legacy_test_fallback",
            "metrics": dict(m),
            "business_metrics": dict(b_metrics),
        }
    return max(qualified, key=lambda r: r["selection_score"])


def _mlflow_context(tracking_uri, experiment_name):
    try:
        import mlflow
    except Exception:
        return None
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def _safe_log_model(mlflow, pipeline, artifact_path="model"):
    if mlflow is None:
        return
    try:
        import mlflow.sklearn
        mlflow.sklearn.log_model(pipeline, artifact_path)
    except Exception:
        pass


def train_model_suite(snapshot: pd.DataFrame, config: SuiteConfig, progress=None):
    """Train all five models, track them, apply gates, and export a candidate package."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    experiment_name = config.experiment_name or f"cancellation-risk__{config.dataset_version}__{stamp[:8]}"
    policy = {**DEFAULT_POLICY, **(config.business_policy or {})}
    gates = {**DEFAULT_GATES, **(config.gates or {})}
    env = environment_metadata()
    fingerprint = dataset_fingerprint(snapshot)

    train_df, val_df, test_df = temporal_split(snapshot)
    manifest = split_manifest(train_df, val_df, test_df, fingerprint)
    if min(train_df.is_canceled.nunique(), val_df.is_canceled.nunique(), test_df.is_canceled.nunique()) < 2:
        raise ValueError("Each temporal split must contain both outcome classes.")
    Xtr, ytr = train_df[FEATURES], train_df["is_canceled"].astype(int)
    Xv, yv = val_df[FEATURES], val_df["is_canceled"].astype(int)
    Xt, yt = test_df[FEATURES], test_df["is_canceled"].astype(int)

    # Transform once for tree models and once for scaled Logistic Regression.
    if progress:
        progress("preprocess", 0.05, "Preparing leakage-safe features")
    tree_prep = _preprocessor(scale_numeric=False)
    Xtr_tree = tree_prep.fit_transform(Xtr, ytr)
    Xv_tree = tree_prep.transform(Xv)
    Xt_tree = tree_prep.transform(Xt)
    log_prep = _preprocessor(scale_numeric=True)
    Xtr_log = log_prep.fit_transform(Xtr, ytr)
    Xv_log = log_prep.transform(Xv)
    Xt_log = log_prep.transform(Xt)
    if progress:
        progress("preprocess", 1.0, "Feature matrices ready")

    mlflow = _mlflow_context(config.mlflow_tracking_uri, experiment_name) if config.enable_mlflow else None
    specs = model_specifications(config.random_seed)
    results = []
    for idx, (family, estimator) in enumerate(specs.items(), 1):
        display = DISPLAY_NAMES[family]
        run_name = f"{family}__{stamp}"
        if progress:
            progress(f"train_{family}", 0.05, f"Training {display} ({idx} of {len(specs)})")
        started = time.perf_counter()
        prep = log_prep if family == "logistic_regression" else tree_prep
        Xtr_t = Xtr_log if family == "logistic_regression" else Xtr_tree
        Xv_t = Xv_log if family == "logistic_regression" else Xv_tree
        Xt_t = Xt_log if family == "logistic_regression" else Xt_tree

        run_cm = mlflow.start_run(run_name=run_name) if mlflow else None
        if run_cm:
            run = run_cm.__enter__()
        else:
            run = None
        try:
            estimator, device = _fit_with_gpu_fallback(estimator, Xtr_t, ytr, family, config.use_gpu)
            if progress:
                progress(f"train_{family}", 1.0, f"Finished fitting {display} on {device}")
                progress(f"evaluate_{family}", 0.20, f"Evaluating {display} on validation and held-out test data")
            pipeline = Pipeline([("prep", prep), ("model", estimator)])
            pv = estimator.predict_proba(Xv_t)[:, 1]
            pt = estimator.predict_proba(Xt_t)[:, 1]
            threshold, _ = best_f1_threshold(yv, pv)
            vm = metrics(yv, pv, threshold)
            tm = metrics(yt, pt, threshold)
            vbm = evaluate_business_policy(yv.to_numpy(), pv, policy)
            tbm = evaluate_business_policy(yt.to_numpy(), pt, policy)
            if progress:
                progress(f"evaluate_{family}", 1.0, f"{display}: ROC-AUC {tm['roc_auc']:.3f}, PR-AUC {tm['pr_auc']:.3f}, Brier {tm['brier']:.3f}")
            duration = time.perf_counter() - started

            model_dir = output / run_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pkl"
            if config.persist_model_artifacts:
                with open(model_path, "wb") as f:
                    cloudpickle.dump(pipeline, f)
            pd.DataFrame({
                "order_id": test_df["order_id"].astype(str).to_numpy(),
                "is_canceled": yt.to_numpy(),
                "probability": pt,
            }).to_csv(model_dir / "test_predictions.csv.gz", index=False, compression="gzip")

            record = {
                "model_id": run_name,
                "model_family": family,
                "display_name": display,
                "status": "READY",
                "dataset_version": config.dataset_version,
                "experiment_name": experiment_name,
                "run_name": run_name,
                "run_id": run.info.run_id if run else None,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "training_device": device,
                "training_scope": config.training_scope,
                "source_rows": int(config.source_rows) if config.source_rows is not None else int(len(snapshot)),
                "random_seed": config.random_seed,
                "threshold": threshold,
                "val_metrics": vm,
                "test_metrics": tm,
                "val_business_metrics": vbm,
                "test_business_metrics": tbm,
                "business_metrics": tbm,
                "dataset_fingerprint": fingerprint,
                "split_id": manifest["split_id"],
                "split_manifest": manifest,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "duration_seconds": duration,
                "model_artifact": str(model_path) if config.persist_model_artifacts else None,
                "training_rows": len(train_df),
                "validation_rows": len(val_df),
                "test_rows": len(test_df),
                "reproducibility": env,
            }
            if mlflow:
                mlflow.log_params({
                    "model_family": family,
                    "dataset_version": config.dataset_version,
                    "random_seed": config.random_seed,
                    "training_device": device,
                    "training_scope": config.training_scope,
                    "source_rows": int(config.source_rows) if config.source_rows is not None else int(len(snapshot)),
                    "git_commit_sha": env.get("git_commit_sha"),
                    "dataset_fingerprint": fingerprint,
                    "split_id": manifest["split_id"],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    **{f"gate__{k}": v for k, v in gates.items()},
                    **{f"policy__{k}": v for k, v in policy.items()},
                    **{f"model__{k}": v for k, v in estimator.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))},
                })
                mlflow.log_metrics({f"val_{k}": v for k, v in vm.items()} | {f"test_{k}": v for k, v in tm.items()})
                mlflow.log_metrics({f"val_business_{k}": v for k, v in vbm.items() if isinstance(v, (int, float))})
                mlflow.log_metrics({f"test_business_{k}": v for k, v in tbm.items() if isinstance(v, (int, float))})
                mlflow.set_tags({
                    "model_family": family,
                    "dataset_version": config.dataset_version,
                    "run_role": "model_suite",
                    "candidate_eligible": "pending",
                    "selection_split": "validation",
                    "test_holdout_role": "final_unbiased_evaluation",
                })
                if config.persist_model_artifacts:
                    _safe_log_model(mlflow, pipeline)
            results.append(record)
        finally:
            if run_cm:
                run_cm.__exit__(None, None, None)

    if progress:
        progress("select_candidate", 0.25, "Applying qualification gates across all five models")
    candidate = select_candidate(results, gates)
    if candidate:
        candidate["status"] = "CANDIDATE"
        candidate["selection"] = {
            "selected_by": "validation_qualification_gates_then_weighted_score",
            "selection_score": candidate.get("selection_score"),
            "selection_split": "validation",
            "test_holdout_role": "final_unbiased_evaluation_only",
            "gates": gates,
        }
    if progress:
        if candidate:
            progress("select_candidate", 1.0, f"Selected {candidate['display_name']} as the best qualified Candidate")
        else:
            progress("select_candidate", 1.0, "No model passed all qualification gates")

    if mlflow:
        try:
            client = mlflow.tracking.MlflowClient()
            for record in results:
                if not record.get("run_id"):
                    continue
                client.set_tag(record["run_id"], "qualification_passed", str(bool(record.get("qualification", {}).get("passed"))).lower())
                client.set_tag(record["run_id"], "selected_candidate", str(bool(candidate and record["model_id"] == candidate["model_id"])).lower())
                if record.get("selection_score") is not None:
                    client.log_metric(record["run_id"], "candidate_selection_score", float(record["selection_score"]))
        except Exception:
            pass

    # Keep shared support artifacts with the exported candidate.
    reference = train_df.sample(n=min(20_000, len(train_df)), random_state=config.random_seed)
    historical = test_df.sample(n=min(2_500, len(test_df)), random_state=config.random_seed)
    candidate_package = None
    if candidate and config.package_candidate and config.persist_model_artifacts:
        if progress:
            progress("package_candidate", 0.15, "Packaging the selected Candidate and its support artifacts")
        cand_dir = output / candidate["model_id"]
        cand_preds = pd.read_csv(cand_dir / "test_predictions.csv.gz")
        live = build_balanced_live_queue(test_df, cand_preds["probability"].to_numpy(), max_rows=min(2_400, len(test_df)))
        reference.to_csv(cand_dir / "reference_orders.csv.gz", index=False, compression="gzip")
        historical.to_csv(cand_dir / "historical_demo_orders.csv.gz", index=False, compression="gzip")
        live.to_csv(cand_dir / "demo_orders.csv.gz", index=False, compression="gzip")
        (cand_dir / "metadata.json").write_text(json.dumps(candidate, indent=2, default=str))
        candidate_package = output / f"candidate_package__{candidate['model_id']}.zip"
        with zipfile.ZipFile(candidate_package, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in ["metadata.json", "model.pkl", "test_predictions.csv.gz", "reference_orders.csv.gz", "historical_demo_orders.csv.gz", "demo_orders.csv.gz"]:
                path = cand_dir / filename
                if path.exists():
                    zf.write(path, arcname=f"candidate/{filename}")
        if progress:
            progress("package_candidate", 1.0, "Candidate package is ready for registry handoff")

    summary = {
        "experiment_name": experiment_name,
        "dataset_version": config.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.random_seed,
        "training_mode": "gpu_preferred_with_cpu_fallback" if config.use_gpu else "cpu",
        "training_scope": config.training_scope,
        "source_rows": int(config.source_rows) if config.source_rows is not None else int(len(snapshot)),
        "gates": gates,
        "business_policy": policy,
        "dataset_fingerprint": fingerprint,
        "split_manifest": manifest,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "candidate_selection_split": "validation",
        "test_holdout_role": "final_unbiased_evaluation_only",
        "candidate_model_id": candidate.get("model_id") if candidate else None,
        "candidate_package": str(candidate_package) if candidate_package else None,
        "mlflow_enabled": bool(config.enable_mlflow),
        "persist_model_artifacts": bool(config.persist_model_artifacts),
        "models": results,
        "reproducibility": env,
    }
    (output / "suite_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def train_single_model_experiment(
    snapshot: pd.DataFrame,
    config: SuiteConfig,
    family: str,
    hyperparameters: dict | None = None,
    progress=None,
    register=True,
):
    """Train one developer-tuned model, log it to MLflow, and register it as READY.

    Manual experiments never change Candidate or Production automatically.
    """
    from src.models.registry import register_experiment_model

    if family not in DISPLAY_NAMES:
        raise ValueError(f"Unknown model family: {family}")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp().replace("Z", "") + "Z_" + uuid.uuid4().hex[:6]
    experiment_name = config.experiment_name or f"cancellation-risk-manual__{config.dataset_version}__{stamp[:8]}"
    policy = {**DEFAULT_POLICY, **(config.business_policy or {})}
    gates = {**DEFAULT_GATES, **(config.gates or {})}
    env = environment_metadata()
    fingerprint = dataset_fingerprint(snapshot)

    if progress:
        progress("split", 0.10, "Creating the time-based train, validation, and test split")
    train_df, val_df, test_df = temporal_split(snapshot)
    manifest = split_manifest(train_df, val_df, test_df, fingerprint)
    if min(train_df.is_canceled.nunique(), val_df.is_canceled.nunique(), test_df.is_canceled.nunique()) < 2:
        raise ValueError("Each temporal split must contain both outcome classes.")
    Xtr, ytr = train_df[FEATURES], train_df["is_canceled"].astype(int)
    Xv, yv = val_df[FEATURES], val_df["is_canceled"].astype(int)
    Xt, yt = test_df[FEATURES], test_df["is_canceled"].astype(int)
    if progress:
        progress("split", 1.0, f"Split ready: {len(train_df):,} train · {len(val_df):,} validation · {len(test_df):,} test")
        progress("preprocess", 0.10, "Preparing leakage-safe model features")

    prep = _preprocessor(scale_numeric=(family == "logistic_regression"))
    Xtr_t = prep.fit_transform(Xtr, ytr)
    Xv_t = prep.transform(Xv)
    Xt_t = prep.transform(Xt)
    if progress:
        progress("preprocess", 1.0, "Feature matrices ready")

    estimator = build_estimator(family, config.random_seed, hyperparameters)
    display = DISPLAY_NAMES[family]
    run_name = f"{family}__manual__{stamp}"
    mlflow = _mlflow_context(config.mlflow_tracking_uri, experiment_name) if config.enable_mlflow else None
    run_cm = mlflow.start_run(run_name=run_name) if mlflow else None
    run = run_cm.__enter__() if run_cm else None
    started = time.perf_counter()
    try:
        if progress:
            progress("train", 0.10, f"Training {display} with the selected parameters")
        estimator, device = _fit_with_gpu_fallback(estimator, Xtr_t, ytr, family, config.use_gpu)
        if progress:
            progress("train", 1.0, f"Finished training {display} on {device}")
            progress("evaluate", 0.20, "Tuning the decision threshold on validation data")

        pipeline = Pipeline([("prep", prep), ("model", estimator)])
        pv = estimator.predict_proba(Xv_t)[:, 1]
        pt = estimator.predict_proba(Xt_t)[:, 1]
        threshold, _ = best_f1_threshold(yv, pv)
        vm = metrics(yv, pv, threshold)
        tm = metrics(yt, pt, threshold)
        vbm = evaluate_business_policy(yv.to_numpy(), pv, policy)
        tbm = evaluate_business_policy(yt.to_numpy(), pt, policy)
        passed, checks = qualify_model({
            "val_metrics": vm,
            "test_metrics": tm,
            "val_business_metrics": vbm,
            "test_business_metrics": tbm,
            "business_metrics": tbm,
        }, gates)
        duration = time.perf_counter() - started
        if progress:
            progress("evaluate", 1.0, f"Test ROC-AUC {tm['roc_auc']:.3f} · PR-AUC {tm['pr_auc']:.3f} · Brier {tm['brier']:.3f}")

        model_dir = output / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            cloudpickle.dump(pipeline, f)
        pd.DataFrame({
            "order_id": test_df["order_id"].astype(str).to_numpy(),
            "is_canceled": yt.to_numpy(),
            "probability": pt,
        }).to_csv(model_dir / "test_predictions.csv.gz", index=False, compression="gzip")
        reference = train_df.sample(n=min(20_000, len(train_df)), random_state=config.random_seed)
        historical = test_df.sample(n=min(2_500, len(test_df)), random_state=config.random_seed)
        live = build_balanced_live_queue(test_df, pt, max_rows=min(2_400, len(test_df)))
        reference.to_csv(model_dir / "reference_orders.csv.gz", index=False, compression="gzip")
        historical.to_csv(model_dir / "historical_demo_orders.csv.gz", index=False, compression="gzip")
        live.to_csv(model_dir / "demo_orders.csv.gz", index=False, compression="gzip")

        safe_params = {k: v for k, v in estimator.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))}
        record = {
            "model_id": run_name,
            "model_family": family,
            "display_name": display,
            "status": "READY",
            "dataset_version": config.dataset_version,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "run_id": run.info.run_id if run else None,
            "run_role": "manual_experiment",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_device": device,
            "training_scope": config.training_scope,
            "source_rows": int(config.source_rows) if config.source_rows is not None else int(len(snapshot)),
            "random_seed": config.random_seed,
            "hyperparameters": safe_params,
            "threshold": threshold,
            "val_metrics": vm,
            "test_metrics": tm,
            "val_business_metrics": vbm,
            "test_business_metrics": tbm,
            "business_metrics": tbm,
            "qualification": {"passed": passed, "checks": checks, "evidence_split": "validation"},
            "dataset_fingerprint": fingerprint,
            "split_id": manifest["split_id"],
            "split_manifest": manifest,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "duration_seconds": duration,
            "model_artifact": str(model_path),
            "training_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "reproducibility": env,
        }
        (model_dir / "metadata.json").write_text(json.dumps(record, indent=2, default=str))

        if mlflow:
            if progress:
                progress("mlflow", 0.25, "Logging parameters, metrics, model, and reproducibility metadata to MLflow")
            mlflow.log_params({
                "model_family": family,
                "dataset_version": config.dataset_version,
                "random_seed": config.random_seed,
                "training_device": device,
                "training_scope": config.training_scope,
                "source_rows": record["source_rows"],
                "git_commit_sha": env.get("git_commit_sha"),
                "dataset_fingerprint": fingerprint,
                "split_id": manifest["split_id"],
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                **{f"gate__{k}": v for k, v in gates.items()},
                **{f"policy__{k}": v for k, v in policy.items()},
                **{f"model__{k}": v for k, v in safe_params.items()},
            })
            mlflow.log_metrics({f"val_{k}": v for k, v in vm.items()} | {f"test_{k}": v for k, v in tm.items()})
            mlflow.log_metrics({f"val_business_{k}": v for k, v in vbm.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({f"test_business_{k}": v for k, v in tbm.items() if isinstance(v, (int, float))})
            mlflow.set_tags({
                "model_family": family,
                "dataset_version": config.dataset_version,
                "run_role": "manual_experiment",
                "qualification_passed": str(bool(passed)).lower(),
                "selected_candidate": "false",
                "selection_split": "validation",
                "test_holdout_role": "final_unbiased_evaluation",
            })
            _safe_log_model(mlflow, pipeline)
            if progress:
                progress("mlflow", 1.0, "MLflow run saved")

        if register:
            if progress:
                progress("register", 0.35, "Registering the experiment model as Ready")
            record = register_experiment_model(record, model_dir)
            if progress:
                progress("register", 1.0, "Model registered as Ready; Candidate and Production are unchanged")
        return record
    finally:
        if run_cm:
            run_cm.__exit__(None, None, None)
