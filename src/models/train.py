import json
import shutil
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cloudpickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
)
from src.database.duckdb_manager import log_event
from src.features.build_features import get_model_frame
from src.models.evaluate import best_f1_threshold, metrics
from src.models.registry import save_production_model
from src.monitoring.plots import save_evaluation_plots

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

CANDIDATE_DIR = ARTIFACT_DIR / "candidates"
BASELINE_DIR = ARTIFACT_DIR / "baselines"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)


def temporal_split(df, train_frac=.70, val_frac=.15):
    s = df.sort_values("created_at").reset_index(drop=True)
    n = len(s)
    i = int(n * train_frac)
    j = int(n * (train_frac + val_frac))
    return s.iloc[:i].copy(), s.iloc[i:j].copy(), s.iloc[j:].copy()


def preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def logistic_preprocessor():
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def lightgbm_model(**overrides):
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is not installed. Install dependencies from requirements.txt.")
    params = dict(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


def logistic_model():
    return LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def native_feature_importance(pipe):
    try:
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        names = prep.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = abs(model.coef_[0])
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name("cancellation-risk")
    if exp is None:
        kwargs = {}
        if str(MLFLOW_TRACKING_URI).startswith(("sqlite", "file")):
            kwargs["artifact_location"] = MLFLOW_ARTIFACT_ROOT.resolve().as_uri()
        mlflow.create_experiment("cancellation-risk", **kwargs)
    mlflow.set_experiment("cancellation-risk")


def _fit_and_log(
    snapshot,
    dataset_version,
    estimator,
    run_name,
    run_source,
    model_type,
    progress_callback=None,
):
    def progress(step, message):
        if progress_callback:
            progress_callback(step, message)

    display_name = "Logistic Regression" if model_type == "logistic_regression" else "LightGBM"
    progress(5, f"Preparing MLflow experiment for {display_name}")
    _setup_mlflow()
    progress(10, "Creating temporal train / validation / test splits")
    train_df, val_df, test_df = temporal_split(snapshot)
    Xtr, ytr = get_model_frame(train_df)
    Xv, yv = get_model_frame(val_df)
    Xt, yt = get_model_frame(test_df)
    if min(ytr.nunique(), yv.nunique(), yt.nunique()) < 2:
        raise ValueError("Each temporal split must contain both classes. Add more data before training.")

    with mlflow.start_run(run_name=run_name) as run:
        pipe = Pipeline([("prep", logistic_preprocessor() if model_type == "logistic_regression" else preprocessor()), ("model", estimator)])
        progress(20, f"Training {display_name}")
        pipe.fit(Xtr, ytr)
        progress(65, "Selecting the validation F1 threshold")
        pv = pipe.predict_proba(Xv)[:, 1]
        threshold, _ = best_f1_threshold(yv, pv)
        vm = metrics(yv, pv, threshold)
        progress(72, "Evaluating the held-out test set")
        pt = pipe.predict_proba(Xt)[:, 1]
        tm = metrics(yt, pt, threshold)

        params = {
            f"model__{k}": v
            for k, v in estimator.get_params().items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        params.update({
            "dataset_version": dataset_version,
            "model_type": model_type,
            "run_source": run_source,
            "decision_threshold": threshold,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
        })
        mlflow.log_params(params)
        mlflow.log_metrics(
            {f"val_{k}": v for k, v in vm.items()}
            | {f"test_{k}": v for k, v in tm.items()}
        )

        evaluation_dir = ARTIFACT_DIR / dataset_version / model_type / run.info.run_id[:8]
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        progress(80, "Generating evaluation artifacts")
        plots = save_evaluation_plots(yt, pt, evaluation_dir, "test")
        fi = native_feature_importance(pipe)
        fi_path = evaluation_dir / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        for p in [*plots.values(), str(fi_path)]:
            mlflow.log_artifact(p)
        mlflow.sklearn.log_model(
            pipe,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        progress(88, f"{display_name} model and metrics logged")

    return {
        "model_name": model_type,
        "run_id": run.info.run_id,
        "pipeline": pipe,
        "threshold": threshold,
        "val": vm,
        "test": tm,
        "feature_importance": fi,
        "train_df": train_df,
        "test_df": test_df,
        "test_probabilities": pt,
        "split_cancellation_prevalence": {
            "train": float(ytr.mean()),
            "validation": float(yv.mean()),
            "test": float(yt.mean()),
        },
        "evaluation_dir": evaluation_dir,
    }


def train_logistic_baseline(snapshot, dataset_version, progress_callback=None):
    start = time.perf_counter()
    result = _fit_and_log(
        snapshot,
        dataset_version,
        logistic_model(),
        run_name=f"baseline_logistic_{dataset_version}_{uuid.uuid4().hex[:6]}",
        run_source="baseline",
        model_type="logistic_regression",
        progress_callback=progress_callback,
    )
    baseline_id = f"base_{result['run_id'][:10]}"
    metadata = {
        "baseline_id": baseline_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "logistic_regression",
        "run_id": result["run_id"],
        "dataset_version": dataset_version,
        "threshold": result["threshold"],
        "val_metrics": result["val"],
        "test_metrics": result["test"],
        "evaluation_dir": str(result["evaluation_dir"]),
        "duration_seconds": time.perf_counter() - start,
    }
    out = BASELINE_DIR / baseline_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    log_event(
        f"event_{uuid.uuid4().hex[:10]}",
        "MODEL_BASELINE",
        "SUCCESS",
        f"Trained Logistic Regression baseline {baseline_id} on {dataset_version}",
        {"run_id": result["run_id"], "dataset_version": dataset_version, "duration_seconds": metadata["duration_seconds"]},
    )
    return metadata


def train_candidate(
    snapshot,
    dataset_version,
    progress_callback=None,
    fast_mode=False,
    include_baseline=True,
    **model_params,
):
    started = time.perf_counter()
    baseline = None
    try:
        if include_baseline:
            def baseline_progress(percent, message):
                if progress_callback:
                    progress_callback(int(percent * 0.35), f"Baseline: {message}")
            baseline = train_logistic_baseline(snapshot, dataset_version, progress_callback=baseline_progress)

        if fast_mode and "n_estimators" not in model_params:
            model_params["n_estimators"] = 80
        estimator = lightgbm_model(**model_params)

        def candidate_progress(percent, message):
            if progress_callback:
                if include_baseline:
                    progress_callback(35 + int(percent * 0.65), f"Candidate: {message}")
                else:
                    progress_callback(percent, message)

        result = _fit_and_log(
            snapshot,
            dataset_version,
            estimator,
            run_name=f"candidate_{dataset_version}_{uuid.uuid4().hex[:6]}",
            run_source="candidate",
            model_type="lightgbm",
            progress_callback=candidate_progress,
        )

        candidate_id = f"cand_{result['run_id'][:10]}"
        out = CANDIDATE_DIR / candidate_id
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / "model.pkl"
        holdout_path = out / "holdout.parquet"
        predictions_path = out / "test_predictions.csv.gz"
        reference_path = out / "reference_orders.csv.gz"
        demo_path = out / "demo_orders.csv.gz"
        with open(model_path, "wb") as f:
            cloudpickle.dump(result["pipeline"], f)
        result["test_df"].to_parquet(holdout_path, index=False)
        result["train_df"].sample(
            n=min(20000, len(result["train_df"])), random_state=RANDOM_STATE
        ).to_csv(reference_path, index=False, compression="gzip")
        result["test_df"].sample(
            n=min(2500, len(result["test_df"])), random_state=RANDOM_STATE
        ).to_csv(demo_path, index=False, compression="gzip")
        pd.DataFrame({
            "order_id": result["test_df"]["order_id"].astype(str).to_numpy(),
            "is_canceled": result["test_df"]["is_canceled"].astype(int).to_numpy(),
            "probability": result["test_probabilities"],
        }).to_csv(predictions_path, index=False, compression="gzip")

        metadata = {
            "candidate_id": candidate_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": "lightgbm",
            "run_id": result["run_id"],
            "dataset_version": dataset_version,
            "threshold": result["threshold"],
            "val_metrics": result["val"],
            "test_metrics": result["test"],
            "model_artifact": str(model_path),
            "holdout_path": str(holdout_path),
            "test_predictions_path": str(predictions_path),
            "reference_path": str(reference_path),
            "demo_path": str(demo_path),
            "evaluation_dir": str(result["evaluation_dir"]),
            "model_params": {
                k: v
                for k, v in estimator.get_params().items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
            "baseline": baseline,
            "split_cancellation_prevalence": result["split_cancellation_prevalence"],
            "reference_prediction_summary": {
                "holdout_rows": int(len(result["test_df"])),
                "mean_probability": float(pd.Series(result["test_probabilities"]).mean()),
                "median_probability": float(pd.Series(result["test_probabilities"]).median()),
                "p95_probability": float(pd.Series(result["test_probabilities"]).quantile(0.95)),
            },
            "duration_seconds": time.perf_counter() - started,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        log_event(
            f"event_{uuid.uuid4().hex[:10]}",
            "MODEL_CANDIDATE",
            "SUCCESS",
            f"Trained candidate {candidate_id} on {dataset_version}",
            {
                "run_id": result["run_id"],
                "dataset_version": dataset_version,
                "duration_seconds": metadata["duration_seconds"],
                "baseline_run_id": baseline.get("run_id") if baseline else None,
            },
        )
        if progress_callback:
            progress_callback(100, "Experiment set complete; production unchanged")
        return metadata
    except Exception as exc:
        log_event(
            f"event_{uuid.uuid4().hex[:10]}",
            "MODEL_CANDIDATE",
            "FAILED",
            f"Candidate training failed on {dataset_version}: {exc}",
            {"dataset_version": dataset_version, "duration_seconds": time.perf_counter() - started},
        )
        raise


def list_candidates():
    rows = []
    for p in sorted(CANDIDATE_DIR.glob("*/metadata.json"), reverse=True):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    return rows


def promote_candidate(candidate_id):
    metadata_path = CANDIDATE_DIR / candidate_id / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Candidate {candidate_id} was not found.")
    candidate = json.loads(metadata_path.read_text())
    with open(candidate["model_artifact"], "rb") as f:
        model = cloudpickle.load(f)

    model_version = candidate_id
    try:
        _setup_mlflow()
        mv = mlflow.register_model(f"runs:/{candidate['run_id']}/model", REGISTERED_MODEL_NAME)
        model_version = str(mv.version)
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "dataset_version", candidate["dataset_version"])
        client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "decision_threshold", str(candidate["threshold"]))
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", model_version)
    except Exception:
        # Local production artifact remains sufficient for inference if registry operations are unavailable.
        pass

    prod_eval = ARTIFACT_DIR / "production_evaluation"
    if prod_eval.exists():
        shutil.rmtree(prod_eval)
    src_eval = Path(candidate["evaluation_dir"])
    if src_eval.exists():
        shutil.copytree(src_eval, prod_eval)
    else:
        prod_eval.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate["holdout_path"], ARTIFACT_DIR / "latest_holdout.parquet")
    predictions_path = candidate.get("test_predictions_path")
    if predictions_path and Path(predictions_path).exists():
        shutil.copy2(predictions_path, prod_eval / "test_predictions.csv.gz")
    reference_path = candidate.get("reference_path")
    if reference_path and Path(reference_path).exists():
        shutil.copy2(reference_path, ARTIFACT_DIR / "reference_orders.csv.gz")
    demo_path = candidate.get("demo_path")
    if demo_path and Path(demo_path).exists():
        shutil.copy2(demo_path, ARTIFACT_DIR / "demo_orders.csv.gz")

    baseline = candidate.get("baseline") or {}
    if baseline:
        comparison = pd.DataFrame([
            {
                "Model": "Logistic Regression",
                "Threshold": baseline.get("threshold"),
                "Validation ROC-AUC": baseline.get("val_metrics", {}).get("roc_auc"),
                "Validation PR-AUC": baseline.get("val_metrics", {}).get("pr_auc"),
                "Validation Brier": baseline.get("val_metrics", {}).get("brier"),
                "Test ROC-AUC": baseline.get("test_metrics", {}).get("roc_auc"),
                "Test PR-AUC": baseline.get("test_metrics", {}).get("pr_auc"),
                "Test Brier": baseline.get("test_metrics", {}).get("brier"),
                "Test F1": baseline.get("test_metrics", {}).get("f1"),
                "Test Precision": baseline.get("test_metrics", {}).get("precision"),
                "Test Recall": baseline.get("test_metrics", {}).get("recall"),
                "Recall at 80% Precision": baseline.get("test_metrics", {}).get("recall_at_80_precision"),
                "Recall at 90% Precision": baseline.get("test_metrics", {}).get("recall_at_90_precision"),
            },
            {
                "Model": "LightGBM Production",
                "Threshold": candidate.get("threshold"),
                "Validation ROC-AUC": candidate.get("val_metrics", {}).get("roc_auc"),
                "Validation PR-AUC": candidate.get("val_metrics", {}).get("pr_auc"),
                "Validation Brier": candidate.get("val_metrics", {}).get("brier"),
                "Test ROC-AUC": candidate.get("test_metrics", {}).get("roc_auc"),
                "Test PR-AUC": candidate.get("test_metrics", {}).get("pr_auc"),
                "Test Brier": candidate.get("test_metrics", {}).get("brier"),
                "Test F1": candidate.get("test_metrics", {}).get("f1"),
                "Test Precision": candidate.get("test_metrics", {}).get("precision"),
                "Test Recall": candidate.get("test_metrics", {}).get("recall"),
                "Recall at 80% Precision": candidate.get("test_metrics", {}).get("recall_at_80_precision"),
                "Recall at 90% Precision": candidate.get("test_metrics", {}).get("recall_at_90_precision"),
            },
        ])
        comparison.to_csv(prod_eval / "baseline_comparison.csv", index=False)
        (prod_eval / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2, default=str))

    active = save_production_model(model, {
        "model_name": "lightgbm",
        "model_version": model_version,
        "run_id": candidate["run_id"],
        "dataset_version": candidate["dataset_version"],
        "threshold": candidate["threshold"],
        "val_metrics": candidate["val_metrics"],
        "test_metrics": candidate["test_metrics"],
        "evaluation_dir": str(prod_eval.relative_to(ARTIFACT_DIR.parent)),
        "source": "promoted_candidate",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "baseline": candidate.get("baseline"),
        "split_cancellation_prevalence": candidate.get("split_cancellation_prevalence", {}),
        "reference_prediction_summary": candidate.get("reference_prediction_summary", {}),
    })
    log_event(
        f"event_{uuid.uuid4().hex[:10]}",
        "MODEL_PROMOTION",
        "SUCCESS",
        f"Promoted {candidate_id} as production model {model_version}",
        {"dataset_version": candidate["dataset_version"], "run_id": candidate["run_id"]},
    )
    return active


def train_all(snapshot, dataset_version, promote=True, run_source="automatic_batch", progress_callback=None, fast_mode=False):
    """Backward-compatible wrapper. New UI uses explicit candidate training + promotion."""
    candidate = train_candidate(snapshot, dataset_version, progress_callback=progress_callback, fast_mode=fast_mode)
    active = promote_candidate(candidate["candidate_id"]) if promote else None
    comparison = pd.DataFrame([{
        "model": "LightGBM",
        "run_id": candidate["run_id"],
        "threshold": candidate["threshold"],
        **{f"val_{k}": v for k, v in candidate["val_metrics"].items()},
        **{f"test_{k}": v for k, v in candidate["test_metrics"].items()},
    }])
    return {
        "winner": candidate,
        "model_version": active.get("model_version") if active else None,
        "comparison": comparison,
        "test_df": pd.read_parquet(candidate["holdout_path"]),
    }


def run_manual_experiment(snapshot, dataset_version, experiment_name="manual_lightgbm", progress_callback=None, **params):
    estimator = lightgbm_model(**params)
    result = _fit_and_log(
        snapshot,
        dataset_version,
        estimator,
        experiment_name,
        "manual",
        "lightgbm",
        progress_callback=progress_callback,
    )
    return {"run_id": result["run_id"], "threshold": result["threshold"], "val": result["val"], "test": result["test"]}
