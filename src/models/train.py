import json, uuid, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import (NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE, ARTIFACT_DIR,
                        MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, MLFLOW_ARTIFACT_ROOT)
from src.features.build_features import get_model_frame
from src.models.evaluate import best_f1_threshold, metrics
from src.monitoring.plots import save_evaluation_plots
from src.database.duckdb_manager import log_event
warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def temporal_split(df, train_frac=.70, val_frac=.15):
    s = df.sort_values("created_at").reset_index(drop=True)
    n = len(s); i = int(n * train_frac); j = int(n * (train_frac + val_frac))
    return s.iloc[:i].copy(), s.iloc[i:j].copy(), s.iloc[j:].copy()


def preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def lightgbm_model(**overrides):
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is not installed. Install dependencies from requirements.txt.")
    params = dict(
        n_estimators=220,
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


def native_feature_importance(pipe):
    """Fast LightGBM importance; avoids expensive permutation importance on every upload."""
    try:
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        names = prep.get_feature_names_out()
        values = model.feature_importances_
        return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name("cancellation-risk")
    if exp is None:
        kwargs = {}
        # File artifact roots work for local/hosted SQLite. Remote MLflow servers manage their own artifact location.
        if str(MLFLOW_TRACKING_URI).startswith("sqlite") or str(MLFLOW_TRACKING_URI).startswith("file"):
            kwargs["artifact_location"] = MLFLOW_ARTIFACT_ROOT.resolve().as_uri()
        mlflow.create_experiment("cancellation-risk", **kwargs)
    mlflow.set_experiment("cancellation-risk")


def _fit_and_log(snapshot: pd.DataFrame, dataset_version: str, estimator, run_name: str, run_source: str):
    _setup_mlflow()
    train_df, val_df, test_df = temporal_split(snapshot)
    Xtr, ytr = get_model_frame(train_df); Xv, yv = get_model_frame(val_df); Xt, yt = get_model_frame(test_df)
    if min(ytr.nunique(), yv.nunique(), yt.nunique()) < 2:
        raise ValueError("Each temporal split must contain both classes. Add more data before training.")

    with mlflow.start_run(run_name=run_name) as run:
        pipe = Pipeline([("prep", preprocessor()), ("model", estimator)])
        pipe.fit(Xtr, ytr)
        pv = pipe.predict_proba(Xv)[:, 1]
        threshold, _ = best_f1_threshold(yv, pv)
        vm = metrics(yv, pv, threshold)
        pt = pipe.predict_proba(Xt)[:, 1]
        tm = metrics(yt, pt, threshold)

        params = {f"model__{k}": v for k, v in estimator.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))}
        params.update({
            "dataset_version": dataset_version,
            "model_type": "lightgbm",
            "run_source": run_source,
            "decision_threshold": threshold,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
        })
        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": v for k, v in vm.items()} | {f"test_{k}": v for k, v in tm.items()})

        out = ARTIFACT_DIR / dataset_version / "lightgbm" / run.info.run_id[:8]
        out.mkdir(parents=True, exist_ok=True)
        plots = save_evaluation_plots(yt, pt, out, "test")
        fi = native_feature_importance(pipe)
        fi_path = out / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        for p in [*plots.values(), str(fi_path)]:
            mlflow.log_artifact(p)
        mlflow.sklearn.log_model(pipe, "model")

    return {
        "model_name": "lightgbm", "run_id": run.info.run_id, "pipeline": pipe,
        "threshold": threshold, "val": vm, "test": tm,
        "feature_importance": fi, "test_df": test_df,
    }


def train_all(snapshot: pd.DataFrame, dataset_version: str, promote=True, run_source="automatic_batch"):
    """Kept for backward compatibility; now trains exactly one LightGBM model."""
    estimator = lightgbm_model()
    result = _fit_and_log(
        snapshot, dataset_version, estimator,
        run_name=f"{run_source}_{dataset_version}_lightgbm_{uuid.uuid4().hex[:6]}",
        run_source=run_source,
    )

    model_version = None
    if promote:
        model_uri = f"runs:/{result['run_id']}/model"
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        model_version = str(mv.version)
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "dataset_version", dataset_version)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "decision_threshold", str(result["threshold"]))
        client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version, "model_type", "lightgbm")
        try:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", model_version)
        except Exception:
            pass
        (ARTIFACT_DIR / "active_model.json").write_text(json.dumps({
            "model_name": "lightgbm", "model_version": model_version,
            "run_id": result["run_id"], "dataset_version": dataset_version,
            "threshold": result["threshold"],
        }, indent=2))
        holdout_path = ARTIFACT_DIR / "latest_holdout.parquet"
        result["test_df"].to_parquet(holdout_path, index=False)
        log_event(
            f"event_{uuid.uuid4().hex[:10]}", "MODEL_PROMOTION", "SUCCESS",
            f"Promoted LightGBM v{model_version}",
            {"dataset_version": dataset_version, "threshold": result["threshold"]},
        )

    comparison = pd.DataFrame([{
        "model": "LightGBM",
        "run_id": result["run_id"],
        "threshold": result["threshold"],
        **{f"val_{k}": v for k, v in result["val"].items()},
        **{f"test_{k}": v for k, v in result["test"].items()},
    }])
    return {"winner": result, "model_version": model_version, "comparison": comparison, "test_df": result["test_df"]}


def run_manual_experiment(snapshot: pd.DataFrame, dataset_version: str, experiment_name="manual_lightgbm", **params):
    estimator = lightgbm_model(**params)
    result = _fit_and_log(snapshot, dataset_version, estimator, experiment_name, "manual")
    return {"run_id": result["run_id"], "threshold": result["threshold"], "val": result["val"], "test": result["test"]}
