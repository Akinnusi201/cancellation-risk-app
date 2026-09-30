"""Offline one-time builder for the packaged Pakistan baseline artifacts.

This script is for maintainers. It is not called by Streamlit at startup.
"""
import json
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE
from src.data.aggregation import aggregate_to_orders, first_existing
from src.data.validation import validate_raw
from src.config import ORDER_ID_CANDIDATES
from src.features.build_features import get_model_frame
from src.models.evaluate import best_f1_threshold, metrics
from src.monitoring.plots import save_evaluation_plots


def temporal_split(df, train_frac=.70, val_frac=.15):
    s = df.sort_values("created_at").reset_index(drop=True)
    n = len(s)
    i = int(n * train_frac)
    j = int(n * (train_frac + val_frac))
    return s.iloc[:i].copy(), s.iloc[i:j].copy(), s.iloc[j:].copy()


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])


def main(raw_csv):
    raw_csv = Path(raw_csv)
    print(f"Reading {raw_csv} ...", flush=True)
    raw = pd.read_csv(raw_csv, low_memory=False).dropna(how="all").reset_index(drop=True)
    print(f"Nonblank source rows: {len(raw):,}", flush=True)
    valid, quarantine, checks, fatal = validate_raw(raw)
    if fatal:
        raise RuntimeError("Required columns missing")

    order_col = first_existing(valid.columns, ORDER_ID_CANDIDATES)
    inconsistent = set()
    if order_col:
        counts = valid.groupby(order_col, dropna=False)["status"].nunique()
        inconsistent = set(counts[counts > 1].index.astype(str))
        if inconsistent:
            mask = valid[order_col].astype("string").isin(inconsistent)
            extra = valid.loc[mask].copy()
            extra["quarantine_reason"] = "inconsistent_order_status;"
            quarantine = pd.concat([quarantine, extra], ignore_index=True, sort=False)
            valid = valid.loc[~mask].copy()
    checks.append(("order_status_consistency", "PASS" if not inconsistent else "WARN", len(inconsistent), "Orders with multiple final statuses quarantined"))

    print(f"Valid item rows: {len(valid):,}; quarantined: {len(quarantine):,}", flush=True)
    orders = aggregate_to_orders(valid)
    print(f"Order rows: {len(orders):,}; canceled: {int(orders.is_canceled.sum()):,}; complete: {int((1-orders.is_canceled).sum()):,}", flush=True)

    seed_dir = ROOT / "data" / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    orders.to_csv(seed_dir / "pakistan_orders_v1.csv.gz", index=False, compression="gzip")
    quarantine.head(5000).to_csv(seed_dir / "pakistan_seed_quarantine_sample.csv.gz", index=False, compression="gzip")
    (seed_dir / "pakistan_seed_quarantine.csv").write_text("quarantine_reason\n")
    (seed_dir / "pakistan_seed_validation.json").write_text(json.dumps({
        "version": "pakistan_seed_v1",
        "raw_rows": len(raw),
        "valid_item_rows": len(valid),
        "quarantined_item_rows": len(quarantine),
        "order_rows": len(orders),
        "complete_orders": int((orders.is_canceled == 0).sum()),
        "canceled_orders": int((orders.is_canceled == 1).sum()),
        "eligible_complete_canceled_orders_before_quality_quarantine": int(raw.loc[raw["status"].astype(str).str.lower().str.strip().isin(["complete", "canceled"]), "increment_id"].astype("string").nunique()),
        "checks": [dict(check_name=a, status=b, affected_rows=int(c), details=d) for a,b,c,d in checks],
    }, indent=2, default=str))

    train_df, val_df, test_df = temporal_split(orders)
    Xtr, ytr = get_model_frame(train_df)
    Xv, yv = get_model_frame(val_df)
    Xt, yt = get_model_frame(test_df)
    print(f"Temporal split: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}", flush=True)

    model = Pipeline([
        ("prep", preprocessor()),
        ("model", LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31, max_depth=-1,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )),
    ])
    print("Training LightGBM baseline ...", flush=True)
    model.fit(Xtr, ytr)
    pv = model.predict_proba(Xv)[:,1]
    threshold, _ = best_f1_threshold(yv, pv)
    vm = metrics(yv, pv, threshold)
    pt = model.predict_proba(Xt)[:,1]
    tm = metrics(yt, pt, threshold)
    print("Validation", vm, flush=True)
    print("Test", tm, flush=True)
    print("Threshold", threshold, flush=True)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "production_model.pkl", "wb") as f:
        cloudpickle.dump(model, f)

    eval_dir = artifacts / "production_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_plots(yt, pt, eval_dir, "test")
    names = model.named_steps["prep"].get_feature_names_out()
    imp = model.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).to_csv(eval_dir / "feature_importance.csv", index=False)

    # Reference sample is drawn from training history; demo queue comes from the final holdout.
    reference = train_df.sample(n=min(20000, len(train_df)), random_state=RANDOM_STATE)
    demo = test_df.sample(n=min(2500, len(test_df)), random_state=RANDOM_STATE)
    reference.to_csv(artifacts / "reference_orders.csv.gz", index=False, compression="gzip")
    demo.to_csv(artifacts / "demo_orders.csv.gz", index=False, compression="gzip")

    (artifacts / "business_policy.json").write_text(json.dumps({
        "avoidable_fulfillment_cost": 2000.0,
        "intervention_effectiveness": 0.55,
        "intervention_cost": 150.0,
        "false_positive_friction_cost": 50.0,
    }, indent=2))

    (artifacts / "active_model.json").write_text(json.dumps({
        "model_name": "lightgbm",
        "model_version": "seed-1",
        "dataset_version": "pakistan_seed_v1",
        "threshold": threshold,
        "val_metrics": vm,
        "test_metrics": tm,
        "split_cancellation_prevalence": {
            "train": float(ytr.mean()), "validation": float(yv.mean()), "test": float(yt.mean())
        },
        "model_artifact": "artifacts/production_model.pkl",
        "evaluation_dir": "artifacts/production_evaluation",
        "source": "packaged_seed",
        "training_note": "Trained once offline from the packaged Pakistan order-level baseline. Streamlit startup does not retrain.",
    }, indent=2))

    print("Artifacts written.", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/build_seed_artifacts.py /path/to/Pakistan.csv")
    main(sys.argv[1])
