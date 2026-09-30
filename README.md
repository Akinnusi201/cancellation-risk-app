# Profit-Aware E-Commerce Order Cancellation Risk

Group 10 Streamlit prototype for **DataOps + ModelOps + DevOps** around cancellation-risk scoring.

## What changed in this version

The deployed app is now prediction-ready at startup. Operations users no longer upload the Pakistan dataset or retrain a model before scoring orders.

The repository ships these baseline artifacts:

- `data/seed/pakistan_orders_v1.csv.gz` — cleaned, order-level Pakistan baseline used for reproducible developer retraining.
- `artifacts/production_model.pkl` — pretrained LightGBM production pipeline.
- `artifacts/active_model.json` — production model version, dataset lineage, threshold, and metrics.
- `artifacts/reference_orders.csv.gz` — reference sample used for lightweight prediction explanations.
- `artifacts/demo_orders.csv.gz` — held-out historical orders used for the operations simulation.
- `artifacts/production_evaluation/` — ROC, PR, calibration, and feature-importance artifacts.

The original 100+ MB raw CSV is intentionally **not** committed. GitHub's normal single-file limit makes the raw source a poor deployment artifact, and production inference does not need it.

## Role-based application

Navigation is built with `st.navigation` and only exposes pages permitted for the signed-in role.

### Operations Manager

- Operations Dashboard
- Score Order
  - incoming-order simulation
  - manual order scoring
  - batch CSV scoring
- Decision History

Operations users can change economic assumptions for a scoring session, but they cannot ingest datasets, train models, or promote candidates.

### Developer

- Developer Dashboard
- DataOps
- ModelOps
- Model Monitoring
- MLflow Experiments
- System Status

## Role-based landing page

The app opens on a passwordless role-selection page for the class demonstration. Choose **Operations Manager** or **Developer**.

- Operations Manager can only access Operations Dashboard, Score Order, and Decision History.
- Developer can only access Developer Dashboard, DataOps, ModelOps, Model Monitoring, MLflow Experiments, and System Status.
- Role-specific pages are registered dynamically with `st.navigation`, and page files live under `views/` instead of Streamlit's special `pages/` directory. This prevents hidden developer pages from appearing in Operations navigation.

For a real production deployment, replace the demo role selector with SSO/OIDC or another authenticated identity provider.

## Production startup path

```text
Streamlit starts
    ↓
register packaged seed dataset metadata (fresh runtime only)
    ↓
load artifacts/production_model.pkl
    ↓
operations scoring is ready
```

**No model training occurs at startup.**

## DataOps path

The packaged Pakistan dataset is the baseline dataset version. Developers upload only new item-level batches.

```text
new CSV batch
    ↓
SHA-256 duplicate protection
    ↓
schema / date / range / target validation
    ↓
quarantine invalid records
    ↓
order-status and existing-order checks
    ↓
item → order aggregation
    ↓
recompute leakage-safe prior customer history across the cumulative timeline
    ↓
new immutable dataset version
```

DataOps stops there. It does not retrain or promote a model.

`demo_batch.csv` is packaged as a safe new batch with non-overlapping order IDs so the DataOps workflow can be demonstrated without replacing the baseline.

## ModelOps path

```text
select dataset version
    ↓
train Logistic Regression baseline
    ↓
train LightGBM candidate on the same temporal split
    ↓
validation threshold selection
    ↓
held-out evaluation + MLflow logging
    ↓
review candidate vs production
    ↓
explicit Promote Candidate action
    ↓
replace packaged/runtime production artifact
```

Training a candidate never changes production automatically. Evaluation includes ROC-AUC, PR-AUC, Brier score, precision, recall, F1, and recall at fixed precision.

## Monitoring and business evaluation

The developer monitoring page now measures runtime model latency, intervention/high-risk rates, pipeline success, prediction drift, input-feature drift, and labeled holdout business impact. A packaged holdout probability file supports aggregate estimated savings and sensitivity analysis across fulfillment-cost, intervention-effectiveness, and intervention-cost assumptions without retraining the model.

## DevOps automation

- `.github/workflows/ci.yml` runs the full test suite, smoke-tests the packaged production model, and builds the Docker image after tests pass.
- `Dockerfile` provides a reproducible Python 3.12 container with a Streamlit health check.
- Streamlit Community Cloud can continue to deploy from the GitHub `main` branch, while CI validates each push first.

## Profit-aware decision rule

The model estimates cancellation probability. The decision layer then applies configurable business assumptions:

```text
Expected Avoidable Cost
    = P(Cancellation) × Avoidable Fulfillment Cost × Intervention Effectiveness

Net Expected Savings
    = Expected Avoidable Cost
      - Intervention Cost
      - Expected False-Intervention Friction
```

The app recommends **Hold for Verification** when net expected savings are positive. This keeps the probability model separate from business policy.

## Dataset notes

The Kaggle CSV physically contains 1,048,575 lines when read as rows, but only **584,524 nonblank transaction rows**. The long blank tail is dropped before validation because blank physical rows are not business records.

Among the nonblank source rows, the documented complete/canceled target contains 318,158 unique orders. The packaged model-ready baseline contains **318,135 orders** after the project's validation rules quarantine a small number of invalid item records instead of silently fixing them. The exact counts and checks are saved in `data/seed/pakistan_seed_validation.json`.

The final temporal holdout has a much higher cancellation prevalence than earlier periods, so the monitoring page should be used to discuss temporal drift and why calibration/business metrics matter in addition to ROC-AUC.

## MLflow

Runtime candidate and manual experiment training is logged to MLflow. The packaged seed production model is deliberately independent of runtime MLflow state so a fresh Streamlit deployment does not need an MLflow database to perform inference.

By default:

- MLflow backend: local SQLite
- MLflow artifacts: `artifacts/mlflow/`
- DuckDB metadata/audit store: `cancellation.duckdb`

These runtime stores are fine for a classroom prototype. For a persistent production deployment, move them to persistent external services.

## Streamlit Community Cloud deployment

1. Push the project to GitHub.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Use Python 3.12.
4. Deploy.

The app should open to the role-selection login page and be prediction-ready immediately after choosing a workspace.

## Rebuilding the packaged baseline

The one-time maintainer script is included for reproducibility:

```bash
python scripts/build_seed_artifacts.py "/path/to/Pakistan Largest Ecommerce Dataset.csv"
```

This script is not called by Streamlit startup.
