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

## Authentication

Credentials are read from Streamlit secrets or equivalent environment variables. **Do not commit real passwords.**

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` for local development, or add the same TOML in Streamlit Community Cloud → App settings → Secrets:

```toml
[auth.manager]
username = "operations"
password = "<manager-password>"

[auth.developer]
username = "developer"
password = "<developer-password>"
```

Equivalent environment variables are:

```text
AUTH_MANAGER_USERNAME
AUTH_MANAGER_PASSWORD
AUTH_DEVELOPER_USERNAME
AUTH_DEVELOPER_PASSWORD
```

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
train LightGBM candidate
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

Training a candidate never changes production automatically.

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
4. Add the authentication secrets shown above.
5. Deploy.

The app should open to the login page and be prediction-ready immediately after authentication.

## Rebuilding the packaged baseline

The one-time maintainer script is included for reproducibility:

```bash
python scripts/build_seed_artifacts.py "/path/to/Pakistan Largest Ecommerce Dataset.csv"
```

This script is not called by Streamlit startup.
