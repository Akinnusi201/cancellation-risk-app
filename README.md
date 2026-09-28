# Profit-Aware E-Commerce Order Cancellation Risk

Group 10 web prototype: automated batch DataOps, one fast LightGBM training pipeline, MLflow experiment/model versioning, and a manager-facing Streamlit cancellation-risk queue.

## Current prototype scope

- **One production model only: LightGBM.** This keeps automatic retraining practical for the demo.
- Batch CSV ingestion with atomic SHA-256 duplicate protection.
- Re-uploading the same file is a safe no-op. It does not throw a DuckDB unique-constraint error and does not retrain.
- Automated schema/range/date/target/duplicate validation.
- Safe cleaning and quarantine of records that should not be guessed/fixed.
- Deterministic item-to-order aggregation and leakage-safe prior customer cancellation history.
- Cumulative timestamped Parquet dataset versions plus downloadable validation/quarantine artifacts.
- Temporal 70/15/15 train/validation/test split.
- Validation-set F1 threshold optimization.
- MLflow run tracking, artifacts, model registry, and model versions.
- Manual **LightGBM** experiments with adjustable trees, learning rate, and leaves. Manual experiments never auto-promote.
- Held-out test orders simulated as incoming manager orders.
- Approve/Hold decisions, actual historical outcome reveal, and audit history.

## Deploy as an internet web app

The project is prepared for **Streamlit Community Cloud**. Community Cloud deploys directly from a GitHub repository and gives the app a public `streamlit.app` URL.

### 1. Put this folder in a GitHub repository

The repository root should contain:

```text
app.py
pages/
src/
requirements.txt
packages.txt
.streamlit/config.toml
```

Do **not** commit the large Pakistan CSV to GitHub. Upload it through the Data Pipeline page after the app is deployed.

### 2. Deploy on Streamlit Community Cloud

1. Sign in at `share.streamlit.io` with GitHub.
2. Choose **Create app**.
3. Select the repository and branch.
4. Set the entrypoint to `app.py`.
5. Choose Python **3.12** in Advanced settings.
6. Deploy.

Community Cloud reads `requirements.txt`, `packages.txt`, and `.streamlit/config.toml` automatically.

### 3. Use the hosted app

Open the generated `https://<your-app>.streamlit.app` address, go to **Data Pipeline**, and upload the Pakistan e-commerce CSV. A successful new batch automatically:

```text
validates → cleans → quarantines → versions → trains LightGBM → logs MLflow → registers/promotes model
```

## MLflow

You do not need to run a second MLflow web server for this version. The Streamlit pages read MLflow runs directly and display the metrics, experiment history, model versions, ROC curve, PR curve, calibration plot, and feature importance.

By default, MLflow uses a SQLite backend inside the hosted app. `MLFLOW_TRACKING_URI` can later be supplied as an environment variable/secret to point the exact same code at a persistent remote MLflow server.

## Important cloud-storage note

Streamlit Community Cloud does not guarantee persistence of files created at runtime. This is acceptable for a classroom demonstration, but a reboot may clear uploaded datasets, DuckDB history, and local MLflow files. For a longer-lived production version, move the database/artifact store to persistent cloud services.

## Modeling safeguards

- Threshold selection happens on validation data, never the test set.
- The newest temporal holdout is reserved for final metrics and the incoming-order simulation.
- Outcome-derived columns such as `status`, `BI Status`, and `is_canceled` are blocked from model features.
- Customer cancellation history only uses prior orders.
- Identifier columns are normalized as strings before aggregation and Parquet persistence.


## Fast demo vs full-data mode

The Data Pipeline page now has two execution modes:

- **Demo Sample** samples complete orders with a fixed random seed and trains an isolated LightGBM model with fewer trees. This is intended for a fast classroom demo.
- **Full Dataset** uses the cumulative versioned dataset and performs the full retraining workflow.

The pipeline displays **Step X of 14**, the operation currently running, a progress bar, and short explanations based on the original notebook (temporal splitting, customer cancellation history, LightGBM, F1 thresholding, ROC-AUC, PR-AUC, calibration, and feature importance).

### MLflow serialization on Streamlit Cloud

The LightGBM sklearn pipeline is logged with MLflow using explicit `cloudpickle` serialization. This avoids `skops` trust errors for third-party LightGBM estimator types on newer MLflow releases.
