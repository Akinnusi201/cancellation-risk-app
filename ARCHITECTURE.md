# Prototype architecture

## Automated path

1. A CSV is uploaded in the Streamlit **Data Pipeline** page.
2. The file hash is atomically reserved in DuckDB before processing. Exact re-uploads become a safe no-op, including Streamlit reruns/concurrent requests.
3. Schema, date, numeric range, target, duplicate, and order-status checks run automatically.
4. Safely usable rows continue; questionable rows are written to a quarantine CSV with a reason.
5. Existing order IDs from prior versions are excluded/quarantined.
6. Item rows are aggregated to one order, and customer cancellation history uses prior orders only.
7. A cumulative timestamped Parquet snapshot is created and registered in DuckDB.
8. One **LightGBM** model is trained automatically. This is intentionally limited to one model family for prototype speed.
9. The oldest 70% of orders trains the model; the next 15% selects the maximum-F1 decision threshold.
10. ROC-AUC, PR-AUC, Brier score, F1, precision, recall, curves, native LightGBM feature importance, parameters, model artifact, and dataset version are logged to MLflow.
11. The successful automated LightGBM run is registered as a new production model version and becomes the active model.
12. The newest 15% remains the final test/held-out queue and feeds the manager simulation.
13. Approve/Hold decisions and historical outcomes are recorded in DuckDB.

## Manual experiments

The MLflow Experiments page can train additional **LightGBM configurations** by changing `n_estimators`, `learning_rate`, and `num_leaves`. These runs are fully logged and versioned in MLflow but never auto-promote to production.

## Separation of concerns

- **Streamlit**: internet-facing UI.
- **DuckDB**: batch/version/audit metadata and manager decisions.
- **Parquet/CSV/JSON**: cleaned snapshots, quarantine outputs, validation reports.
- **MLflow**: experiment runs, metrics, parameters, artifacts, and model versions.
- **LightGBM/scikit-learn**: production inference pipeline.

## Threshold rule

The classification threshold is chosen only on validation data by maximum F1. Test data never selects the threshold.

## Cloud deployment

The repository is structured for Streamlit Community Cloud. The default DuckDB and MLflow SQLite files live inside the hosted app instance. For a classroom prototype this is sufficient, but Community Cloud does not guarantee runtime-file persistence across reboots. `MLFLOW_TRACKING_URI` and `DUCKDB_PATH` are configurable so persistent cloud storage can be introduced later.
