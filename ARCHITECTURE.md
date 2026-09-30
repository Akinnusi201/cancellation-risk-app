# Architecture

## Separation of operational concerns

### Production inference

`artifacts/production_model.pkl` is the only artifact required to perform model inference. `active_model.json` supplies version, lineage, threshold, and evaluation metadata. The application does not retrain at startup.

### DataOps

1. Receive a new immutable CSV batch.
2. Calculate a SHA-256 fingerprint and reject exact re-uploads.
3. Drop fully blank physical rows.
4. Validate required schema, dates, numeric ranges, target status, IDs, and duplicates.
5. Quarantine invalid records and inconsistent order statuses.
6. Exclude order IDs already present in the latest cumulative version.
7. Aggregate item records deterministically to one row per order.
8. Append the new orders to the cumulative dataset.
9. Recompute prior-customer cancellation history across the full chronological snapshot.
10. Persist the cleaned snapshot, quarantine output, validation report, and lineage metadata as a new dataset version.

DataOps never trains or promotes a model.

### ModelOps

1. Developer selects a versioned dataset.
2. Logistic Regression is trained as a baseline on a temporal 70/15/15 split.
3. A LightGBM candidate is trained on the exact same split and feature set.
4. The validation period selects the F1 threshold for each model.
5. Test metrics include ROC-AUC, PR-AUC, Brier score, F1, precision, recall, and recall at fixed precision.
6. Evaluation artifacts and both experiment runs are logged to MLflow.
7. The candidate is persisted separately from production.
8. Developer compares the baseline, candidate, and active production metrics.
9. An explicit promotion action replaces the production model artifact and metadata.

A failed or poor candidate does not affect operations scoring.

## Role boundaries

### Operations Manager

Can access production scoring, economic recommendations, and decision history. Cannot access ingestion, training, experiments, or promotion controls.

### Developer

Can access DataOps, candidate training/promotion, monitoring, MLflow experiments, and system status. Developer navigation is separate from the operations workflow.

## Profit-aware decision layer

The machine-learning model outputs `P(Cancellation)`. The action policy is calculated separately and presented in plain operational language.

The business layer estimates how much money verification could save, then subtracts the cost of verifying the order and the expected cost of unnecessarily verifying an order that would have completed normally. Positive estimated net savings produces a **Verify before fulfillment** recommendation.

The UI exposes four assumptions: loss from a late cancellation, loss prevented by verification, cost per verification, and extra cost of an unnecessary verification. These assumptions change the action recommendation, not the model probability.

## Packaged baseline

The raw Pakistan CSV is not required at runtime. A cleaned order-level compressed CSV is stored under `data/seed/`, while the pretrained model, reference sample, simulation queue, and evaluation artifacts are stored under `artifacts/`.

A fresh DuckDB runtime registers the packaged dataset metadata on first use. This bootstrapping action does not perform data transformation or model training.

## Runtime persistence

Streamlit Community Cloud does not guarantee persistence for files created after deployment. The packaged baseline remains available after reboot because it is part of the repository. Runtime DuckDB history, uploaded batches, candidate artifacts, and local MLflow state may reset unless external persistent storage is configured.

## Production monitoring

Single-order inference records core model latency, probability, recommendation, policy inputs, and the model feature vector used for scoring. Monitoring computes runtime latency summaries, pipeline success rates, prediction PSI, numeric-feature PSI, categorical total-variation drift, and aggregate expected verification activity. At least 100 eligible production-like runtime observations are required before drift estimates are displayed.

The packaged temporal holdout includes production probabilities and labels. This supports historical business-value evaluation and three simple what-if scenarios without retraining. These savings are counterfactual estimates because the source dataset does not contain warehouse labor, verification-cost, or verification-success measurements.

## DevOps

GitHub Actions runs tests and a packaged-inference smoke test on each push/pull request, then performs a Docker build. The Docker image uses Python 3.12, exposes Streamlit on port 8501, and includes a health check. Streamlit Community Cloud remains the classroom deployment target.
