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
2. A LightGBM candidate is trained with a temporal 70/15/15 split.
3. The validation period selects the F1 threshold.
4. Test metrics and evaluation artifacts are generated.
5. The run is logged to MLflow.
6. The candidate is persisted separately from production.
7. Developer compares candidate metrics with the active production model.
8. An explicit promotion action replaces the production model artifact and metadata.

A failed or poor candidate does not affect operations scoring.

## Role boundaries

### Operations Manager

Can access production scoring, economic recommendations, and decision history. Cannot access ingestion, training, experiments, or promotion controls.

### Developer

Can access DataOps, candidate training/promotion, monitoring, MLflow experiments, and system status. Developer navigation is separate from the operations workflow.

## Profit-aware decision layer

The machine-learning model outputs `P(Cancellation)`. The action policy is calculated separately:

`Expected Avoidable Cost = P(Cancellation) × Avoidable Fulfillment Cost × Intervention Effectiveness`

`Net Expected Savings = Expected Avoidable Cost - Intervention Cost - Expected False-Intervention Friction`

Positive net expected savings produces a Hold-for-Verification recommendation. This means model training and business-cost policy can change independently.

## Packaged baseline

The raw Pakistan CSV is not required at runtime. A cleaned order-level compressed CSV is stored under `data/seed/`, while the pretrained model, reference sample, simulation queue, and evaluation artifacts are stored under `artifacts/`.

A fresh DuckDB runtime registers the packaged dataset metadata on first use. This bootstrapping action does not perform data transformation or model training.

## Runtime persistence

Streamlit Community Cloud does not guarantee persistence for files created after deployment. The packaged baseline remains available after reboot because it is part of the repository. Runtime DuckDB history, uploaded batches, candidate artifacts, and local MLflow state may reset unless external persistent storage is configured.
