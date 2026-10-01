# Architecture

## System flow

```text
Incoming order / new transaction data
        │
        ├──────────── Operations scoring ──────────────┐
        │                                              │
        ▼                                              ▼
      DataOps                                   Production model
 ingest / validate                              cancellation risk
 quarantine / aggregate                              │
 version / lineage                                    ▼
        │                                     economic decision
        ▼                                              │
 immutable dataset version                            ▼
        │                                      Operations Manager
        ▼                                      release / verify order
 retraining request
        │
        ▼
 automatic retraining enabled? ── no ──► queued request
        │ yes
        ▼
 five-model training workflow
        │
        ├── Logistic Regression (CPU)
        ├── Random Forest (CPU)
        ├── Extra Trees (CPU)
        ├── LightGBM (GPU preferred, CPU fallback)
        └── XGBoost (GPU preferred, CPU fallback)
        │
        ▼
 MLflow experiment tracking
        │
        ▼
 qualification gates + weighted comparison
        │
        ▼
 best qualified model = Candidate
        │
        ▼
 Developer review and approval
        │
        ▼
 Promote Candidate to Production
        │
        ▼
 Runtime monitoring
        │
        └── sustained drift / confirmed degradation
             creates a new retraining request
```

## Operations paths

The application supports production-like manual and batch scoring. An optional Incoming Order Demo uses packaged historical examples to demonstrate the same review lifecycle.

Manual single-order scoring creates an interactive review state. Batch scoring records predictions for monitoring and applies the automated economic recommendation to each row.

Demo simulation can be included in Business Impact for presentation purposes, but it is always excluded from technical drift monitoring.

## DataOps boundary

DataOps owns ingestion, validation, quarantine, deterministic item-to-order aggregation, lineage, leakage-safe customer-history recomputation, and immutable dataset versioning.

Creating a dataset version can create a retraining request. If automatic retraining is enabled, the request is passed to the five-model training workflow. DataOps never promotes a model or changes Production directly.

## ModelOps boundary

ModelOps trains and compares five model families using the same temporal split and feature contract. Each run is tracked in MLflow with dataset version, random seed, hyperparameters, device, metrics, business value, Git SHA, and environment information.

Candidate selection is automatic after qualification gates. Deployment remains manual.

## Training workloads

### Standard full dataset

This is the default retraining mode. All available orders in the selected immutable dataset version are used.

### Quick sampled run

This is an optional deterministic, time-spanning sample used for demonstrations, diagnostics, or resource-constrained hosts. It is not the default operating assumption.

## Monitoring trigger

Retraining can be requested by:

- a new dataset version
- sustained severe prediction or feature drift
- confirmed ROC-AUC degradation
- confirmed Brier-score deterioration
- non-positive observed business value

Technical drift uses production-like traffic only. Demo simulation and historical evaluation remain excluded.

## Deployment governance

Training and Candidate selection may be automated. Deployment is not.

```text
Candidate
   ↓
Developer reviews metrics, scope, and business impact
   ↓
Explicit approval checkbox
   ↓
Promote Candidate to Production
```

## Optional Colab path

`notebooks/end_to_end_ml_workflow.ipynb` remains available as an accelerated-compute alternative when separate compute or a T4 GPU is useful. The application itself can run the same five-model workflow directly from the Developer workspace.

## Manual experiment path

Developers can run a single tuned model directly from **Experiments & MLflow** without waiting for DataOps, monitoring, or the five-model automatic retraining workflow:

```text
Versioned dataset
      ↓
Choose model + hyperparameters
      ↓
Time-based train / validation / test
      ↓
Train + evaluate
      ↓
MLflow run
      ↓
Model Registry: READY
      ↓
Optional developer action: mark Candidate
      ↓
Separate approval: Promote to Production
```

This path shares the same feature schema, temporal evaluation, business metrics, reproducibility metadata, and deployment governance as the automated workflow. It never changes Production automatically.

## Final model-selection contract

The formal five-model benchmark and automated retraining workflow use one deterministic temporal split per dataset version:

```text
full versioned dataset
        |
        +--> first 70% by time  -> training
        +--> next 15% by time   -> validation / threshold tuning / qualification / Candidate selection
        +--> final 15% by time  -> final unbiased test reporting only
```

All models in one suite share the same dataset fingerprint, split ID, feature-schema version, random-seed policy, and business assumptions. The test holdout is never used to rank or qualify models. Promotion remains a separate developer-controlled action after reviewing the final test evidence.

The packaged report-ready benchmark is stored in `artifacts/final_benchmark/` and can be regenerated from the Experiments & MLflow page or `scripts/run_final_benchmark.py`.
