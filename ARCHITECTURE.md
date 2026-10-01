# Architecture

## Prototype system flow

```text
Customer / new order data
        │
        ├─────────────── Operations path ───────────────┐
        │                                               │
        ▼                                               ▼
     DataOps                                      Production model
 ingest / validate                               cancellation risk
 quarantine / aggregate                               │
 version data                                          ▼
        │                                      profit-aware decision
        ▼                                               │
 versioned dataset                                      ▼
        │                                      Operations Manager
        ▼                                      release / verify order
 retraining request
        │
        ▼
 Prototype retraining settings
 automatic enabled? ── no ──► queued request
        │ yes
        ▼
 direct five-model training in Streamlit
        │
        ├── Logistic Regression
        ├── Random Forest
        ├── Extra Trees
        ├── LightGBM
        └── XGBoost
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
 Developer approval
        │
        ▼
 Promote Candidate to Production
        │
        ▼
 Monitoring
        │
        └── sustained degradation creates a new retraining request
```

## DataOps boundary

DataOps owns ingestion, validation, quarantine, aggregation, lineage, customer-history recomputation, and immutable dataset versioning.

Creating a new dataset version can create a retraining request. If automatic retraining is enabled, the request is handed to the direct prototype trainer. DataOps never promotes a model or changes Production itself.

## ModelOps boundary

ModelOps trains and compares five model families using the same temporal split and feature contract. Each run is tracked in MLflow with reproducibility metadata.

Candidate selection is automatic after qualification gates, but deployment is manual.

## Prototype retraining modes

### Fast prototype

A deterministic time-spanning sample, default 60,000 orders, is selected from the chosen dataset version. This is intended for hosted Streamlit CPU environments and live demonstrations.

### Full active dataset

The same five-model workflow runs on the complete selected dataset version. This is more computationally expensive and may take several minutes.

## Monitoring trigger

Retraining can be requested by:

- a new dataset version
- sustained severe prediction or feature drift
- confirmed ROC-AUC degradation
- confirmed Brier-score deterioration
- non-positive observed business value

Simulation traffic remains excluded from drift monitoring, although live simulation can count toward prototype business-impact reporting.

## Deployment governance

Training and Candidate selection may be automated. Deployment is not.

```text
Candidate
   ↓
Developer reviews metrics and business impact
   ↓
Explicit approval checkbox
   ↓
Promote Candidate to Production
```

## Optional notebook

`notebooks/end_to_end_ml_workflow.ipynb` remains available as an optional separate-compute workflow. It is not required by the prototype and does not change the app's direct retraining design.
