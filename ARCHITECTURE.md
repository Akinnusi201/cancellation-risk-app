# Architecture

## System goal

The application separates operational scoring from model development. Operations always has a deployable model available; training jobs happen independently and cannot replace Production without explicit developer approval.

## End-to-end lifecycle

```text
Incoming item-level data
        ↓
DataOps
  fingerprint / validate / quarantine
  aggregate to order level
  recompute historical customer features
  create immutable dataset version
        ↓
Training request
        ↓
Google Colab / Colab Enterprise
        ↓
Five-model experiment suite
  Logistic Regression
  Random Forest
  Extra Trees
  LightGBM
  XGBoost
        ↓
MLflow tracking + reproducibility metadata
        ↓
Qualification gates
        ↓
Best qualified model → Candidate
        ↓
Developer approval
        ↓
Production model
        ↓
Operations scoring
        ↓
Runtime + drift + observed-performance monitoring
        ↓
Degradation policy
        ↓
Retraining request
```

## Production inference

`artifacts/production_model.pkl` and `artifacts/active_model.json` define the active Production model. Streamlit startup loads these artifacts and never launches training.

The initial Production model remains the packaged LightGBM model. The repository also includes a starter registry containing trained Logistic Regression, Random Forest, Extra Trees, and XGBoost artifacts for immediate comparison.

## DataOps

DataOps owns trustworthy data products, not model training. It performs duplicate protection, raw preservation, validation, quarantine, deterministic order aggregation, cumulative customer-history recomputation, and immutable dataset versioning.

A successful new dataset version creates a training request. Ordinary Colab requires a developer to open the notebook; an optional Colab Enterprise integration can submit a notebook execution when cloud configuration is present.

## Model training

Formal training runs in `notebooks/end_to_end_ml_workflow.ipynb` and uses `src/models/suite.py`.

All five models receive the same time-based train / validation / test split and leakage-safe feature contract. Random Forest, Extra Trees, and Logistic Regression use CPU. LightGBM and XGBoost prefer a GPU and fall back to CPU if needed.

The notebook stores full experiment output and MLflow tracking data in Google Drive by default so the Streamlit host does not become the computational bottleneck.

## Experiment tracking and reproducibility

Experiments and runs have deterministic readable identities such as:

```text
cancellation-risk__pakistan_seed_v1__20260930
xgboost__20260930T180000Z
```

Each run stores the dataset version, timestamp, random seed, split sizes, hyperparameters, training device, technical metrics, business metrics, Git commit SHA, Python/library versions, and serialized model artifact.

## Candidate selection

Candidate selection is a two-stage process:

1. Models must pass qualification gates for discrimination, calibration, high-precision recall, and positive business value.
2. Qualified models receive a weighted score that combines ROC-AUC, PR-AUC, calibration, recall at 90% precision, and normalized business value.

The best qualified model becomes Candidate. Candidate creation can be automated; Production deployment cannot.

## Model registry and deployment

The developer Model Registry exposes three plain-language states:

- **Ready:** trained and available for comparison.
- **Candidate:** proposed Production replacement.
- **Production:** currently scoring orders.

A developer must review the Candidate and click **Promote Candidate to Production**. The promotion step replaces the Production artifact and refreshes reference prediction/support artifacts used by monitoring and simulation.

## Monitoring and degradation

Monitoring separates:

- runtime health and latency,
- data/prediction drift,
- live prototype business impact,
- historical model/business evaluation,
- observed production performance once final outcomes are available.

Simulation remains excluded from technical drift monitoring because the simulation queue is deliberately risk-stratified. It can still contribute to prototype business impact after the Operations Manager makes a decision.

### Hybrid retraining trigger

Drift is an early warning, not proof that predictive performance has degraded. Severe unlabeled drift must persist across consecutive health checks. When at least 100 real final outcomes are available, observed ROC-AUC, Brier score, and business value can confirm degradation directly.

A qualifying trigger creates a retraining request. The resulting model suite can automatically choose a new Candidate, but the developer still controls deployment.

## Colab Enterprise option

The classroom path uses ordinary Google Colab. The optional production-style path uses a Colab Enterprise Notebook Execution Job. When the required `COLAB_ENTERPRISE_*` settings and Google Application Default Credentials are available, the app can submit the training notebook programmatically.

## Role boundaries

### Operations Manager

Can use the Operations Dashboard, Score Order, and Decision History. Operations does not see DataOps, experiments, the model registry, or deployment controls.

### Developer

Can manage DataOps, the model registry, experiments, monitoring, retraining, and system status.

## DevOps

GitHub Actions runs tests and packaged-inference checks and builds the Docker image. The Streamlit deployment remains lightweight because computational training is pushed to Colab. The application can still run locally or in Docker.

## Currency layer

Model features remain in the original PKR units used for training. The UI converts money to USD for presentation using the FX adapter. Manual USD inputs are converted back to PKR before inference, preserving the model feature contract.
