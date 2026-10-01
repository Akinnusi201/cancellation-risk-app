# Profit-Aware E-Commerce Order Cancellation Risk

A deployable machine-learning application that helps an e-commerce operations team decide whether a newly placed order should be released to fulfillment or held briefly for verification.

The system combines cancellation-risk prediction with a simple economic decision rule, and separates day-to-day Operations work from DataOps, ModelOps, deployment, and monitoring.

## What problem does this solve?

Late order cancellations can waste warehouse labor, inventory reservation, payment-processing effort, customer-service time, and logistics work. Reviewing every order would also create unnecessary delay and cost.

The application therefore asks a practical question:

> **Is verifying this order expected to save more money than the verification costs?**

The ML model estimates cancellation probability. The business layer combines that probability with configurable assumptions about cancellation loss, verification cost, and how much loss verification can prevent.

## Who uses the application?

- **Operations Manager:** scores or reviews incoming orders and decides whether to release or verify them.
- **Developer / Data Scientist:** manages datasets, experiments, retraining, candidates, deployment, and monitoring.
- **Finance / Business stakeholder:** can inspect the cost assumptions, expected savings, and historical business backtests without reading model code.

No password is required in the course deployment. The landing page asks the user to choose the Operations Manager or Developer workspace. The role determines which pages are registered and visible.

## What happens when the app starts?

The application does **not** retrain on startup.

A pretrained LightGBM model is packaged as the initial Production model, so inference is available immediately:

```text
Open app
   ↓
Choose a workspace
   ↓
Load packaged Production model
   ↓
Ready to score orders
```

Retraining happens only from a DataOps handoff, a monitoring trigger, or an explicit Developer action.

## Starter model registry

Five trained model families are included from the first deployment:

| Model | Initial role | Typical compute |
|---|---|---|
| Logistic Regression | Ready / baseline | CPU |
| Random Forest | Ready | CPU |
| Extra Trees | Ready | CPU |
| LightGBM | **Production** | CPU or GPU when available |
| XGBoost | Ready | CPU or GPU when available |

The packaged LightGBM model remains Production until a developer explicitly promotes a Candidate.

## Operations workflow

The Operations workspace contains:

- Operations Dashboard
- Score Order
- Decision History

There are three scoring paths:

### Manual Order

A single incoming order can be entered in the UI. It is scored by the current Production model, placed in Operations review, and the manager chooses:

```text
Order received
   ↓
Risk score + economic recommendation
   ↓
Awaiting Operations Review
   ├── Release to Fulfillment
   └── Keep for Verification
```

Manual scores are production-like traffic and contribute to runtime monitoring.

### Batch CSV

A CSV containing one row per new order can be scored in one run. Batch predictions are persisted for monitoring and Business Impact reporting. A scored CSV can be downloaded afterward.

### Incoming Order Demo

A packaged simulation is available for presentations and walkthroughs. It demonstrates the same customer-to-Operations review flow using historical orders selected across low, medium, and high model-risk bands.

Demo traffic is always excluded from technical drift monitoring. Developers can choose whether demo simulation should be included in Business Impact for presentation purposes.

## End-to-end ML workflow

```text
New item-level data
      ↓
DataOps
  ingest
  validate
  quarantine
  aggregate
  version
      ↓
Retraining request
      ↓
Automatic retraining enabled?
      ↓ yes
Five-model training workflow
      ↓
Logistic Regression
Random Forest
Extra Trees
LightGBM
XGBoost
      ↓
MLflow experiment tracking
      ↓
Qualification gates
      ↓
Best qualified model → Candidate
      ↓
Developer review
      ↓
Promote Candidate to Production
      ↓
Operations scoring
      ↓
Monitoring
      ↓
Sustained degradation or confirmed performance decline?
      ↓ yes
Create retraining request and repeat
```

Candidate selection can be automated. Production deployment is not.

## DataOps

The DataOps pipeline converts incoming item-level transactions into a reproducible order-level data product.

It performs:

1. File fingerprinting and duplicate-batch protection
2. Raw-source preservation
3. Schema and data-quality validation
4. Quarantine of invalid or inconsistent records
5. Deterministic item-to-order aggregation
6. Leakage-safe customer-history recomputation
7. Immutable dataset version creation
8. Retraining-request creation for the new dataset version

The Pakistan baseline is already packaged as a validated order-level seed dataset. New uploads create additional immutable dataset versions.

If automatic retraining is enabled, DataOps passes the new version into the five-model training workflow. DataOps itself never changes the Production model.

## Retraining settings

Open:

```text
Developer → Model Registry & Deployment
```

The application supports two training workloads.

### Standard full dataset

This is the default and normal retraining path. All available orders in the selected immutable dataset version are used for the time-based train/validation/test workflow.

### Quick sampled run

This optional mode selects a deterministic, time-spanning sample. It exists for demonstrations, diagnostics, or resource-constrained compute environments. It is not the default operating assumption.

The Developer can also choose whether LightGBM and XGBoost should prefer GPU execution when a compatible GPU is available. Random Forest and Extra Trees remain CPU-based in scikit-learn.

### Automatic retraining

When enabled, retraining starts when either:

- DataOps creates a new dataset version, or
- monitoring detects a sustained degradation signal that satisfies the retraining policy.

Developers can also click:

```text
Run Five-Model Retraining Now
```

to retrain immediately on a selected dataset version.

## MLflow experiment tracking

Each five-model suite creates a named MLflow experiment, for example:

```text
cancellation-risk__pakistan_seed_v1__20260930
```

Runs receive readable, timestamped names:

```text
logistic_regression__20260930T180000Z
random_forest__20260930T180000Z
extra_trees__20260930T180000Z
lightgbm__20260930T180000Z
xgboost__20260930T180000Z
```

Each run records information such as:

- dataset version
- model family
- training timestamp
- random seed
- train / validation / test sizes
- hyperparameters
- training device
- ROC-AUC
- PR-AUC
- Brier score
- F1, precision, and recall
- recall at fixed precision
- estimated business value
- Git commit SHA
- Python and library versions
- fitted model artifact

## Candidate selection and deployment

A model does not become Candidate solely because it has the highest ROC-AUC.

The workflow first applies qualification gates covering:

- discrimination
- probability calibration
- recall at high precision
- positive estimated business value

Qualified models are then compared using a weighted technical-and-business score. The best qualified model is registered as **Candidate**.

Production remains unchanged until a developer checks the approval box and clicks:

```text
Promote Candidate to Production
```

This keeps automated experimentation separate from production deployment.

## Monitoring and retraining

### Runtime health

Production-like manual, batch, and future API traffic can contribute to:

- prediction volume
- average predicted risk
- inference latency
- pipeline reliability
- verification rate

### Drift

Prediction and feature distributions are compared with the Production model's reference population. Demo simulation and retrospective historical evaluation are excluded so their sampling design cannot create false drift alerts.

A minimum runtime sample is required before the application labels the system as stable, watch, or drift.

### Actual performance

When final outcomes become available, a developer can upload:

```text
order_id,final_outcome
100001,Completed
100002,Canceled
```

The application can then calculate observed runtime performance such as ROC-AUC, Brier score, and high-precision recall.

### Retraining policy

Retraining is intentionally conservative:

- one small or noisy batch does not trigger retraining
- sustained severe drift can create a retraining request
- confirmed ROC-AUC degradation can trigger retraining
- confirmed Brier-score deterioration can trigger retraining
- non-positive observed business value can trigger retraining

When automatic retraining is enabled, the request runs through the five-model workflow and may create a Candidate. Production still requires human approval.

## Business Impact

Business Impact can report expected value from manual and batch Operations scoring. For class demonstrations, Developers may also enable inclusion of the live demo simulation.

The configurable assumptions are presented in plain language:

- **Loss if a canceled order reaches fulfillment**
- **Loss prevented by verification**
- **Cost to verify one order**
- **Extra cost if a good order is verified**

The user-facing UI displays money in USD using the latest available PKR-to-USD rate. Model features remain in PKR internally so the trained model contract is unchanged.

Historical business backtesting is kept separate from live Operations Business Impact.

## Optional Google Colab training

The repository includes:

```text
notebooks/end_to_end_ml_workflow.ipynb
```

The application does not require Colab to function. The notebook is an optional accelerated-compute path when separate compute or a T4 GPU is useful. It runs the same five-model experiment and can export a Candidate package for import into the Model Registry.

## Reproducibility

Model runs record enough context to trace a result back to its data, code, and configuration:

- immutable dataset version
- time-based split
- random seed
- experiment and run names
- Git commit SHA
- environment and library versions
- hyperparameters
- technical metrics
- business metrics
- serialized model artifact

## Local setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Then open:

```text
http://localhost:8501
```

## Docker

```bash
docker build -t cancellation-risk-app .
docker run --rm -p 8501:8501 cancellation-risk-app
```

## GitHub Actions

The repository includes:

```text
.github/workflows/ci.yml
```

CI installs dependencies, runs automated tests, verifies packaged inference, and checks that the Docker image builds.

If a file-copy operation skipped hidden folders, restore the workflow with:

```bash
mkdir -p .github/workflows
cp GITHUB_ACTIONS_CI.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI"
git push
```

## Project structure

```text
app.py                         Streamlit entry point
views/                         Role-specific application pages
src/data/                      Ingestion, validation, aggregation, versioning
src/features/                  Feature construction and inference contract
src/models/                    Training, registry, inference, five-model suite
src/monitoring/                Runtime, drift, business and performance metrics
src/retraining.py              Automatic/manual retraining orchestration
notebooks/end_to_end_ml_workflow.ipynb
                               Optional accelerated-compute workflow
artifacts/model_registry/      Packaged trained model suite
artifacts/retraining/          Retraining settings, requests, and generated runs
artifacts/production_model.pkl Current deployed artifact
artifacts/active_model.json    Production model metadata
tests/                         Automated regression tests
.github/workflows/ci.yml       Continuous integration
Dockerfile                     Container deployment
```

## Deployment notes and limitations

The core scoring, DataOps, ModelOps, monitoring, retraining, and deployment-control workflows are functional. A few deployment choices are intentionally simplified for the course environment:

- role selection is passwordless and demonstrates role-specific routing, not enterprise authentication
- in-app retraining uses the compute available to the Streamlit host and may be slow on CPU-only services
- persistent runtime history depends on durable storage; some hosted Streamlit environments use ephemeral local filesystems
- business-cost inputs are configurable because the public dataset does not contain a retailer's internal warehouse or verification costs
- automatic retraining does not imply automatic deployment; Candidate promotion remains a human-controlled action
- the Incoming Order Demo is optional presentation traffic and is kept separate from technical production monitoring

## Dataset

The project uses **Pakistan's Largest E-Commerce Dataset** by Zeeshan-ul-hassan Usmani and collaborator. The public source is item-level; the prediction unit in this system is one order.

The packaged seed contains the validated and aggregated complete/canceled order population used by the initial model suite.

## Quick glossary

- **DataOps:** makes incoming data trustworthy, versioned, and reproducible.
- **Experiment:** one model-training run with recorded settings and results.
- **MLflow:** tracks experiment parameters, metrics, and model artifacts.
- **Candidate:** a model proposed as a replacement for Production.
- **Production:** the model currently used for operational scoring.
- **Drift:** live data or predictions have changed relative to the model's reference population.
- **Retraining:** create new model versions using a selected immutable dataset version.
- **Deployment / Promotion:** make an approved Candidate the Production model.
