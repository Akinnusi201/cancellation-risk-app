# Profit-Aware E-Commerce Order Cancellation Risk

A production-style machine-learning prototype that helps an e-commerce team decide whether a newly placed order should be released to fulfillment or briefly held for verification.

The application predicts cancellation risk, converts that risk into an expected financial impact, and gives an Operations Manager a simple decision workflow. Developers get separate tools for DataOps, model experimentation, MLflow tracking, model comparison, retraining, deployment, and monitoring.

## What problem does this solve?

Orders canceled after warehouse work begins can waste picking, packing, inventory reservation, payment-processing, customer-service, and logistics effort. Reviewing every order would also create unnecessary delay and cost.

The application therefore asks one practical question:

> **Is verifying this order expected to save more money than the verification costs?**

The ML model estimates cancellation probability. The business layer combines that probability with configurable assumptions about late-cancellation loss and verification cost.

## Who is this for?

You do not need to be a machine-learning specialist to understand the app.

- **Operations Manager:** reviews incoming orders and chooses whether to release or verify them.
- **Developer / Data Scientist:** manages datasets, model experiments, retraining, candidates, deployment, and monitoring.
- **Finance / Business stakeholder:** can understand the economic assumptions and business backtests without reading model code.

## What happens when the app starts?

The application **does not retrain at startup**.

A pretrained LightGBM model is packaged as the initial Production model, so scoring is available immediately:

```text
Open app
   ↓
Choose Operations Manager or Developer
   ↓
Load packaged Production model
   ↓
Ready to score orders
```

Retraining happens only when the Developer workflow starts it automatically or manually.

## Starter model registry

The repository ships with trained versions of five model families:

| Model | Initial role | Prototype compute |
|---|---|---|
| Logistic Regression | Ready / baseline | CPU |
| Random Forest | Ready | CPU |
| Extra Trees | Ready | CPU |
| LightGBM | **Production** | CPU |
| XGBoost | Ready | CPU |

The current LightGBM model remains Production until a developer explicitly promotes a Candidate.

## Application roles

### Operations Manager

The Operations workspace contains only:

- Operations Dashboard
- Score Order
- Decision History

The live prototype behaves like an incoming customer-order workflow:

```text
Customer places order
        ↓
Cancellation-risk screening
        ↓
Awaiting Operations Review
        ↓
Operations Manager chooses
   ├── Release to Fulfillment
   └── Keep for Verification
        ↓
Business Impact updates
```

### Developer

The Developer workspace contains:

- Developer Dashboard
- DataOps
- Model Registry & Deployment
- Model Monitoring
- Experiments / MLflow
- System Status

## End-to-end ML workflow

The prototype can run the complete retraining workflow directly inside Streamlit:

```text
New order data
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
Run five-model suite directly in app
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
Best qualified model becomes Candidate
      ↓
Developer review
      ↓
Promote Candidate to Production
      ↓
Operations scoring
      ↓
Monitoring
      ↓
Sustained degradation?
      ↓ yes
Create retraining request and repeat
```

Training may select a Candidate automatically, but **deployment is never automatic**.

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

The Pakistan baseline is already packaged as a validated order-level seed dataset. Developers normally upload only new batches.

When automatic retraining is enabled, DataOps hands the new dataset version directly to the five-model training workflow. DataOps itself never changes the Production model.

## Direct prototype retraining

Open:

```text
Developer → Model Registry & Deployment
```

The **Prototype retraining settings** section provides two controls:

### Automatic retraining

When enabled, retraining starts automatically when either:

- DataOps creates a new dataset version, or
- Monitoring detects sustained degradation that satisfies the retraining policy.

### Training workload

**Fast prototype** is the recommended hosted-Streamlit setting. It selects a deterministic time-spanning sample from the active dataset and trains all five models on that sample.

The default limit is:

```text
60,000 orders
```

This keeps the workflow practical on CPU while preserving orders across the dataset timeline.

**Full active dataset** trains the same five models on the complete selected dataset version and may take substantially longer.

Developers can also click:

```text
Run Five-Model Retraining Now
```

for an immediate retraining run without waiting for a monitoring trigger.

## MLflow experiment tracking

Every five-model suite creates a named MLflow experiment, for example:

```text
cancellation-risk__pakistan_seed_v1__20260930
```

Runs receive readable names:

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
- model hyperparameters
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

## Candidate selection

A model does not become Candidate simply because it has the highest ROC-AUC.

The workflow first applies qualification gates for:

- discrimination
- probability calibration
- recall at high precision
- positive estimated business value

Qualified models are then compared using a weighted score that combines technical performance and business value.

The highest-ranked qualified model becomes **Candidate** automatically.

Production remains unchanged until a developer checks the approval box and clicks:

```text
Promote Candidate to Production
```

## Monitoring and retraining

Monitoring separates several concepts.

### Runtime health

Tracks production-like scoring traffic, including:

- prediction volume
- average predicted risk
- model latency
- pipeline reliability
- verification rate

### Drift

Compares production-like traffic with the Production model's reference population.

Live simulation and historical-evaluation traffic are excluded from drift monitoring so demo sampling does not generate false drift alerts.

### Actual performance

When final order outcomes become available, a developer can upload:

```text
order_id,final_outcome
100001,Completed
100002,Canceled
```

The application then measures observed production performance.

### Retraining policy

The trigger is intentionally conservative:

- one small or noisy batch does not start retraining
- sustained severe drift can trigger retraining
- confirmed ROC-AUC degradation can trigger retraining
- confirmed Brier-score deterioration can trigger retraining
- non-positive observed business value can trigger retraining

When automatic retraining is enabled, an eligible request runs directly in the app. The five runs are tracked in MLflow and the best qualified model becomes Candidate.

Deployment still requires human approval.

## Optional Colab notebook

The repository still includes:

```text
notebooks/end_to_end_ml_workflow.ipynb
```

It is an optional heavy-compute utility, not a requirement for the prototype. The primary workflow now runs directly from the Developer UI.

If you choose to use the notebook later, it runs the same five-model experiment and can export a Candidate package for import into the app.

## Business decision logic

The Operations UI uses plain language.

The configurable assumptions are:

- **Loss if a canceled order reaches fulfillment**
- **Loss prevented by verification**
- **Cost to verify one order**
- **Extra cost if a good order is verified**

The user-facing application displays money in USD using a current PKR-to-USD conversion when available. Model features stay in PKR internally so the trained model contract is not changed.

## Reproducibility

Formal model runs record:

- immutable dataset version
- time-based split
- random seed
- experiment and run names
- Git commit SHA
- environment and library versions
- hyperparameters
- metrics
- business assumptions
- serialized model artifact

This allows a Production model to be traced back to the code, data, configuration, and experiment that created it.

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

No password is required in this classroom prototype. Choose the Operations Manager or Developer workspace from the landing page.

## Docker

```bash
docker build -t cancellation-risk-app .
docker run --rm -p 8501:8501 cancellation-risk-app
```

Then open:

```text
http://localhost:8501
```

## GitHub Actions

The repository includes:

```text
.github/workflows/ci.yml
```

CI installs the project, runs automated tests, verifies packaged inference, and checks that the Docker image builds.

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
src/retraining.py              Direct prototype retraining orchestration
notebooks/end_to_end_ml_workflow.ipynb
                               Optional heavy-compute notebook
artifacts/model_registry/      Packaged trained model suite
artifacts/retraining/          Retraining settings, requests, and generated runs
artifacts/production_model.pkl Current deployed artifact
artifacts/active_model.json    Production model metadata
tests/                         Automated regression tests
.github/workflows/ci.yml       Continuous integration
Dockerfile                     Container deployment
```

## Important prototype limitations

This is a school ML-system prototype, not a production commerce platform.

- Passwordless role selection demonstrates role-specific UX, not enterprise authentication.
- Direct retraining runs inside the Streamlit process. Hosted environments may impose CPU, memory, or execution-time limits.
- Fast prototype mode intentionally trades training-set size for responsiveness.
- Business-cost assumptions are configurable because the public dataset does not include a retailer's real warehouse or verification costs.
- Runtime state on some hosted Streamlit environments may be ephemeral unless external persistence is added.
- Automatic retraining does not mean automatic deployment. Candidate promotion intentionally remains a human approval step.

## Dataset

The project uses **Pakistan's Largest E-Commerce Dataset** by Zeeshan-ul-hassan Usmani and collaborator. The raw public data is item-level; the ML prediction unit is one order.

The packaged seed contains the validated, aggregated complete/canceled order population used by the initial model.

## Quick glossary

- **DataOps:** makes incoming data trustworthy, versioned, and reproducible.
- **Experiment:** one model-training run with recorded settings and results.
- **MLflow:** tracks experiment parameters, metrics, and model artifacts.
- **Candidate:** a model proposed as a replacement for Production.
- **Production:** the model currently used for scoring decisions.
- **Drift:** live data or predictions are changing relative to the model's reference population.
- **Retraining:** create new model versions using the selected versioned dataset.
- **Deployment / Promotion:** make an approved Candidate the Production model.
