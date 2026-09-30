# Profit-Aware E-Commerce Order Cancellation Risk

A production-style machine-learning prototype that helps e-commerce operations teams decide whether an order should be released to fulfillment immediately or briefly held for verification.

The system predicts cancellation risk, translates that risk into an expected business value, and gives an Operations Manager a simple action recommendation. Developers get separate tools for DataOps, model experimentation, MLflow tracking, deployment, and monitoring.

## What problem does this solve?

An order that is canceled after warehouse work begins can waste picking, packing, inventory, payment-processing, customer-service, and logistics effort. Not every cancellation can be prevented, and reviewing every order would create unnecessary customer friction.

This application asks a narrower business question:

> **Is verifying this order expected to save more money than the verification costs?**

The ML model estimates cancellation probability. The business layer then combines that probability with configurable assumptions about cancellation loss and verification cost.

## Who is this for?

You do not need to be a machine-learning specialist to use the application.

- **Operations Manager:** reviews incoming orders, sees risk and expected savings, and decides whether to release or verify an order.
- **Developer / Data Scientist:** manages datasets, model experiments, candidates, deployment, monitoring, and retraining.
- **Finance / Business stakeholder:** can understand the economic assumptions and historical business backtests without reading model code.

## What happens when the app starts?

The application **does not train a model at startup**.

A pretrained LightGBM model is packaged as the initial Production model, so scoring is immediately available:

```text
Open app
   ↓
Choose Operations Manager or Developer
   ↓
Load packaged Production model
   ↓
Ready to score orders
```

Training is an explicit developer workflow and is isolated from day-to-day Operations use.

## Starter model registry

The repository ships with trained versions of five model families:

| Model | Initial role | Compute |
|---|---|---|
| Logistic Regression | Ready / baseline | CPU |
| Random Forest | Ready | CPU |
| Extra Trees | Ready | CPU |
| LightGBM | **Production** | CPU packaged model; GPU preferred in Colab retraining |
| XGBoost | Ready | GPU preferred in Colab retraining |

The starter Random Forest, Extra Trees, Logistic Regression, and XGBoost artifacts are packaged as fast benchmark models. Formal retraining in Colab uses the complete selected dataset version.

## Application roles

### Operations Manager

The Operations workspace focuses only on order decisions:

- Operations Dashboard
- Score Order
- Decision History

The prototype simulation behaves like a customer order workflow:

```text
Customer places order
        ↓
Automated cancellation-risk screening
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

The Developer workspace exposes the ML-system lifecycle:

- Developer Dashboard
- DataOps
- Model Registry & Deployment
- Model Monitoring
- Experiments / MLflow
- System Status

## End-to-end ML workflow

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
Training request
      ↓
Google Colab / Colab Enterprise
      ↓
Train five models
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
Degradation detected?
      ↓ yes
Retraining request
```

Production deployment is intentionally **human-approved**. Automated training can create a Candidate, but it cannot silently replace the Production model.

## DataOps

The DataOps pipeline converts incoming item-level e-commerce transactions into a reproducible order-level data product.

It performs:

1. File fingerprinting and duplicate-batch protection
2. Raw-source preservation
3. Schema and data-quality validation
4. Quarantine of invalid or inconsistent records
5. Deterministic item-to-order aggregation
6. Leakage-safe customer-history recomputation
7. Immutable dataset version creation
8. Training-request creation for the new dataset version

The original Pakistan dataset is already packaged as a validated order-level seed dataset. Developers normally upload only new batches.

## Heavy training in Google Colab

The computational workflow lives in:

```text
notebooks/end_to_end_ml_workflow.ipynb
```

For ordinary Colab:

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Set `REPO_URL` to this GitHub repository.
4. Run all cells.
5. MLflow artifacts and model outputs are written to Google Drive.
6. Download the generated `candidate_package__*.zip`.
7. In the Streamlit app, open **Developer → Model Registry & Deployment**.
8. Import the candidate package.
9. Review it and click **Promote Candidate to Production** if approved.

### Which models use the T4?

- **XGBoost:** GPU preferred, CPU fallback
- **LightGBM:** GPU preferred, CPU fallback
- **Random Forest:** CPU
- **Extra Trees:** CPU
- **Logistic Regression:** CPU

The experiment metadata records the device that was actually used.

## MLflow experiment tracking

Every formal five-model training suite uses one named experiment, for example:

```text
cancellation-risk__pakistan_seed_v1__20260930
```

Individual runs receive readable, reproducible names:

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

The training workflow first applies qualification gates for:

- discrimination
- probability calibration
- recall at high precision
- positive estimated business value

Qualified models are then compared using a transparent weighted score that combines technical performance and business value.

The highest-ranked qualified model becomes **Candidate**.

The Candidate still requires a developer to click:

```text
Promote Candidate to Production
```

## Monitoring and retraining

Monitoring separates three concepts:

### 1. Runtime health

Tracks production-like scoring traffic, including:

- prediction volume
- average predicted risk
- inference latency
- pipeline reliability
- verification rate

### 2. Drift

Compares production-like traffic with the Production model's reference population.

Simulation and historical-evaluation traffic remain excluded from drift calculations so demo sampling cannot create false alerts.

### 3. Actual performance

When final outcomes become available, a developer can upload an outcome file with:

```text
order_id,final_outcome
100001,Completed
100002,Canceled
```

The application can then calculate observed production metrics instead of relying only on drift.

### Retraining policy

The system uses a hybrid trigger:

- **Drift** is an early warning.
- **Confirmed labeled degradation** is stronger evidence.
- One small or noisy batch should not launch a full retraining job.
- Sustained severe drift or confirmed metric degradation creates a retraining request.

Ordinary Colab provides the practical classroom workflow. An optional Colab Enterprise configuration can submit a notebook execution automatically when the required Google Cloud configuration and credentials are available.

Even after automated retraining, deployment remains manual.

## Business decision logic

The Operations UI deliberately uses plain language.

The configurable assumptions are:

- **Loss if a canceled order reaches fulfillment**
- **Loss prevented by verification**
- **Cost to verify one order**
- **Extra cost if a good order is verified**

The user-facing application displays money in USD using a current PKR-to-USD conversion when available. Model features remain in the original PKR units so the trained feature contract is not changed.

## Reproducibility

Formal model runs record:

- immutable dataset version
- time-based split
- random seed
- experiment and run names
- Git commit SHA
- environment and library versions
- training device
- hyperparameters
- metrics
- business assumptions
- serialized model artifact

This makes it possible to trace a Production model back to the code, data, configuration, and experiment that created it.

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

The app is available locally at:

```text
http://localhost:8501
```

No password is required in the classroom prototype. Choose the Operations Manager or Developer workspace from the landing page.

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

CI installs the project, runs automated tests, verifies that packaged inference can load, and checks that the Docker image builds.

If a file-copy operation skipped hidden folders, restore the workflow with:

```bash
mkdir -p .github/workflows
cp GITHUB_ACTIONS_CI.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI"
git push
```

## Optional Colab Enterprise automation

The app only enables the **Trigger Colab Enterprise Retraining** action when the required environment variables are configured:

```text
COLAB_ENTERPRISE_PROJECT_ID
COLAB_ENTERPRISE_LOCATION
COLAB_ENTERPRISE_RUNTIME_TEMPLATE_ID
COLAB_ENTERPRISE_NOTEBOOK_GCS_URI
COLAB_ENTERPRISE_OUTPUT_GCS_URI
COLAB_ENTERPRISE_SERVICE_ACCOUNT
AUTO_TRIGGER_COLAB_ENTERPRISE=true
```

`COLAB_ENTERPRISE_EXECUTION_USER` can be used instead of a service account.

Google Application Default Credentials must also be available to the deployed developer environment.

## Project structure

```text
app.py                         Streamlit entry point
views/                         Role-specific application pages
src/data/                      Ingestion, validation, aggregation, versioning
src/features/                  Feature construction and inference contract
src/models/                    Training, registry, inference, five-model suite
src/monitoring/                Runtime, drift, business and performance metrics
src/retraining.py              Hybrid degradation and retraining orchestration
notebooks/end_to_end_ml_workflow.ipynb
                               Colab heavy-compute workflow
artifacts/model_registry/      Packaged trained model suite
artifacts/production_model.pkl Initial/current deployed artifact
artifacts/active_model.json    Production model metadata
tests/                         Automated regression tests
.github/workflows/ci.yml       Continuous integration
Dockerfile                     Container deployment
```

## Important prototype limitations

This is a school ML-system prototype, not a production commerce platform.

- Passwordless role selection demonstrates role-specific UX, not enterprise authentication.
- Business-cost assumptions are configurable because the public dataset does not contain the retailer's real warehouse or verification costs.
- Runtime state on some hosted Streamlit environments may be ephemeral unless external persistence is configured.
- The packaged Pakistan dataset covers historical behavior and should not be assumed to represent every modern e-commerce population.
- Automatic retraining does not mean automatic deployment. Candidate promotion intentionally remains a human governance step.

## Dataset

The project uses **Pakistan's Largest E-Commerce Dataset** by Zeeshan-ul-hassan Usmani and collaborator. The raw public data is item-level; the ML prediction unit is one order.

The packaged seed contains the validated, aggregated complete/canceled order population used by the initial model.

## Quick glossary

- **DataOps:** makes incoming data trustworthy, versioned, and reproducible.
- **Experiment:** one model-training run with recorded settings and results.
- **MLflow:** tracks experiments, metrics, parameters, and model artifacts.
- **Candidate:** a model proposed as a replacement for Production.
- **Production:** the model currently used for real scoring decisions.
- **Drift:** live data or predictions are changing relative to the model's reference population.
- **Retraining:** train new model versions using newer data.
- **Deployment / Promotion:** make an approved Candidate the Production model.
