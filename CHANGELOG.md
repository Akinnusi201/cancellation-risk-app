# Changelog

## v4.2 - Full application workflow with optional demo mode

- Made full-dataset retraining the default operating mode.
- Kept deterministic sampled training as an optional quick run for demos, diagnostics, and constrained compute.
- Added GPU preference for LightGBM/XGBoost with automatic CPU fallback.
- Kept direct automatic/manual retraining, MLflow tracking, candidate selection, and human-controlled promotion.
- Upgraded Manual Order scoring to use the same Operations review lifecycle as interactive incoming orders.
- Renamed the simulation path as an optional Incoming Order Demo instead of the basis of the application.
- Added a setting to include or exclude demo simulation from Business Impact while always excluding it from technical drift monitoring.
- Added training-scope metadata to new model runs for clearer reproducibility and deployment review.
- Rewrote README and system copy to describe a functional deployment first, with course/demo simplifications documented separately.

## v4.1 - Direct prototype retraining

- Removed the external managed-notebook execution path from the prototype.
- Added Developer retraining settings with automatic/manual control.
- Added direct in-app five-model retraining with MLflow tracking.
- New data and sustained degradation can launch retraining automatically when enabled.
- Added Fast prototype and Full active dataset training scopes.
- Best qualified model is registered as Candidate automatically; deployment still requires explicit developer promotion.
- Kept the ordinary Colab notebook only as an optional heavy-compute utility.

## v4.0 - Colab end-to-end MLOps workflow
- Added a five-model starter registry: Logistic Regression, Random Forest, Extra Trees, LightGBM, and XGBoost.
- Kept the existing LightGBM artifact as the initial Production model.
- Added an end-to-end Google Colab notebook for DataOps input, five-model training, MLflow tracking, candidate selection, and candidate-package export.
- Added GPU-preferred training for LightGBM and XGBoost with CPU fallback; scikit-learn Random Forest and Extra Trees remain CPU models.
- Added reproducible experiment/run names, random seed, Git SHA, environment versions, device, data sizes, metrics, and business-value metadata.
- Added qualification gates and weighted multi-metric candidate selection.
- Rebuilt ModelOps as an intuitive model registry with Ready / Candidate / Production states and explicit manual deployment approval.
- Added candidate-package import from Colab.
- Added hybrid monitoring-driven retraining requests using sustained drift and labeled performance degradation.
- Added outcome-feedback upload for observed runtime performance metrics.
- Reworked the Experiments page into a simple MLflow-style run browser and model-comparison interface.
- Rewrote README.md as a first-deployment guide for technical and non-technical readers.

## v3.11 - Prototype customer order lifecycle
- Live Operations Simulation now mimics a customer order entering a temporary Operations review state before fulfillment.
- Added persistent prototype order states: awaiting review, released to fulfillment, and verification required.
- Operations Dashboard now shows the prototype customer order queue and current status counts.
- Live simulation orders now count toward prototype Business Impact, while remaining excluded from technical drift monitoring.
- Prototype business value follows the Operations Manager's actual decision when available; pending/released simulated orders do not claim verification savings.
- Historical Evaluation remains excluded from live prototype business impact.

## v3.10 - Monitoring import compatibility
- Prevented Model Monitoring from crashing during partial deployments where `views/7_Model_Monitoring.py` is newer than `src/monitoring/metrics.py`.
- Added a local Operations Business Impact fallback and lazy monitoring-helper resolution.
- Added a regression test for the exact import mismatch seen on Streamlit Cloud.


## v3.9 - Operations business impact
- Business Impact now uses only Operations Manager manual/batch scoring records.
- Live simulation, historical evaluation, and holdout rows are excluded from live business totals.
- Re-scored order IDs are deduplicated using the latest production-like score.
- Historical economic evaluation moved under Model Evaluation as a labeled holdout backtest.
- Live business metrics are explicitly expected values until downstream order outcomes are available.
# Changelog


## v3.8 - USD rolling-upgrade compatibility

- Removed hard imports of new currency helpers from `src/ui/common.py` across Operations Dashboard, Score Order, Decision History, and Model Monitoring.
- Added page-level fallbacks to `src.currency` so an older `common.py` cannot crash a partially upgraded Streamlit deployment.
- Kept business assumptions USD-facing even when the deployed shared UI module is older.
- Added regression coverage and repository verification for mixed-version USD deployments.

## v3.6 - Clearer business and economic language
- Replaced abstract UI terms such as intervention effectiveness and false-intervention friction with plain operational language.
- Reframed the decision as a simple question: is verifying this order expected to save more money than it costs?
- Renamed scoring outputs to Expected Money Saved and Expected Net Savings, with a plain-English explanation of the calculation.
- Updated Operations Dashboard and Decision History labels for readability.
- Simplified Business Impact from a 27-row sensitivity grid to three what-if scenarios: Conservative, Current assumptions, and Favorable.
- Kept the underlying profit-aware calculations and stored policy fields backward compatible.

## v3.5 - Production-only monitoring population
- Excluded live simulation, historical evaluation, legacy simulation, and unknown scoring modes from production drift and runtime statistics.
- Added a 100-observation minimum before monitoring can report STABLE, WATCH, or DRIFT.
- Added a monitoring-population breakdown so excluded demo traffic remains visible and auditable.
- Persisted individual batch predictions with `scoring_mode=batch` so batch traffic contributes to production monitoring.
- Added regression tests for monitoring-population isolation and the minimum-sample guard.

## v3.4 - Score-page rolling-upgrade compatibility
- Removed the hard import of `load_historical_demo_orders` from `src.ui.common`.
- Historical Evaluation now reads the packaged historical queue directly when an older UI helper module is still deployed.
- System Status distinguishes a packaged CI definition from an installed GitHub Actions workflow.
- Added a regression test for partial-upgrade compatibility.

# Refactor summary

## Production inference
- Added a pretrained LightGBM production artifact and production metadata.
- Added packaged reference and simulation datasets.
- Removed startup dependence on prior DataOps/MLflow runtime state.

## Authentication and roles
- Fixed Streamlit Cloud startup failure caused by an extra DuckDB prepared-statement parameter during seed registration.
- Moved runtime bootstrap until after role selection so the landing page is independent of the data store.
- Moved role pages out of Streamlit's special `pages/` directory to prevent automatic sidebar exposure.
- Added passwordless Operations Manager / Developer role selection for the class demo, with role-specific dynamic navigation.
- Added role-specific `st.navigation` menus.

## Operations workflow
- Added Operations Dashboard, Score Order, and Decision History.
- Added incoming-order simulation with live and historical-evaluation modes.
- Added manual and batch CSV scoring.
- Added profit-aware expected-savings calculations and audit logging of the policy inputs.

## DataOps
- DataOps now stops after validation, quarantine, aggregation, customer-history recomputation, and dataset versioning.
- Added packaged Pakistan order-level seed dataset.
- Added safe demo batch for DataOps demonstrations.
- Improved large-batch aggregation performance.
- Recomputed prior-customer cancellation history after cumulative dataset merges.

## ModelOps
- Added explicit candidate training.
- Added candidate vs production comparison.
- Added explicit promotion control; training no longer changes production automatically.
- Kept MLflow for developer experiments and candidate runs.

## Monitoring and reproducibility
- Production evaluation artifacts are packaged so monitoring works on a fresh deployment.
- Added temporal cancellation-prevalence context to highlight dataset shift.
- Added one-time baseline rebuild script.
- Added tests confirming packaged inference works without retraining and customer history spans batches.

## Computational completeness update
- Added Logistic Regression as a tracked baseline using the same temporal split and features as LightGBM.
- Added recall at 80% and 90% fixed precision to evaluation metrics.
- Added packaged holdout probabilities for business and drift analysis.
- Added aggregate estimated net savings, avoided cost per 1,000 orders, false-intervention cost, capture rate, and sensitivity analysis.
- Added core model inference latency logging and P95/maximum runtime monitoring.
- Added prediction PSI plus numeric/categorical input drift monitoring.
- Added DataOps, ModelOps, promotion, and batch-inference reliability summaries.
- Added GitHub Actions CI, packaged-model smoke testing, and Docker image builds.
- Expanded automated tests from 6 to 13.

## 2026-09-30 hotfix
- Added rolling-upgrade compatibility for scoring and monitoring.
- Added fallback business-evaluation module to prevent mixed-version ImportError.
- CI detection now recognizes any YAML workflow and reports when hidden .github files were not copied.
- Added runtime contract tests to catch partial repository upgrades.


## 2026-09-30 experiment responsiveness update
- Replaced the misleading single 0-100 MLflow progress bar with one stage-local progress bar that is cleared between steps.
- Added short live descriptions for dataset loading, temporal splitting, preprocessing, actual LightGBM tree iterations, threshold tuning, holdout evaluation, artifact generation, and MLflow logging.
- Added deterministic Fast Demo mode using up to 90,000 orders spread across the full timeline; runs are explicitly labeled in MLflow.
- Cached immutable dataset-version loading between experiments.
- Manual MLflow experiments now skip full model serialization because they cannot be promoted from that page; ModelOps candidates still log promotable model artifacts.
- Added explicit 100% completion and elapsed-time reporting so successful runs no longer appear stuck at 88%.

## v3.3 - Runtime hardening and simulation integrity
- Isolated Streamlit/MLflow progress callbacks from LightGBM hyperparameters to prevent callback serialization crashes.
- Manual experiment hyperparameters now use an explicit backend contract instead of forwarding arbitrary keyword arguments.
- Fast-demo experiments use up to 60,000 temporally distributed orders and skip plot rendering for faster live iteration.
- Split Operations simulation into a risk-stratified live queue and a natural historical holdout queue.
- Live simulation uses real historical rows selected by predicted risk bands; labels and probabilities are never altered.
- Candidate promotion now carries both live and historical simulation artifacts forward.
- Added packaged simulation integrity tests and callback-isolation regression coverage.
- Added a visible GitHub Actions recovery template and clearer repair instructions when `.github` is not committed.

## v3.7 - Live USD display
- Added live PKR→USD exchange-rate lookup using the State Bank of Pakistan feed through Frankfurter, with blended-rate and packaged fallbacks.
- Converted Operations and Business Impact monetary displays to U.S. dollars.
- Manual and batch scoring now accept USD monetary inputs and convert them back to PKR internally because the production model was trained on PKR-denominated historical data.
- Added hourly FX caching, source/date disclosure, USD batch exports, and currency regression tests.
