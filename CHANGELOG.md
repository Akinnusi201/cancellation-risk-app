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
