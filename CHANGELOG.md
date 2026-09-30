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
