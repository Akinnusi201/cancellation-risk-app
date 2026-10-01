import pandas as pd
import streamlit as st

from src.auth import require_role
from src.config import MLFLOW_TRACKING_URI
from src.currency import fetch_pkr_to_usd_rate, pkr_to_usd
from src.models.registry import list_registered_models, set_candidate
from src.models.benchmark import load_final_benchmark, load_final_benchmark_table
from src.models.suite import DISPLAY_NAMES, model_parameter_defaults
import src.retraining as retraining

require_role("developer")
st.title("🧪 Experiments & MLflow")
st.caption(
    "Tune and run a model directly in the app, compare reproducible runs, and inspect MLflow tracking. "
    "Manual experiments are registered as Ready and never change Production automatically."
)


def _run_progress():
    stage_slot = st.empty()
    progress_slot = st.empty()
    detail_slot = st.empty()
    completed_slot = st.empty()
    current = {"stage": None}
    completed = []

    def progress(stage_key, fraction, message):
        if stage_key != current["stage"]:
            if current["stage"] is not None:
                completed.append(current["stage"].replace("_", " ").title())
            progress_slot.empty()
            detail_slot.empty()
            current["stage"] = stage_key
        stage_slot.markdown(f"#### {stage_key.replace('_', ' ').title()}")
        progress_slot.progress(max(0, min(100, int(float(fraction) * 100))), text=message)
        detail_slot.caption(
            "This experiment runs in the current app environment. Its parameters, metrics, model artifact, dataset version, "
            "random seed, and environment metadata are recorded for reproducibility."
        )
        if completed:
            completed_slot.caption("Completed: " + " · ".join(completed[-6:]))

    return progress, stage_slot, progress_slot, detail_slot


def _hyperparameter_controls(family):
    defaults = model_parameter_defaults(family)
    if family == "logistic_regression":
        c1, c2 = st.columns(2)
        c = c1.number_input("Regularization strength (C)", min_value=0.01, max_value=20.0, value=float(defaults["C"]), step=0.10)
        max_iter = c2.number_input("Maximum iterations", min_value=100, max_value=2000, value=int(defaults["max_iter"]), step=100)
        class_weight_label = st.selectbox("Class weighting", ["None", "Balanced"], index=0)
        return {"C": float(c), "max_iter": int(max_iter), "class_weight": None if class_weight_label == "None" else "balanced"}

    if family in {"random_forest", "extra_trees"}:
        c1, c2 = st.columns(2)
        n_estimators = c1.number_input("Number of trees", min_value=50, max_value=600, value=int(defaults["n_estimators"]), step=25)
        max_depth_default = int(defaults["max_depth"] or 24)
        max_depth = c2.number_input("Maximum tree depth", min_value=4, max_value=60, value=max_depth_default, step=1)
        c3, c4 = st.columns(2)
        min_samples_leaf = c3.number_input("Minimum samples per leaf", min_value=1, max_value=20, value=int(defaults["min_samples_leaf"]), step=1)
        max_features = c4.selectbox("Features considered at each split", ["sqrt", "log2"], index=0)
        return {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
        }

    if family == "lightgbm":
        c1, c2, c3 = st.columns(3)
        n_estimators = c1.number_input("Boosting trees", 50, 800, int(defaults["n_estimators"]), 25)
        learning_rate = c2.number_input("Learning rate", 0.005, 0.30, float(defaults["learning_rate"]), 0.005, format="%.3f")
        num_leaves = c3.number_input("Leaves per tree", 8, 128, int(defaults["num_leaves"]), 1)
        c4, c5, c6 = st.columns(3)
        min_child_samples = c4.number_input("Minimum child samples", 5, 100, int(defaults["min_child_samples"]), 5)
        subsample = c5.slider("Row sample", 0.50, 1.00, float(defaults["subsample"]), 0.05)
        colsample = c6.slider("Feature sample", 0.50, 1.00, float(defaults["colsample_bytree"]), 0.05)
        return {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "min_child_samples": int(min_child_samples),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample),
        }

    if family == "xgboost":
        c1, c2, c3 = st.columns(3)
        n_estimators = c1.number_input("Boosting trees", 50, 800, int(defaults["n_estimators"]), 25)
        learning_rate = c2.number_input("Learning rate", 0.005, 0.30, float(defaults["learning_rate"]), 0.005, format="%.3f")
        max_depth = c3.number_input("Maximum tree depth", 2, 16, int(defaults["max_depth"]), 1)
        c4, c5, c6 = st.columns(3)
        min_child_weight = c4.number_input("Minimum child weight", 1, 20, int(defaults["min_child_weight"]), 1)
        subsample = c5.slider("Row sample", 0.50, 1.00, float(defaults["subsample"]), 0.05)
        colsample = c6.slider("Feature sample", 0.50, 1.00, float(defaults["colsample_bytree"]), 0.05)
        reg_lambda = st.number_input("L2 regularization", 0.0, 20.0, float(defaults["reg_lambda"]), 0.25)
        return {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "min_child_weight": int(min_child_weight),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample),
            "reg_lambda": float(reg_lambda),
        }
    return {}


run_tab, runs_tab, compare_tab, benchmark_tab, tracking_tab = st.tabs([
    "Run Experiment", "Experiment Runs", "Compare Models", "Final Benchmark", "MLflow Tracking"
])

with run_tab:
    st.subheader("Manual experiment")
    st.write(
        "Use this when you want to test a model configuration yourself. Choose a versioned dataset, tune a few useful parameters, "
        "and run it immediately. The resulting run is tracked in MLflow and added to the registry as **Ready**."
    )
    try:
        versions = retraining.list_dataset_versions()
    except Exception as exc:
        versions = []
        st.warning(f"Dataset versions could not be loaded: {exc}")

    if not versions:
        st.info("Create a versioned dataset in DataOps before running an experiment.")
    else:
        version_ids = [v["dataset_version"] for v in versions]
        selected_version = st.selectbox(
            "Dataset version",
            version_ids,
            format_func=lambda vid: next((
                f"{v['dataset_version']} · {v['order_count']:,} orders" + (" · active" if v.get("active") else "")
                for v in versions if v["dataset_version"] == vid
            ), vid),
        )
        family = st.selectbox(
            "Model",
            list(DISPLAY_NAMES),
            format_func=lambda x: DISPLAY_NAMES[x],
            help="Each experiment trains one model so you can isolate the effect of the parameters you change.",
        )

        c1, c2, c3 = st.columns(3)
        scope_label = c1.selectbox("Training data", ["Full dataset", "Quick sample"])
        max_rows = c2.number_input(
            "Quick-sample rows",
            min_value=10_000,
            max_value=150_000,
            value=60_000,
            step=10_000,
            disabled=scope_label == "Full dataset",
        )
        random_seed = c3.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)
        prefer_gpu = st.toggle(
            "Prefer GPU for LightGBM/XGBoost when available",
            value=True,
            disabled=family not in {"lightgbm", "xgboost"},
        )

        st.markdown("#### Tune parameters")
        st.caption("Only the most useful parameters are exposed here so the experiment screen stays understandable.")
        hyperparameters = _hyperparameter_controls(family)

        if st.button(f"Run {DISPLAY_NAMES[family]} Experiment", type="primary", use_container_width=True):
            progress, stage_slot, progress_slot, detail_slot = _run_progress()
            try:
                result = retraining.run_manual_experiment(
                    dataset_version=selected_version,
                    model_family=family,
                    hyperparameters=hyperparameters,
                    training_scope="quick_sample" if scope_label == "Quick sample" else "full_dataset",
                    max_training_rows=int(max_rows),
                    prefer_gpu=bool(prefer_gpu),
                    random_seed=int(random_seed),
                    progress=progress,
                )
                progress_slot.empty()
                detail_slot.empty()
                stage_slot.success(f"Experiment complete: {result['run_name']}")
                st.session_state["latest_manual_experiment_id"] = result["model_id"]
            except Exception as exc:
                progress_slot.empty()
                detail_slot.empty()
                stage_slot.error(f"Experiment failed: {exc}")
                st.exception(exc)

        latest_id = st.session_state.get("latest_manual_experiment_id")
        if latest_id:
            latest = next((m for m in list_registered_models() if m.get("model_id") == latest_id), None)
            if latest:
                st.markdown("#### Latest result")
                tm = latest.get("test_metrics", {})
                b = latest.get("business_metrics", {})
                a, b1, c, d = st.columns(4)
                a.metric("ROC-AUC", f"{tm.get('roc_auc', float('nan')):.3f}")
                b1.metric("PR-AUC", f"{tm.get('pr_auc', float('nan')):.3f}")
                c.metric("Brier", f"{tm.get('brier', float('nan')):.3f}")
                d.metric("Recall @ 90% precision", f"{tm.get('recall_at_90_precision', float('nan')):.1%}")
                passed = bool(latest.get("qualification", {}).get("passed"))
                st.caption(
                    f"Status: **Ready** · Validation qualification gates: **{'Passed' if passed else 'Needs review'}** · "
                    f"MLflow run ID: `{latest.get('run_id') or 'local tracking unavailable'}`"
                )
                if latest.get("status") != "PRODUCTION":
                    allow = passed
                    if not passed:
                        st.warning("This run did not pass all validation qualification gates. Nomination requires an explicit developer override.")
                        allow = st.checkbox(
                            "Override failed qualification gates for this experiment",
                            key=f"manual_candidate_override_{latest['model_id']}",
                        )
                    if st.button("Mark latest experiment as Candidate", use_container_width=True, disabled=not allow):
                        set_candidate(latest["model_id"], "Developer selected tuned manual experiment", latest.get("qualification", {}))
                        st.success("The experiment is now Candidate. Production is unchanged until explicit promotion in Model Registry.")
                        st.rerun()

with runs_tab:
    st.subheader("Registered experiment runs")
    st.caption("Each row is one reproducible trained model. Production, Candidate, and Ready are lifecycle statuses.")
    models = list_registered_models()
    if not models:
        st.info("No model runs are registered yet.")
    else:
        rows = []
        for m in models:
            tm = m.get("test_metrics", {})
            rows.append({
                "Run name": m.get("run_name", m.get("model_id")),
                "Model": m.get("display_name"),
                "Role": m.get("run_role", "suite / packaged"),
                "Status": m.get("status", "READY"),
                "Dataset": m.get("dataset_version"),
                "Scope": m.get("training_scope"),
                "Device": m.get("training_device"),
                "ROC-AUC": tm.get("roc_auc"),
                "PR-AUC": tm.get("pr_auc"),
                "Brier": tm.get("brier"),
                "Recall @ 90% precision": tm.get("recall_at_90_precision"),
                "Trained": m.get("trained_at"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        selected = st.selectbox(
            "Run details",
            [m["model_id"] for m in models],
            format_func=lambda mid: next((f"{m['display_name']} · {m.get('run_name', mid)}" for m in models if m["model_id"] == mid), mid),
        )
        record = next(m for m in models if m["model_id"] == selected)
        c1, c2, c3 = st.columns(3)
        c1.metric("Random seed", str(record.get("random_seed", 42)))
        c2.metric("Training rows", f"{int(record.get('training_rows', 0) or 0):,}")
        c3.metric("Duration", f"{record.get('duration_seconds'):.1f}s" if record.get("duration_seconds") else "Packaged production")
        st.caption(f"Experiment: `{record.get('experiment_name', 'n/a')}` · MLflow run: `{record.get('run_id') or 'n/a'}`")
        with st.expander("Hyperparameters"):
            st.json(record.get("hyperparameters", {}))
        with st.expander("Reproducibility metadata"):
            st.json(record.get("reproducibility", {}))

with compare_tab:
    st.subheader("Model comparison")
    models = list_registered_models()
    if len(models) < 2:
        st.info("At least two registered models are required for comparison.")
    else:
        names = {m["model_id"]: f"{m['display_name']} · {m.get('status','READY')} · {m.get('run_name', m['model_id'])}" for m in models}
        choices = st.multiselect("Choose models", list(names), default=list(names)[: min(5, len(names))], format_func=lambda x: names[x])
        metrics = [
            ("ROC-AUC", "roc_auc"),
            ("PR-AUC", "pr_auc"),
            ("Brier score", "brier"),
            ("F1", "f1"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("Recall @ 90% precision", "recall_at_90_precision"),
        ]
        rows = []
        for model_id in choices:
            m = next(x for x in models if x["model_id"] == model_id)
            row = {"Model": m["display_name"], "Run": m.get("run_name"), "Status": m.get("status", "READY")}
            for label, key in metrics:
                row[label] = m.get("test_metrics", {}).get(key)
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.info("Higher ROC-AUC, PR-AUC, F1, precision, recall, and fixed-precision recall are better. Lower Brier score means better probability calibration.")

with benchmark_tab:
    st.subheader("Final fair five-model benchmark")
    st.write(
        "This benchmark is the report-ready comparison. All five model families use the **same complete dataset**, "
        "the same deterministic temporal split, the same feature schema, and the same business assumptions. "
        "Candidate qualification/ranking uses validation evidence only; the final test holdout is used only after selection."
    )
    benchmark = load_final_benchmark()
    table = load_final_benchmark_table()
    if benchmark:
        contract = benchmark.get("benchmark_contract", {})
        manifest = benchmark.get("split_manifest", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset", benchmark.get("dataset_version", "n/a"))
        c2.metric("Dataset rows", f"{int(benchmark.get('source_rows', 0) or 0):,}")
        c3.metric("Split ID", manifest.get("split_id", "n/a"))
        c4.metric("Selection evidence", "Validation only")
        st.caption(
            f"Dataset fingerprint: `{str(benchmark.get('dataset_fingerprint', 'n/a'))[:20]}…` · "
            f"Feature schema: `{benchmark.get('feature_schema_version', 'n/a')}` · "
            "Test holdout reserved for final unbiased evaluation."
        )
        if not table.empty:
            fx = fetch_pkr_to_usd_rate(timeout=1.5)
            fx_rate = float(fx["rate"])
            display = table.copy()
            if "validation_net_savings_per_1000_orders" in display.columns:
                display["validation_net_savings_per_1000_usd"] = display["validation_net_savings_per_1000_orders"].map(
                    lambda x: pkr_to_usd(float(x), fx_rate) if pd.notna(x) else None
                )
            if "test_net_savings_per_1000_orders" in display.columns:
                display["test_net_savings_per_1000_usd"] = display["test_net_savings_per_1000_orders"].map(
                    lambda x: pkr_to_usd(float(x), fx_rate) if pd.notna(x) else None
                )
            display = display.rename(columns={
                "model": "Model",
                "qualified_on_validation": "Qualified",
                "selected_candidate": "Selected by validation",
                "validation_roc_auc": "Val ROC-AUC",
                "validation_pr_auc": "Val PR-AUC",
                "validation_brier": "Val Brier",
                "validation_recall_at_90_precision": "Val Recall @ 90% Precision",
                "validation_net_savings_per_1000_usd": "Val Net Savings / 1,000 ($)",
                "test_roc_auc": "Test ROC-AUC",
                "test_pr_auc": "Test PR-AUC",
                "test_brier": "Test Brier",
                "test_recall_at_90_precision": "Test Recall @ 90% Precision",
                "test_net_savings_per_1000_usd": "Test Net Savings / 1,000 ($)",
            })
            keep = [c for c in [
                "Model", "Qualified", "Selected by validation",
                "Val ROC-AUC", "Val PR-AUC", "Val Brier", "Val Recall @ 90% Precision", "Val Net Savings / 1,000 ($)",
                "Test ROC-AUC", "Test PR-AUC", "Test Brier", "Test Recall @ 90% Precision",
                "Test Net Savings / 1,000 ($)", "training_device", "duration_seconds",
            ] if c in display.columns]
            st.dataframe(display[keep], use_container_width=True, hide_index=True)
        selected_id = benchmark.get("candidate_model_id")
        selected_model = next((m for m in benchmark.get("models", []) if m.get("model_id") == selected_id), None)
        if selected_model:
            st.info(
                f"Validation selected **{selected_model.get('display_name')}** for this benchmark. "
                "This benchmark result does not change Candidate or Production status."
            )
    else:
        st.info("No packaged full-data benchmark is available yet. Run one below to create the report-ready comparison.")

    st.markdown("#### Run a fresh full-data benchmark")
    st.caption(
        "This can take several minutes on CPU. It trains all five models on the complete selected dataset and logs every run to MLflow. "
        "It does not register or deploy a model."
    )
    try:
        benchmark_versions = retraining.list_dataset_versions()
    except Exception:
        benchmark_versions = []
    if benchmark_versions:
        benchmark_version = st.selectbox(
            "Benchmark dataset version",
            [v["dataset_version"] for v in benchmark_versions],
            key="benchmark_dataset_version",
            format_func=lambda vid: next((
                f"{v['dataset_version']} · {v['order_count']:,} orders" + (" · active" if v.get("active") else "")
                for v in benchmark_versions if v["dataset_version"] == vid
            ), vid),
        )
        benchmark_gpu = st.toggle("Prefer GPU for LightGBM/XGBoost", value=True, key="benchmark_prefer_gpu")
        if st.button("Run Full Five-Model Benchmark", use_container_width=True):
            progress, stage_slot, progress_slot, detail_slot = _run_progress()
            try:
                result = retraining.run_final_benchmark(benchmark_version, prefer_gpu=benchmark_gpu, progress=progress)
                progress_slot.empty()
                detail_slot.empty()
                stage_slot.success(
                    f"Benchmark complete. Validation-selected model: {result.get('candidate_model_id') or 'none passed gates'}"
                )
                st.rerun()
            except Exception as exc:
                progress_slot.empty()
                detail_slot.empty()
                stage_slot.error(f"Benchmark failed: {exc}")
                st.exception(exc)


with tracking_tab:
    st.subheader("MLflow tracking store")
    st.caption(f"Configured tracking URI: `{MLFLOW_TRACKING_URI}`")
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiments = mlflow.search_experiments(max_results=50)
        if not experiments:
            st.info("No runtime MLflow experiments are stored in this deployment yet. Packaged starter runs remain visible in the other tabs.")
        else:
            exp_rows = [{"Experiment": e.name, "Experiment ID": e.experiment_id, "Lifecycle": e.lifecycle_stage} for e in experiments]
            st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)
            selected_exp = st.selectbox("View MLflow experiment", experiments, format_func=lambda e: e.name)
            runs = mlflow.search_runs([selected_exp.experiment_id], max_results=200, order_by=["start_time DESC"])
            keep = [c for c in [
                "run_id", "tags.mlflow.runName", "tags.run_role", "params.model_family", "params.dataset_version", "params.training_scope", "params.training_device",
                "metrics.test_roc_auc", "metrics.test_pr_auc", "metrics.test_brier", "metrics.test_recall_at_90_precision",
                "metrics.business_net_savings_per_1000_orders",
            ] if c in runs.columns]
            st.dataframe(runs[keep], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"The MLflow tracking store is not available in this runtime: {exc}")
        st.caption("This does not affect production scoring. Manual experiments remain registered locally, and MLflow tracking resumes when its configured store is available.")
