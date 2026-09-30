from pathlib import Path
import pandas as pd
import streamlit as st

from src.auth import require_role
from src.config import ROOT
from src.models.registry import active_metadata
from src.models.train import list_candidates

require_role("developer")
st.title("📈 Model Monitoring")
st.caption("Production metrics are packaged with the active model, so monitoring works even before any runtime MLflow experiments exist.")

meta = active_metadata()
if not meta:
    st.error("No production model metadata found.")
    st.stop()

m = meta.get("test_metrics", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Production Model", f"LightGBM {meta.get('model_version')}")
c2.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
c3.metric("PR-AUC", f"{m.get('pr_auc', float('nan')):.3f}")
c4.metric("Brier Score", f"{m.get('brier', float('nan')):.3f}")
st.caption(f"Dataset {meta.get('dataset_version')} • threshold {float(meta.get('threshold', .5)):.1%} • source {meta.get('source', 'packaged')}")
prev = meta.get("split_cancellation_prevalence", {})
if prev:
    st.info(
        f"Temporal cancellation prevalence changed from {prev.get('train', 0):.1%} in train to "
        f"{prev.get('validation', 0):.1%} in validation and {prev.get('test', 0):.1%} in test. "
        "That distribution shift is important context when comparing metrics and calibration."
    )

eval_dir = Path(meta.get("evaluation_dir", "artifacts/production_evaluation"))
if not eval_dir.is_absolute():
    eval_dir = ROOT / eval_dir
st.subheader("Production Evaluation")
for title, name in [
    ("ROC Curve", "test_roc.png"),
    ("Precision-Recall Curve", "test_pr.png"),
    ("Calibration Plot", "test_calibration.png"),
]:
    p = eval_dir / name
    if p.exists():
        st.markdown(f"#### {title}")
        st.image(str(p))

fi = eval_dir / "feature_importance.csv"
if fi.exists():
    st.markdown("#### Feature Importance")
    x = pd.read_csv(fi).head(15).set_index("feature")
    st.bar_chart(x)

candidates = list_candidates()
if candidates:
    st.subheader("Runtime Candidate History")
    rows = []
    for c in candidates:
        rows.append({
            "candidate_id": c["candidate_id"],
            "created_at": c["created_at"],
            "dataset_version": c["dataset_version"],
            "threshold": c["threshold"],
            **{f"test_{k}": v for k, v in c.get("test_metrics", {}).items()},
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
