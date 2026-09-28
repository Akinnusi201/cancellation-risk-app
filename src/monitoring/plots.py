from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay


def save_evaluation_plots(y_true, probs, out_dir, prefix):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    fig, ax = plt.subplots(); RocCurveDisplay.from_predictions(y_true, probs, ax=ax); fig.tight_layout();
    p=out/f"{prefix}_roc.png"; fig.savefig(p, dpi=140); plt.close(fig); paths["roc_curve"]=str(p)
    fig, ax = plt.subplots(); PrecisionRecallDisplay.from_predictions(y_true, probs, ax=ax); fig.tight_layout();
    p=out/f"{prefix}_pr.png"; fig.savefig(p, dpi=140); plt.close(fig); paths["pr_curve"]=str(p)
    fig, ax = plt.subplots(); CalibrationDisplay.from_predictions(y_true, probs, n_bins=10, strategy="quantile", ax=ax); fig.tight_layout();
    p=out/f"{prefix}_calibration.png"; fig.savefig(p, dpi=140); plt.close(fig); paths["calibration_curve"]=str(p)
    return paths
