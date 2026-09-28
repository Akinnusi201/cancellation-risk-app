import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, precision_score, recall_score


def best_f1_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])


def metrics(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def selection_score(m):
    # Equal emphasis on ranking quality (ROC/PR) with a modest calibration penalty.
    return 0.40*m["roc_auc"] + 0.40*m["pr_auc"] + 0.20*(1.0 - m["brier"])
