import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])


def recall_at_fixed_precision(y_true, probs, min_precision=0.80):
    """Return the best recall attainable while meeting a minimum precision target.

    Precision/recall are evaluated across all probability thresholds. If the requested
    precision is unattainable, recall is reported as 0 and the threshold defaults to 1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return {"recall": 0.0, "precision": 0.0, "threshold": 1.0}

    # sklearn returns one extra precision/recall point with no corresponding threshold.
    p = precision[:-1]
    r = recall[:-1]
    eligible = np.where(p >= float(min_precision))[0]
    if len(eligible) == 0:
        return {"recall": 0.0, "precision": float(np.max(p)), "threshold": 1.0}

    # Maximize recall subject to the precision constraint. In a tie, use the lowest
    # threshold so the operational policy captures as many orders as possible.
    best_recall = np.max(r[eligible])
    tied = eligible[np.where(np.isclose(r[eligible], best_recall))[0]]
    i = int(tied[np.argmin(thresholds[tied])])
    return {
        "recall": float(r[i]),
        "precision": float(p[i]),
        "threshold": float(thresholds[i]),
    }


def metrics(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    result = {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    # 80% is retained for continuity with the project plan. 90% is more useful on
    # the temporally shifted holdout, whose cancellation prevalence is itself above 80%.
    for fixed_precision in (0.80, 0.90):
        fixed = recall_at_fixed_precision(y_true, probs, min_precision=fixed_precision)
        pct = int(round(fixed_precision * 100))
        result[f"recall_at_{pct}_precision"] = fixed["recall"]
        result[f"precision_at_{pct}_precision"] = fixed["precision"]
        result[f"threshold_at_{pct}_precision"] = fixed["threshold"]
    return result


def selection_score(m):
    # Equal emphasis on ranking quality (ROC/PR) with a modest calibration penalty.
    return 0.40 * m["roc_auc"] + 0.40 * m["pr_auc"] + 0.20 * (1.0 - m["brier"])
