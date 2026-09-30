"""Utilities for constructing honest, useful Operations simulation queues.

Live simulation is intentionally stratified by *predicted risk* so a demo contains
low-, medium-, and high-risk decisions. Historical evaluation remains a natural
sample of the temporal holdout and is used when representative prevalence matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE

LOW_MAX = 0.35
HIGH_MIN = 0.70


def risk_band(probability: float) -> str:
    p = float(probability)
    if p < LOW_MAX:
        return "LOW"
    if p < HIGH_MIN:
        return "MEDIUM"
    return "HIGH"


def build_balanced_live_queue(
    rows: pd.DataFrame,
    probabilities,
    max_rows: int = 2400,
    proportions: tuple[float, float, float] = (0.35, 0.30, 0.35),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a deterministic risk-stratified queue for live operational demos.

    The function never changes labels, features, or model probabilities. It only
    chooses which real historical rows appear in the live simulation. If a risk
    band is too small, the remaining capacity is filled from unused rows.
    """
    if rows is None or len(rows) == 0:
        return pd.DataFrame()

    frame = rows.copy().reset_index(drop=True)
    probs = np.asarray(probabilities, dtype=float)
    if len(probs) != len(frame):
        raise ValueError("probabilities must contain one value per simulation row")

    frame["simulation_probability"] = probs
    frame["simulation_risk_band"] = [risk_band(p) for p in probs]

    total = min(int(max_rows), len(frame))
    if total <= 0:
        return frame.iloc[0:0].copy()

    labels = ["LOW", "MEDIUM", "HIGH"]
    proportions = np.asarray(proportions, dtype=float)
    if len(proportions) != 3 or np.any(proportions < 0) or proportions.sum() <= 0:
        raise ValueError("proportions must contain three non-negative values")
    proportions = proportions / proportions.sum()

    desired = np.floor(proportions * total).astype(int)
    # Assign rounding remainder to the largest target groups deterministically.
    for i in np.argsort(-proportions)[: total - desired.sum()]:
        desired[i] += 1

    selected_parts = []
    selected_indices: set[int] = set()
    for label, count in zip(labels, desired):
        pool = frame[frame["simulation_risk_band"] == label]
        take = min(int(count), len(pool))
        if take:
            part = pool.sample(n=take, random_state=random_state + len(selected_parts))
            selected_parts.append(part)
            selected_indices.update(part.index.tolist())

    selected = pd.concat(selected_parts, axis=0) if selected_parts else frame.iloc[0:0].copy()

    shortfall = total - len(selected)
    if shortfall > 0:
        remaining = frame.loc[~frame.index.isin(selected_indices)]
        if len(remaining):
            fill = remaining.sample(n=min(shortfall, len(remaining)), random_state=random_state + 97)
            selected = pd.concat([selected, fill], axis=0)

    # Shuffle bands together so the queue does not appear in low/medium/high blocks.
    return selected.sample(frac=1.0, random_state=random_state + 193).reset_index(drop=True)


def queue_mix(frame: pd.DataFrame) -> dict[str, float]:
    """Return risk-band proportions for diagnostics/tests."""
    if frame is None or len(frame) == 0 or "simulation_risk_band" not in frame.columns:
        return {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0}
    counts = frame["simulation_risk_band"].value_counts(normalize=True)
    return {k: float(counts.get(k, 0.0)) for k in ["LOW", "MEDIUM", "HIGH"]}
