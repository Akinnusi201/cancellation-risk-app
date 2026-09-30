import numpy as np

from src.monitoring.metrics import categorical_total_variation, drift_label, population_stability_index


def test_population_stability_index_detects_large_shift():
    reference = np.linspace(0, 1, 200)
    same = np.linspace(0, 1, 200)
    shifted = np.linspace(2, 3, 200)
    assert population_stability_index(reference, same) < 0.01
    shifted_psi = population_stability_index(reference, shifted)
    assert shifted_psi > 0.25
    assert drift_label(shifted_psi, "psi") == "DRIFT"


def test_categorical_total_variation_is_zero_for_same_distribution():
    reference = ["cod"] * 50 + ["card"] * 50
    current = ["cod"] * 50 + ["card"] * 50
    assert categorical_total_variation(reference, current) == 0.0
