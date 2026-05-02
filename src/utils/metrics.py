"""
metrics.py - Calibration metrics (ECE, reliability curves)
=============================================================
Advanced evaluation metrics for model calibration.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value (lower is better)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability (calibration) curve data.

    Returns:
        Tuple of (mean_predicted_proba, fraction_of_positives, bin_counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_pred = []
    frac_pos = []
    counts = []

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
        counts.append(mask.sum())

    return np.array(mean_pred), np.array(frac_pos), np.array(counts)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    return np.mean((y_prob - y_true) ** 2)
