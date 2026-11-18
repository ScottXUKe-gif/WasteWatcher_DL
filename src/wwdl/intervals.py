"""Prediction intervals from historical in-sample innovations (train-only).

Ported in spirit from Wastewatcher/scripts/evaluate_innovation_calibration.py: the
only method in the previous repos whose coverage was actually calibrated.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Forecaster = Callable[[np.ndarray, int], np.ndarray]


def innovation_residuals(model: Forecaster, z: np.ndarray, h: int, min_origin: int = 8) -> np.ndarray:
    """Residuals of ``model`` at every origin inside the training window.

    Returns (n_inner, N). Each row is a genuine out-of-sample residual within the
    training window, so nothing after the outer origin is touched.
    """
    origins = range(min_origin, len(z) - h + 1)
    return np.stack([z[o + h - 1] - model(z[:o], h) for o in origins])


def interval_from_residuals(
    pred: np.ndarray, resid: np.ndarray, level: float, min_scale: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Central interval: per-taxon scale, pooled standardized quantiles.

    Per-taxon quantiles from ~10 residuals would be noise; standardizing by each
    taxon's residual std and pooling the quantiles across taxa is the compromise.
    """
    scale = np.maximum(resid.std(axis=0), min_scale)
    standardized = (resid / scale).ravel()
    lo_q, hi_q = np.quantile(standardized, [(1 - level) / 2, 1 - (1 - level) / 2])
    return pred + lo_q * scale, pred + hi_q * scale


def coverage(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray | None = None) -> float:
    inside = (truth >= lo) & (truth <= hi)
    return float(inside[mask].mean() if mask is not None else inside.mean())
