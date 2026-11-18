"""Forecast metrics. All inputs are CLR-space arrays unless stated."""

from __future__ import annotations

import numpy as np

from .transform import inverse_clr


def _subclr(clr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CLR of the sub-composition selected by mask (re-centered)."""
    s = clr[..., mask]
    return s - s.mean(axis=-1, keepdims=True)


def fold_metrics(truth: np.ndarray, pred: np.ndarray, last: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Scalar metrics for one fold on one taxon subset.

    truth, pred, last: (N,) CLR values at the target, the forecast, and the last
    observed week. mask: (N,) boolean subset.
    """
    t, p, l = truth[mask], pred[mask], last[mask]
    ts, ps = _subclr(truth, mask), _subclr(pred, mask)
    ct, cp = inverse_clr(ts), inverse_clr(ps)
    motion_true = float(np.sum((t - l) ** 2))
    return {
        "mse": float(np.mean((t - p) ** 2)),
        "mae": float(np.mean(np.abs(t - p))),
        "aitchison": float(np.linalg.norm(ts - ps)),
        "bray_curtis": float(np.sum(np.abs(ct - cp)) / np.sum(ct + cp)),
        "energy_ratio": float(np.sum((p - l) ** 2) / (motion_true + 1e-12)),
    }


def skill(mse_model: np.ndarray, mse_ref: np.ndarray) -> float:
    """1 - pooled MSE ratio across folds."""
    return float(1.0 - np.mean(mse_model) / np.mean(mse_ref))


def win_rate(mse_model: np.ndarray, mse_ref: np.ndarray) -> float:
    return float(np.mean(mse_model < mse_ref))


def per_taxon_r2(truth: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """R^2 of each taxon across folds relative to a reference forecast.

    truth, pred, ref: (F, N). Positive means the model beats the reference on that taxon.
    """
    sse = np.sum((truth - pred) ** 2, axis=0)
    sse_ref = np.sum((truth - ref) ** 2, axis=0)
    return 1.0 - sse / (sse_ref + 1e-12)


def per_taxon_mse(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(N,) MSE per taxon across folds."""
    return np.mean((truth - pred) ** 2, axis=0)
