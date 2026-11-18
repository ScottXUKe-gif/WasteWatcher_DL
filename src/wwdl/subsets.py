"""Train-defined taxon subsets. Ported from dynamic_range/subsets.py."""

from __future__ import annotations

import numpy as np


def spike_score(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(q95 - median) / MAD per column: how spiky a taxon is in the training window."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    return (np.quantile(x, 0.95, axis=0) - med) / (mad + eps)


def make_subsets(
    train_z: np.ndarray,
    train_counts: np.ndarray,
    top_k: int = 500,
    active_prevalence: float = 0.8,
) -> dict[str, np.ndarray]:
    """Boolean masks over taxa, all computed from the training window only."""
    n = train_z.shape[1]
    k = min(top_k, n)

    def top(values: np.ndarray) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        mask[np.argsort(values, kind="stable")[-k:]] = True
        return mask

    prevalence = (train_counts > 0).mean(axis=0)
    return {
        "all": np.ones(n, dtype=bool),
        "top500_variance": top(train_z.var(axis=0)),
        "top500_spike": top(spike_score(train_z)),
        "active": prevalence >= active_prevalence,
    }
