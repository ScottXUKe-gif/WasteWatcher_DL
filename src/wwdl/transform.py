"""Train-only compositional transforms: pseudocount, CLR, z-score.

Rows are time points, columns are taxa. Ported from
microbe-repr/src/microbe_repr/preprocessing.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def adaptive_pseudocount(counts: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Per-time-point pseudocount = fraction * smallest positive count in that sample.

    Depends only on the sample itself, so it carries no information across time.
    """
    X = np.asarray(counts, dtype=float)
    deltas = np.empty(X.shape[0])
    for t, row in enumerate(X):
        pos = row[row > 0]
        deltas[t] = fraction * pos.min() if pos.size else 0.5
    return deltas


def clr_transform(counts: np.ndarray, pseudocount: np.ndarray | None = None) -> np.ndarray:
    """Centered log-ratio of (T, N) counts. Each row sums to zero."""
    X = np.asarray(counts, dtype=float)
    if X.ndim != 2 or np.any(X < 0):
        raise ValueError("counts must be a non-negative (T, N) array")
    delta = adaptive_pseudocount(X) if pseudocount is None else np.asarray(pseudocount, dtype=float)
    log_x = np.log(X + delta[:, None])
    return log_x - log_x.mean(axis=1, keepdims=True)


def inverse_clr(clr: np.ndarray) -> np.ndarray:
    """Map CLR rows back to compositions summing to one."""
    Y = np.asarray(clr, dtype=float)
    e = np.exp(Y - Y.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class ZScaler:
    """Per-taxon standardization fit on the training window only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray, min_std: float = 1e-6) -> "ZScaler":
        return cls(mean=train.mean(axis=0), std=np.maximum(train.std(axis=0), min_std))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


@dataclass
class FoldData:
    """Everything a model may see for one fold, plus the held-out target."""

    train_z: np.ndarray  # (T_train, N) standardized CLR
    target_z: np.ndarray  # (N,) standardized CLR at origin + h - 1
    env_train: np.ndarray  # (T_train, E) standardized covariates
    env_target: np.ndarray  # (E,) covariates at the target time (oracle only)
    scaler: ZScaler
    train_counts: np.ndarray  # (T_train, N) raw counts, for prevalence subsets


def make_fold(counts: np.ndarray, env: np.ndarray, origin: int, horizon: int) -> FoldData:
    """Build one fold. origin = number of observed weeks; target index = origin + h - 1."""
    target_idx = origin + horizon - 1
    if target_idx >= counts.shape[0]:
        raise ValueError("target lies beyond the series")
    clr = clr_transform(counts)  # per-sample pseudocount: no cross-time information
    train_clr, target_clr = clr[:origin], clr[target_idx]
    scaler = ZScaler.fit(train_clr)
    env_mean, env_std = env[:origin].mean(axis=0), np.maximum(env[:origin].std(axis=0), 1e-6)
    env_z = (env - env_mean) / env_std
    return FoldData(
        train_z=scaler.transform(train_clr),
        target_z=scaler.transform(target_clr),
        env_train=env_z[:origin],
        env_target=env_z[target_idx],
        scaler=scaler,
        train_counts=counts[:origin],
    )
