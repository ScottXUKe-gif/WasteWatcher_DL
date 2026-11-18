"""Named forecasters with a uniform (fold, h) -> (N,) interface."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from . import baselines as B
from . import lowrank as L
from .transform import FoldData

Model = Callable[[FoldData, int], np.ndarray]


def _z(fn):  # adapt a (z, h) function to the (fold, h) interface
    return lambda fold, h: fn(fold.train_z, h)


MODELS: dict[str, Model] = {
    "persistence": _z(B.persistence),
    "train_mean": _z(B.train_mean),
    "rolling_mean_4": _z(partial(B.rolling_mean, window=4)),
    "rolling_mean_8": _z(partial(B.rolling_mean, window=8)),
    "ewma": _z(B.ewma),
    "mean_revert_shared": _z(B.mean_revert_shared),
    "mean_revert_taxon": _z(B.mean_revert_taxon),
    "ridge_ar1": _z(partial(B.ridge_ar, p=1)),
    "ridge_ar2": _z(partial(B.ridge_ar, p=2)),
    "ridge_ar3": _z(partial(B.ridge_ar, p=3)),
    "pca_var2": _z(partial(L.pca_var, k=2)),
    "pca_var4": _z(partial(L.pca_var, k=4)),
    "pca_var8": _z(partial(L.pca_var, k=8)),
    "pls_var": _z(partial(L.pls_var, k=4)),
    "ridge_ar1_env_lag": lambda f, h: B.ridge_ar1_env(f.train_z, h, f.env_train, f.env_target, oracle=False),
    "ridge_ar1_env_oracle": lambda f, h: B.ridge_ar1_env(f.train_z, h, f.env_train, f.env_target, oracle=True),
}

TRIVIAL = {"persistence", "train_mean", "rolling_mean_4", "rolling_mean_8", "ewma"}
ORACLE = {"ridge_ar1_env_oracle"}


def shuffled(fold: FoldData, rng: np.random.Generator) -> FoldData:
    """Same fold with the training rows in random time order (the control)."""
    perm = rng.permutation(len(fold.train_z))
    return FoldData(
        train_z=fold.train_z[perm],
        target_z=fold.target_z,
        env_train=fold.env_train[perm],
        env_target=fold.env_target,
        scaler=fold.scaler,
        train_counts=fold.train_counts[perm],
    )


def z_model(name: str) -> Callable[[np.ndarray, int], np.ndarray]:
    """Expose a z-only model for the innovation-interval code."""
    fn = MODELS[name]
    dummy = lambda z: FoldData(z, None, None, None, None, None)  # noqa: E731
    return lambda z, h: fn(dummy(z), h)
