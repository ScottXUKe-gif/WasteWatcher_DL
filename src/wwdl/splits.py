"""Expanding-window rolling origins.

An origin ``t`` means the model has observed weeks ``0 .. t-1`` and forecasts week
``t + h - 1``. Ported from microbe-repr-dynamic-range-v1 splits.py, simplified to
direct multi-horizon evaluation without a validation block.
"""

from __future__ import annotations

import numpy as np


def rolling_origins(T: int, min_train: int, horizon: int) -> list[int]:
    """All origins t with min_train <= t and t + horizon - 1 <= T - 1."""
    if min_train < 2 or horizon < 1:
        raise ValueError("min_train >= 2 and horizon >= 1 required")
    origins = list(range(min_train, T - horizon + 1))
    if not origins:
        raise ValueError("no room for a rolling origin")
    return origins


def inner_origins(T_train: int, horizon: int, n_valid: int = 4) -> list[int]:
    """Origins inside a training window for hyper-parameter selection.

    Uses the last ``n_valid`` reachable targets of the training window so the
    selection never touches anything after the outer origin.
    """
    last = T_train - horizon  # largest origin whose target is inside the window
    first = max(2, last - n_valid + 1)
    return list(range(first, last + 1))


def check_no_leak(origin: int, horizon: int, used_indices: np.ndarray) -> None:
    """Assert every index a model touched is before the origin."""
    if used_indices.size and used_indices.max() >= origin:
        raise AssertionError(f"leak: index {used_indices.max()} >= origin {origin}")
