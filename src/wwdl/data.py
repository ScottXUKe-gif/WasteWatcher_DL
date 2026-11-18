"""Load the New Haven wastewater time series as (T, N) arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ENV_COLUMNS = ["Covid-19", "Temperature_min", "Temperature_max", "Temperature_avg", "Precip (in)"]


@dataclass
class Dataset:
    counts: np.ndarray  # (T, N) raw counts, float
    dates: pd.DatetimeIndex  # length T
    env: np.ndarray  # (T, E) raw covariates
    env_names: list[str]
    taxon_ids: np.ndarray  # (N,) taxID strings
    taxonomy: pd.DataFrame  # (N, ranks)

    @property
    def T(self) -> int:
        return self.counts.shape[0]

    @property
    def N(self) -> int:
        return self.counts.shape[1]


def load_nhv(path: str | Path) -> Dataset:
    """Read nhv_filtered.h5ad (obs = taxa, var = timepoints) and transpose to (T, N)."""
    a = ad.read_h5ad(path)
    X = a.layers["counts"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    counts = np.asarray(X, dtype=float).T
    dates = pd.to_datetime(a.var["Date"])
    order = np.argsort(dates.values, kind="stable")
    if not np.all(order == np.arange(len(order))):
        raise ValueError("timepoints are not in chronological order")
    env = a.var[ENV_COLUMNS].to_numpy(dtype=float)
    return Dataset(
        counts=counts,
        dates=pd.DatetimeIndex(dates),
        env=env,
        env_names=list(ENV_COLUMNS),
        taxon_ids=a.obs.index.to_numpy().astype(str),
        taxonomy=a.obs.copy(),
    )


def primary_taxon_mask(counts: np.ndarray, n_weeks: int = 20, min_prevalence: float = 0.5) -> np.ndarray:
    """Taxa with prevalence >= min_prevalence in the first n_weeks (no future information)."""
    head = counts[:n_weeks]
    return (head > 0).mean(axis=0) >= min_prevalence
