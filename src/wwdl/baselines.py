"""Trivial, mean-reversion and per-taxon ridge-AR baselines.

Every function takes the standardized training window ``z`` of shape (T, N) and a
horizon ``h`` and returns a (N,) forecast of ``z`` at index ``T + h - 1``.
"""

from __future__ import annotations

import numpy as np

from .splits import inner_origins

LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)


# ---------- trivial ----------
def persistence(z: np.ndarray, h: int) -> np.ndarray:
    return z[-1].copy()


def train_mean(z: np.ndarray, h: int) -> np.ndarray:
    return z.mean(axis=0)


def rolling_mean(z: np.ndarray, h: int, window: int) -> np.ndarray:
    return z[-window:].mean(axis=0)


def ewma(z: np.ndarray, h: int, alphas=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9)) -> np.ndarray:
    """EWMA level at the end of the window; alpha chosen by in-sample one-step MSE."""
    best, best_err = None, np.inf
    for a in alphas:
        level = z[0].copy()
        err = 0.0
        for t in range(1, len(z)):
            err += float(np.mean((z[t] - level) ** 2))
            level = a * z[t] + (1 - a) * level
        if err < best_err:
            best, best_err = level, err
    return best


# ---------- mean reversion (mu = 0 because z is train-centered) ----------
def _phi_pooled(z: np.ndarray) -> float:
    num = float(np.sum(z[1:] * z[:-1]))
    den = float(np.sum(z[:-1] ** 2)) + 1e-12
    return float(np.clip(num / den, -1.0, 1.0))


def mean_revert_shared(z: np.ndarray, h: int) -> np.ndarray:
    return _phi_pooled(z) ** h * z[-1]


def mean_revert_taxon(z: np.ndarray, h: int, kappa: float = 10.0) -> np.ndarray:
    """Per-taxon AR(1) coefficient shrunk toward the pooled coefficient."""
    phi0 = _phi_pooled(z)
    num = np.sum(z[1:] * z[:-1], axis=0)
    den = np.sum(z[:-1] ** 2, axis=0) + 1e-12
    n = len(z) - 1
    phi = (num + kappa * phi0 * den / n) / (den + kappa * den / n)
    return np.clip(phi, -1.0, 1.0) ** h * z[-1]


# ---------- per-taxon ridge on own lags (+ optional shared covariates) ----------
def lag_design(z: np.ndarray, p: int, h: int, env: np.ndarray | None = None, env_shift: int = 0):
    """Return X (n, F, N), y (n, N) for direct h-step regression.

    Row t uses lags z[t], z[t-1], .., z[t-p+1] and predicts z[t+h]. ``env`` rows are
    taken at time ``t + env_shift`` (0 = lagged/legal, h = oracle) and broadcast over taxa.
    """
    T, N = z.shape
    rows = range(p - 1, T - h)
    lags = np.stack([np.stack([z[t - j] for j in range(p)], axis=0) for t in rows])  # (n, p, N)
    feats = [lags, np.ones((lags.shape[0], 1, N))]
    if env is not None:
        e = np.stack([env[t + env_shift] for t in rows])  # (n, E)
        feats.append(np.broadcast_to(e[:, :, None], (len(e), e.shape[1], N)))
    X = np.concatenate(feats, axis=1)
    y = np.stack([z[t + h] for t in rows])
    return X, y


def ridge_fit_predict(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, lam: float) -> np.ndarray:
    """Batched per-taxon ridge: X (n, F, N), y (n, N), x_new (F, N) -> (N,)."""
    F = X.shape[1]
    G = np.einsum("tfn,tgn->nfg", X, X) + lam * np.eye(F)[None]
    b = np.einsum("tfn,tn->nf", X, y)
    beta = np.linalg.solve(G, b[:, :, None])[:, :, 0]  # (N, F)
    return np.sum(beta * x_new.T, axis=1)


def _ridge_ar_once(z, h, p, lam, env=None, env_shift=0, env_new=None):
    X, y = lag_design(z, p, h, env, env_shift)
    x_new = [np.stack([z[-1 - j] for j in range(p)]), np.ones((1, z.shape[1]))]
    if env is not None:
        x_new.append(np.broadcast_to(env_new[:, None], (env.shape[1], z.shape[1])))
    return ridge_fit_predict(X, y, np.concatenate(x_new, axis=0), lam)


def select_lambda(fit_predict, z: np.ndarray, h: int, grid=LAMBDA_GRID) -> float:
    """Pick lambda by inner rolling validation inside the training window."""
    errs = {}
    for lam in grid:
        e = 0.0
        for o in inner_origins(len(z), h):
            e += float(np.mean((fit_predict(z[:o], h, lam) - z[o + h - 1]) ** 2))
        errs[lam] = e
    return min(errs, key=errs.get)


def ridge_ar(z: np.ndarray, h: int, p: int) -> np.ndarray:
    lam = select_lambda(lambda zz, hh, l: _ridge_ar_once(zz, hh, p, l), z, h)
    return _ridge_ar_once(z, h, p, lam)


def ridge_ar1_env(z: np.ndarray, h: int, env: np.ndarray, env_target: np.ndarray, oracle: bool) -> np.ndarray:
    """AR(1) + covariates. oracle=False uses env at the origin; True uses env at the target."""
    shift = h if oracle else 0

    def fp(zz, hh, lam):
        T = len(zz)
        e_new = env[T - 1 + hh] if oracle else env[T - 1]
        return _ridge_ar_once(zz, hh, 1, lam, env, shift, e_new)

    lam = select_lambda(fp, z, h)
    e_new = env_target if oracle else env[len(z) - 1]
    return _ridge_ar_once(z, h, 1, lam, env, shift, e_new)
