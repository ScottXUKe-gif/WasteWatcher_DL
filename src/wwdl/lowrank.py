"""Low-rank direct forecasters: PCA + ridge VAR, and NIPALS PLS.

Nothing here ever forms an N x N matrix, so these run on all 20k taxa.
"""

from __future__ import annotations

import numpy as np

from .baselines import LAMBDA_GRID, select_lambda


def _ridge_multi(X: np.ndarray, Y: np.ndarray, x_new: np.ndarray, lam: float) -> np.ndarray:
    """Shared-design ridge: X (n, k), Y (n, m), x_new (k,) -> (m,)."""
    k = X.shape[1]
    beta = np.linalg.solve(X.T @ X + lam * np.eye(k), X.T @ Y)
    return x_new @ beta


def _pca_var_once(z: np.ndarray, h: int, k: int, lam: float) -> np.ndarray:
    # z is train-centered; PCA via thin SVD of (T, N)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    comps = vt[:k]  # (k, N)
    f = z @ comps.T  # (T, k)
    X = np.concatenate([f[:-h], np.ones((len(f) - h, 1))], axis=1)
    x_new = np.concatenate([f[-1], [1.0]])
    f_pred = _ridge_multi(X, f[h:], x_new, lam)
    return f_pred @ comps


def pca_var(z: np.ndarray, h: int, k: int) -> np.ndarray:
    k = min(k, len(z) - h - 1)
    lam = select_lambda(lambda zz, hh, l: _pca_var_once(zz, hh, min(k, len(zz) - hh - 1), l), z, h)
    return _pca_var_once(z, h, k, lam)


def nipals_pls(X: np.ndarray, Y: np.ndarray, k: int, n_iter: int = 200, tol: float = 1e-8):
    """PLS2 weights W, loadings P and regression Q with k components, via NIPALS."""
    X, Y = X.copy(), Y.copy()
    W, P, Q = [], [], []
    for _ in range(k):
        u = Y[:, np.argmax(Y.var(axis=0))].copy()
        for _ in range(n_iter):
            w = X.T @ u
            w /= np.linalg.norm(w) + 1e-12
            t = X @ w
            q = Y.T @ t / (t @ t + 1e-12)
            u_new = Y @ q / (q @ q + 1e-12)
            if np.linalg.norm(u_new - u) < tol:
                u = u_new
                break
            u = u_new
        t = X @ w
        p = X.T @ t / (t @ t + 1e-12)
        q = Y.T @ t / (t @ t + 1e-12)
        X -= np.outer(t, p)
        Y -= np.outer(t, q)
        W.append(w), P.append(p), Q.append(q)
    return np.array(W), np.array(P), np.array(Q)


def pls_predict(W: np.ndarray, P: np.ndarray, Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = x.copy()
    y = np.zeros(Q.shape[1])
    for w, p, q in zip(W, P, Q):
        t = x @ w
        x -= t * p
        y += t * q
    return y


def pls_var(z: np.ndarray, h: int, k: int = 4) -> np.ndarray:
    """Direct low-rank regression z[t+h] ~ z[t] with k PLS components."""
    k = min(k, len(z) - h - 1)
    W, P, Q = nipals_pls(z[:-h], z[h:], k)
    return pls_predict(W, P, Q, z[-1])
