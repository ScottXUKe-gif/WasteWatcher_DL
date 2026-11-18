import numpy as np
import pytest

from wwdl import baselines as B
from wwdl import lowrank as L
from wwdl.registry import MODELS, shuffled
from wwdl.splits import inner_origins, rolling_origins
from wwdl.transform import clr_transform, inverse_clr, make_fold


def ar1_panel(T=40, N=50, phi=0.6, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((T, N))
    for t in range(1, T):
        z[t] = phi * z[t - 1] + rng.normal(size=N)
    return z


def test_clr_rows_sum_to_zero_and_invert():
    counts = np.random.default_rng(0).poisson(5, size=(6, 30)).astype(float)
    clr = clr_transform(counts)
    assert np.allclose(clr.sum(axis=1), 0, atol=1e-9)
    comp = inverse_clr(clr)
    assert np.allclose(comp.sum(axis=1), 1)
    assert np.allclose(clr_transform(comp * 1e6, pseudocount=np.zeros(6)), clr, atol=1e-6)


def test_rolling_origins_counts():
    assert rolling_origins(42, 20, 1) == list(range(20, 42))
    assert len(rolling_origins(42, 20, 6)) == 17
    assert all(o + 1 - 1 < 20 for o in inner_origins(20, 1))


def test_mean_revert_recovers_phi():
    z = ar1_panel(T=400, N=200, phi=0.6)
    assert abs(B._phi_pooled(z) - 0.6) < 0.03


def test_ar_models_beat_persistence_on_ar1():
    z = ar1_panel(T=60, N=200, phi=0.6)
    train, target = z[:-1], z[-1]
    mse = lambda p: np.mean((p - target) ** 2)
    pers = mse(B.persistence(train, 1))
    for fn in (B.mean_revert_shared, B.mean_revert_taxon, lambda zz, h: B.ridge_ar(zz, h, 1)):
        assert mse(fn(train, 1)) < pers


def test_pls_and_pca_beat_persistence_on_lowrank_dynamics():
    rng = np.random.default_rng(1)
    T, N, k = 60, 300, 2
    A = np.array([[0.7, 0.2], [-0.2, 0.7]])
    f = np.zeros((T, k))
    for t in range(1, T):
        f[t] = f[t - 1] @ A + 0.3 * rng.normal(size=k)
    load = rng.normal(size=(k, N))
    z = f @ load + 0.3 * rng.normal(size=(T, N))
    train, target = z[:-1], z[-1]
    mse = lambda p: np.mean((p - target) ** 2)
    pers = mse(B.persistence(train, 1))
    assert mse(L.pca_var(train, 1, 2)) < pers
    assert mse(L.pls_var(train, 1, 2)) < pers


def test_no_leakage_future_corruption_does_not_change_predictions():
    rng = np.random.default_rng(2)
    counts = rng.poisson(20, size=(42, 80)).astype(float) + 1
    env = rng.normal(size=(42, 5))
    origin, h = 20, 2
    fold = make_fold(counts, env, origin, h)
    corrupted = counts.copy()
    corrupted[origin:] = rng.poisson(200, size=(42 - origin, 80))
    env2 = env.copy()
    env2[origin:] = 99.0
    fold2 = make_fold(corrupted, env2, origin, h)
    for name, model in MODELS.items():
        if name == "ridge_ar1_env_oracle":
            continue  # uses target-time covariates by construction; labeled an upper bound
        a, b = model(fold, h), model(fold2, h)
        assert np.allclose(a, b, atol=1e-8), f"{name} leaked future information"


def test_shuffled_fold_keeps_target():
    rng = np.random.default_rng(3)
    counts = rng.poisson(20, size=(30, 20)).astype(float) + 1
    fold = make_fold(counts, rng.normal(size=(30, 5)), 20, 1)
    sh = shuffled(fold, rng)
    assert np.allclose(sh.target_z, fold.target_z)
    assert not np.allclose(sh.train_z, fold.train_z)


def test_registry_outputs_have_right_shape():
    rng = np.random.default_rng(4)
    counts = rng.poisson(20, size=(30, 40)).astype(float) + 1
    fold = make_fold(counts, rng.normal(size=(30, 5)), 20, 3)
    for name, model in MODELS.items():
        out = model(fold, 3)
        assert out.shape == (40,), name
        assert np.all(np.isfinite(out)), name
