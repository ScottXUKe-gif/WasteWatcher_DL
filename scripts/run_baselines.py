"""Run every baseline on every rolling origin and save CLR-space predictions.

Usage: python scripts/run_baselines.py experiments/baseline_v1/config.yaml
Writes experiments/<name>/predictions/<taxon_set>_h<h>.npz, one per horizon.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

from wwdl.data import load_nhv, primary_taxon_mask
from wwdl.intervals import innovation_residuals, interval_from_residuals
from wwdl.registry import MODELS, shuffled, z_model
from wwdl.splits import rolling_origins
from wwdl.subsets import make_subsets
from wwdl.transform import make_fold


def run_horizon(ds, counts, cfg, h, rng):
    origins = rolling_origins(counts.shape[0], cfg["min_train"], h)
    names = list(cfg["models"]) + [f"shuffle__{m}" for m in cfg["shuffle_models"]]
    subset_names = ["all", "top500_variance", "top500_spike", "active"]
    F, N = len(origins), counts.shape[1]
    pred = np.zeros((len(names), F, N), dtype=np.float32)
    truth, last = np.zeros((F, N), np.float32), np.zeros((F, N), np.float32)
    masks = np.zeros((F, len(subset_names), N), dtype=bool)
    levels = cfg["interval_levels"]
    lo = {m: np.zeros((len(levels), F, N), np.float32) for m in cfg["interval_models"]}
    hi = {m: np.zeros_like(v) for m, v in lo.items()}

    for fi, origin in enumerate(origins):
        t0 = time.time()
        fold = make_fold(counts, ds.env, origin, h)
        sub = make_subsets(fold.train_z, fold.train_counts, **cfg["subsets"])
        masks[fi] = np.stack([sub[s] for s in subset_names])
        truth[fi] = fold.scaler.inverse(fold.target_z)
        last[fi] = fold.scaler.inverse(fold.train_z[-1])
        sh = shuffled(fold, rng)
        for mi, name in enumerate(names):
            base = name.split("__")[-1]
            z_hat = MODELS[base](sh if name.startswith("shuffle__") else fold, h)
            pred[mi, fi] = fold.scaler.inverse(z_hat)
        for m in cfg["interval_models"]:
            resid = innovation_residuals(z_model(m), fold.train_z, h)
            z_hat = pred[names.index(m), fi]
            z_hat = fold.scaler.transform(z_hat)
            for li, level in enumerate(levels):
                a, b = interval_from_residuals(z_hat, resid, level)
                lo[m][li, fi], hi[m][li, fi] = fold.scaler.inverse(a), fold.scaler.inverse(b)
        print(f"h={h} origin={origin} ({fi + 1}/{F}) {time.time() - t0:.1f}s", flush=True)

    out = dict(pred=pred, truth=truth, last=last, masks=masks, origins=np.array(origins),
               models=np.array(names), subsets=np.array(subset_names), levels=np.array(levels))
    for m in cfg["interval_models"]:
        out[f"lo__{m}"], out[f"hi__{m}"] = lo[m], hi[m]
    return out


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    exp_dir = Path(config_path).parent
    root = exp_dir.parents[1]
    ds = load_nhv(root / cfg["data"])
    if cfg["taxon_set"] == "primary":
        keep = primary_taxon_mask(ds.counts, cfg["primary_weeks"], cfg["primary_min_prevalence"])
    else:
        keep = np.ones(ds.N, dtype=bool)
    counts = ds.counts[:, keep]
    print(f"taxon_set={cfg['taxon_set']}: {keep.sum()} taxa, T={ds.T}", flush=True)
    rng = np.random.default_rng(cfg["seed"])
    out_dir = exp_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    for h in cfg["horizons"]:
        res = run_horizon(ds, counts, cfg, h, rng)
        res["taxon_ids"] = ds.taxon_ids[keep]
        res["dates"] = ds.dates.strftime("%Y-%m-%d").to_numpy()
        np.savez_compressed(out_dir / f"{cfg['taxon_set']}_h{h}.npz", **res)
        print(f"saved h={h}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
