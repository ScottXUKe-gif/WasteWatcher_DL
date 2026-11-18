"""Turn saved predictions into tables. Never recomputes a model.

Usage: python scripts/summarize.py experiments/baseline_v1/config.yaml
Writes tables/<set>_fold_metrics.csv, <set>_summary.csv, <set>_coverage.csv,
<set>_per_taxon_r2_h<h>.csv and <set>_decision.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wwdl.intervals import coverage
from wwdl.metrics import fold_metrics, per_taxon_r2, skill, win_rate


def load(exp_dir: Path, taxon_set: str, h: int) -> dict:
    return dict(np.load(exp_dir / "predictions" / f"{taxon_set}_h{h}.npz", allow_pickle=False))


def fold_table(d: dict, h: int) -> pd.DataFrame:
    rows = []
    for mi, model in enumerate(d["models"]):
        for fi, origin in enumerate(d["origins"]):
            for si, subset in enumerate(d["subsets"]):
                m = fold_metrics(d["truth"][fi], d["pred"][mi, fi], d["last"][fi], d["masks"][fi, si])
                rows.append(dict(h=h, model=str(model), origin=int(origin), subset=str(subset), **m))
    return pd.DataFrame(rows)


def summary_table(folds: pd.DataFrame, refs: list[str]) -> pd.DataFrame:
    rows = []
    for (h, subset), g in folds.groupby(["h", "subset"]):
        piv = g.pivot(index="origin", columns="model", values="mse")
        for model in piv.columns:
            row = dict(h=h, subset=subset, model=model, rmse=float(np.sqrt(piv[model].mean())),
                       mse_median=float(piv[model].median()),
                       energy_ratio=float(g[g.model == model]["energy_ratio"].median()),
                       aitchison=float(g[g.model == model]["aitchison"].mean()),
                       bray_curtis=float(g[g.model == model]["bray_curtis"].mean()))
            for ref in refs:
                row[f"skill_vs_{ref}"] = skill(piv[model].to_numpy(), piv[ref].to_numpy())
                row[f"win_vs_{ref}"] = win_rate(piv[model].to_numpy(), piv[ref].to_numpy())
            rows.append(row)
    return pd.DataFrame(rows)


def r2_table(d: dict, ref: str) -> pd.DataFrame:
    models = list(d["models"])
    ref_pred = d["pred"][models.index(ref)]
    cols = {m: per_taxon_r2(d["truth"], d["pred"][mi], ref_pred) for mi, m in enumerate(models)}
    return pd.DataFrame(cols, index=d["taxon_ids"])


def coverage_table(d: dict, cfg: dict, h: int) -> pd.DataFrame:
    rows = []
    all_mask = d["masks"][:, 0]
    for m in cfg["interval_models"]:
        for li, level in enumerate(d["levels"]):
            lo, hi = d[f"lo__{m}"][li], d[f"hi__{m}"][li]
            rows.append(dict(h=h, model=m, level=float(level),
                             coverage=coverage(d["truth"], lo, hi, all_mask),
                             mean_width=float((hi - lo)[all_mask].mean())))
    return pd.DataFrame(rows)


def decision_table(summary: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = cfg["decision"]
    g = summary[(summary.h == dc["horizon"]) & (summary.subset == dc["subset"])].set_index("model")
    sk, wr = f"skill_vs_{dc['reference']}", f"win_vs_{dc['reference']}"
    rows = []
    for model in cfg["models"]:
        shuffle_key = f"shuffle__{model}"
        shuffle_skill = g.loc[shuffle_key, sk] if shuffle_key in g.index else np.nan
        passes = (g.loc[model, sk] > dc["min_skill"]) and (g.loc[model, wr] >= dc["min_win_rate"])
        rows.append(dict(model=model, skill=g.loc[model, sk], win_rate=g.loc[model, wr],
                         shuffle_skill=shuffle_skill,
                         shuffle_ok=bool(abs(shuffle_skill) < dc["shuffle_tolerance"]) if not np.isnan(shuffle_skill) else None,
                         passes_a_b=bool(passes)))
    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    exp_dir = Path(config_path).parent
    tables = exp_dir / "tables"
    tables.mkdir(exist_ok=True)
    ts = cfg["taxon_set"]
    folds, covs = [], []
    for h in cfg["horizons"]:
        d = load(exp_dir, ts, h)
        folds.append(fold_table(d, h))
        covs.append(coverage_table(d, cfg, h))
        r2_table(d, cfg["decision"]["reference"]).to_csv(tables / f"{ts}_per_taxon_r2_h{h}.csv")
    folds = pd.concat(folds, ignore_index=True)
    folds.to_csv(tables / f"{ts}_fold_metrics.csv", index=False)
    summary = summary_table(folds, cfg["references"])
    summary.to_csv(tables / f"{ts}_summary.csv", index=False)
    pd.concat(covs, ignore_index=True).to_csv(tables / f"{ts}_coverage.csv", index=False)
    dec = decision_table(summary, cfg)
    dec.to_csv(tables / f"{ts}_decision.csv", index=False)
    print(dec.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
