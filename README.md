# WasteWatcher_DL

Predictability of wastewater microbiome time series — restarted from baselines.

This repository consolidates four earlier attempts (`Microbiome_Time_Series`,
`Wastewatcher`, `microbe-repr`, `microbe-repr-dynamic-range-v1`) and starts over with one
question: **is there predictable signal beyond mean reversion in the New Haven 42-week
data, and if so where?** No deep model is built until that question has a quantified answer.

- What was learned from the previous repos: `docs/00_lessons_from_four_repos.md`
- Working rules: `docs/01_rules.md`
- Decision records (preregistered before results): `docs/decisions/`

## Status

| Experiment | ADR | State |
|---|---|---|
| `baseline_v1` | 001 | preregistered, not yet run |

## Layout

```
data/raw/            nhv_filtered.h5ad (20,054 taxa × 42 weeks) + covariate CSV
src/wwdl/            data, transform, splits, subsets, baselines, metrics, intervals
scripts/             run_baselines.py (config → tables + predictions), make_figures.py
experiments/<name>/  config.yaml, tables/, predictions/, figures/, decision_readout.md
notebooks/           one notebook per experiment: trajectories + error distributions
tests/               includes a leakage test
```

## Setup

```
pip install -e .
pytest
python scripts/run_baselines.py experiments/baseline_v1/config.yaml
```
