# Lessons from four prior repos (audit of 2026-11-18)

This document consolidates what was actually learned across four earlier attempts at
deep-learning time-series prediction on wastewater microbiome data. It separates claims
that are backed by result files from claims that only appear in prose. Paths are given so
every statement can be re-checked.

Repos audited (all under `/gpfs/gibbs/pi/gerstein/kx44/project/`):

| Period | Repo | Intent | Outcome |
|---|---|---|---|
| 2025-05 → 2026-03 | `Microbiome_Time_Series` | Forecasting (ARIMA, Graph-Mamba) → fell back to clustering (SplineMixtureEM) | EM loses to DTW on real data and to raw GMM on toy; README numbers have no matching result files |
| 2026-03 → 2026-08 | `Wastewatcher` (BioFlow) | Conditional generative forecaster + learned delay-interaction graph | All May "30× better than persistence" claims invalid (target leakage); delay graph r≈0 on synthetic data; the one valid result (1.616 vs 2.221 MSE) never had a mean baseline |
| 2026-05 → 2026-07 | `microbe-repr` | Unify the two above | 22 ADRs in five days; failed gates were moved instead of stopping; full reset on 2026-06-18 to a toy-only "factor model" scaffold; never returned to real data |
| 2026-08 | `microbe-repr-dynamic-range-v1` | Diagnose "why forecasts flatten" | 7 SDLDA versions all failed preregistered gates; v7 results purged from scratch, now prose-only; no real data at all |

## 1. The data (the one fact that constrains everything)

There is exactly **one real dataset** across all four repos:

- New Haven wastewater metagenomics, **one site**, **42 weekly time points** (2022-09-07 → 2023-06-28), regular weekly sampling.
- Canonical file: `Microbiome_Time_Series/Data/Processed/nhv_filtered.h5ad` → **20,054 taxa × 42**, layers `counts/clr/log1p/z_score`, obs = taxa with taxonomy, var = timepoints with 5 covariates (COVID-19 load, Tmin/Tmax/Tavg, precipitation). Copied to `data/raw/` in this repo.
  - The `Wastewatcher` repo used a 73,301-taxon version; `microbe-repr` ADR-009 declared the 20k version canonical.
- Raw matrix is 86% zeros; across-taxa Fano ≈ 2.25, Spearman(mean, var) = 0.97 (`Microbiome_Time_Series/Notebooks/0.1_preprocessing.ipynb`).
- Toy datasets: `simulated_microbiome_5000x42.h5ad` (5 known pattern classes) and `microbe-repr/experiments/toy_demo/toy_data.npz` (T=80, 60 taxa, 4 modules). **Every experiment after 2026-06-18 in two repos ran only on the 80×60 toy.**

T = 42, N = 20k, one subject. This is not a deep-learning forecasting dataset. The question
"can DL predict this" was never answered because the prerequisite question "is there any
predictable signal beyond mean reversion on the real data" was never asked properly.

## 2. Hard facts reproduced independently across repos

These are backed by result files and in several cases found by different methods in
different repos. They are the real output of ~18 months of work.

1. **Series are strongly mean-reverting with little autoregressive structure.**
   - Community-level lag-1 ACF = 0.069 (`Wastewatcher/notebooks/01_EDA_and_CLR.ipynb`).
   - Residual AR(1) median autocorrelation −0.046 (`microbe-repr/docs/research_log.md` §18).
   - Under CV, the recent mean beats persistence by 33%: last_mean 1.256 vs persistence 1.865 vs ARIMA(0,1,1) 1.266 (`Wastewatcher/experiments/20260512_baseline_comparison/results/comparison_table.txt`).
2. **No learned model beats persistence or the mean on point error under any valid protocol**, real or toy.
   - Toy strict rolling, h=4, composition RMSE: persistence 0.00783; best neural (Transformer+motion, 3 seeds) 0.00878; best overall is linear PLS ridge VAR 0.00737 (`microbe-repr/experiments/toy_demo/strict_factor_baselines/summary.csv`).
   - Toy CLR RMSE: last value 0.174 beats every ridge/MLP/Transformer variant (0.19–0.45) (`microbe-repr-dynamic-range-v1/experiments/dynamic_range_diagnostics_v1/tables/stage0_*.csv`).
   - The only "win" (RDL+gate α=0.70) chose α on the test fold; honest train-only selection loses again (microbe-repr ADR-024).
   - Real data: neural location MSE 1.616 vs rolling persistence 2.221 at t→t+1 (`Wastewatcher/experiments/20260512_ablation_no_env/results/rolling_innovation_calibration.json`) — but **no rolling mean or AR baseline was ever run on real data**, and the model is "persistence anchor + residual", so this may be pure shrinkage toward the mean.
3. **Low RMSE means a flat forecast.** Neural models trade RMSE for amplitude; energy ratio of good-RMSE models is 0–0.07, while models that predict motion over-amplify (ratio 2.4–2.6) and lose on RMSE (`dynamic_range_diagnostics_v1`, `direct_multihorizon_dynamic_v1`).
4. **Per-edge delays / interaction graphs are not identifiable from forecast loss.** Found three times independently: Wastewatcher nb14/nb16 (r(τ)≈0 across λ sweep, T=200, noise ladder, 7 architectures); microbe-repr ADR-016/018/020 (6 architectures at edge AUROC ≈ 0.5; τ learnable only with direct supervision); dynamic-range-v1 v7 (PCA-VARX recovers the simulator, SDLDA does not). Caveat: the Wastewatcher oracle-τ OLS check also failed (r(A) = −0.10), which suggests a generator/model convention mismatch and has not been resolved.
5. **Spline/EM representations carry no forecast signal** (microbe-repr ADR-005: θ predicts forecast error with R² ≈ 0; only taxonomy does, R² = 0.14) and are not the best clustering either (DTW silhouette 0.368 vs EM −0.043 on real data; raw GMM ARI 0.996 vs EM 0.860 on toy; `Microbiome_Time_Series/outputs/evaluations/`).
6. **Natural cluster count on real data is K=2 (dormant/active), not 7** (microbe-repr ADR-004: silhouette positive only at K=2–3; HDBSCAN finds 0 clusters).
7. **Learned uncertainty is overconfident; empirical innovation resampling is calibrated.** 90% coverage: Gaussian residual flow 0.670, Student-t 0.41–0.44, autoregressive Student-t 0.712; historical-innovation resampling 0.893 (`Wastewatcher/experiments/20260821_*`, `20260512_ablation_no_env`).
8. **Environment covariates looked useless or harmful** in every experiment — but only under the leaky CV protocol. Status: **unresolved**.

## 3. Why the projects diverged (the failure pattern)

The same pattern appears in all four repos:

- **Simple baselines were never the opponent at the start.** Each repo built an architecture first and added persistence months later; a rolling-mean baseline on real data still does not exist anywhere.
- **Gates were moved when they failed** (microbe-repr ADR-010/012/013: "forecast gate deferred to Phase B") or the question was changed without a record (the 2026-06-18 reset has no ADR).
- **Documentation ran ahead of results.** README numbers with no backing file (MTS); a report written before leakage was discovered (WW); v7 conclusions that evaporated with scratch cleanup (DR-v1); ADR-025 is a 0-byte file; July 2026 work was never committed; ADRs 001–022 were deleted from git.
- **Retreat to toy data was permanent.** After June 2026 no real-data experiment was run in either repr repo; "transfer to NHV" was always "Priority 4".
- **Architecture-first.** Spline, flow matching, graph attention, delay attention, VAE, JEPA/REPA, factor models, Transformer — each added a mechanism before establishing there was signal for it to learn.

## 4. What is reusable (ported into `src/wwdl/`, by copying functions, not importing)

| Component | Source | Status |
|---|---|---|
| Train-only pseudocount CLR / inverse CLR | `microbe-repr/src/microbe_repr/preprocessing.py` | ported |
| Expanding-window rolling origins with leakage check | `microbe-repr-dynamic-range-v1/src/microbe_repr/dynamic_range/splits.py` | ported |
| Train-defined taxon subsets (variance, spike, prevalence) | `.../dynamic_range/subsets.py` | ported |
| Persistence / mean / MA / trend / ridge baselines | `.../dynamic_range/baselines.py`, `microbe-repr/scripts/strict_rolling_factor_baselines.py` | ported and extended |
| Historical-innovation interval + coverage | `Wastewatcher/scripts/evaluate_innovation_calibration.py` | ported |
| Energy / range ratio metrics | `.../dynamic_range/forecast_metrics.py` | ported |
| GloBI + taxonomy prior graph | `Wastewatcher/bioflow/graph_builder.py`, `data/processed/prior_graph*.pt` | not needed yet |
| Toy simulators | `Microbiome_Time_Series/wastewatcher/data_generator.py`, `microbe-repr/src/microbe_repr/simulation.py` | not needed yet |

## 5. What this repo does differently

See `docs/01_rules.md`. In one line: baselines first, preregistered gates that are not
moved, every number traceable to a committed file, real data always, figures with
trajectories and error distributions for every experiment.
