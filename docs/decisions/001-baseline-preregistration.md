# ADR-001: Baseline experiment `baseline_v1` — preregistration

Status: accepted, preregistered 2026-11-18, before any result exists.

## Question

On the New Haven 42-week wastewater data, is there predictable signal beyond mean
reversion? If yes, in which taxa, at which horizons, and how much?

## Why this comes before any model

Across four prior repos no learned model beat persistence or the mean under a valid
protocol, and on real data the mean baseline was never run at all
(`docs/00_lessons_from_four_repos.md` §2). Any deep model built now would have no ceiling
to be measured against.

## Data protocol

- Input: `data/raw/nhv_filtered.h5ad`, `counts` layer, transposed to (T=42, N=20054).
- Per fold, using the training window only: adaptive pseudocount, CLR, per-taxon z-score.
  Targets are evaluated in CLR space (primary) and composition space (secondary).
- **Primary taxon set**: taxa with prevalence ≥ 50% in the first 20 weeks (fixed for all
  folds, defined with no future information). Secondary: all 20,054 taxa.
- **Subsets** (all defined on each fold's training window): `all`, `top500_variance`,
  `top500_spike`, `active` (non-dormant: train prevalence ≥ 80%).
- **Splits**: expanding window, `min_train = 20`, one origin per week, evaluated
  separately for horizons h ∈ {1, 2, 4, 6}. Each horizon h has origins 20 … 42−h
  (22 folds at h=1, 17 at h=6). Direct multi-step: each h is predicted from the origin,
  not recursively.

## Models (all linear/statistical; no neural network in this experiment)

| Group | Model |
|---|---|
| Trivial | `persistence`; `train_mean`; `rolling_mean_4`, `rolling_mean_8`; `ewma` (α chosen on train) |
| Mean reversion | `mean_revert_shared`: x̂ = μ + φ^h (x_t − μ), one φ for all taxa; `mean_revert_taxon`: per-taxon φ with shrinkage toward the shared φ |
| Autoregression | `ridge_ar_p` for p ∈ {1, 2, 3}, per taxon, direct for each h, λ chosen on train by inner rolling CV |
| Low rank | `pca_var_k` for k ∈ {2, 4, 8}: PCA on train CLR, ridge VAR on factors, decode; `pls_var` |
| Covariates | `ridge_ar1_env_lag`: ridge AR(1) + lagged covariates (legal); `ridge_ar1_env_oracle`: same with covariates at the target time (an upper bound, labeled as such, never a candidate) |
| Control | `shuffle_control`: best non-trivial model refit on a time-shuffled training window; skill must be ≈ 0 |

## Metrics (per horizon, per subset)

- Primary: CLR RMSE; **skill** = 1 − MSE_model / MSE_ref for ref ∈ {persistence, rolling_mean_4}.
- Paired fold-wise win rate vs each reference (fraction of folds where model MSE < ref MSE).
- Per-taxon R² relative to rolling_mean_4, reported as a distribution.
- Secondary: Aitchison distance and Bray–Curtis in composition space; energy ratio
  (predicted motion / observed motion) so the flatness axis stays visible.
- Uncertainty: for the best non-trivial model, historical-innovation resampling intervals;
  report 50/80/90% coverage.

## Decision rule (fixed now)

**Signal present** if, at h = 1 on the primary taxon set, some model other than
persistence / train_mean / rolling means / ewma has
(a) skill vs `rolling_mean_4` > 0.05,
(b) wins on ≥ 70% of folds against `rolling_mean_4`, and
(c) `shuffle_control` skill in (−0.05, 0.05).
Then report the subset and the largest h at which (a) and (b) still hold.

**No signal** otherwise. The conclusion is then "NHV single-site 42-week data is not
predictable beyond mean reversion with the available covariates", and the next ADR must
choose between (i) a characterization/negative-result write-up and (ii) different data
(multi-site or long longitudinal series). Changing the model architecture is **not** an
allowed response to this outcome.

Neither outcome triggers a deep model. A deep model requires a separate ADR that cites
the ceiling found here.

## Required figures (in `notebooks/01_baseline_v1.ipynb`)

1. Skill vs horizon, one line per model, one panel per subset.
2. Paired fold-wise MSE differences vs `rolling_mean_4`, one point per fold.
3. Per-taxon R² histograms per model at h=1 and h=4.
4. A fixed panel of 20 taxa (10 top-variance, 10 random active): observed CLR trajectory
   with all model forecasts from 3 origins overlaid, plus the 90% innovation interval.
5. Effect of covariates: `ridge_ar1` vs `ridge_ar1_env_lag` vs `ridge_ar1_env_oracle`.

## Outputs

`experiments/baseline_v1/{config.yaml, tables/*.csv, predictions/*.npz, figures/*.png, decision_readout.md}`.
The readout answers the decision rule line by line with the file each number comes from.
