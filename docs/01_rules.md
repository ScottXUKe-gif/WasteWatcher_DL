# Project rules

Each rule is a direct response to a failure mode documented in
`00_lessons_from_four_repos.md`.

1. **Baselines first.** The first two rows of every results table are persistence and
   rolling mean. A model with no baseline next to it has no result.
2. **Preregister, then run.** Every experiment has a decision record in `docs/decisions/`
   with the pass/fail criterion written and committed *before* results exist. A failed gate
   is a failure. Gates are not moved, deferred, or redefined after the fact. If the gate
   turns out to be wrong, write a new ADR that says so and why, and rerun.
3. **Every number points to a file.** A claim in any document must cite a path under
   `experiments/<name>/tables/` or `predictions/`. Result tables and figures are committed
   to git; predictions (`.npz`) live on gibbs, never on scratch.
4. **Real data first.** Experiments run on `data/raw/nhv_filtered.h5ad`. Synthetic data is
   used only to validate that a mechanism-learning component can recover known ground truth
   before it is claimed on real data.
5. **No leakage, tested.** Anything fit on data (pseudocount, scaling, taxon selection,
   hyperparameters) is fit on the training window of each fold only. `tests/` contains a
   test that corrupts the future and checks predictions do not change.
6. **Every experiment ends in a notebook** under `notebooks/` that shows, at minimum:
   (a) observed trajectories with each model's forecast overlaid for a fixed panel of taxa,
   (b) the distribution across taxa and folds of the error metric, not only its median,
   (c) paired fold-wise comparison against the baselines.
7. **Small files, one job each.** Source files stay under 200 lines. Scripts run one
   experiment from one config and write to one experiment folder.
8. **Commit as you go.** Uncommitted work older than a day is treated as lost.
