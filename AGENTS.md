# Binderopt agent guide

Binderopt is a reproducible Mosaic-based protein-binder design package.
Keep code and documentation focused on the supported pipeline.

## Code design

- `configs/` owns targets, model recipes, optimizer parameters, experiments, and scientific
  constants. Do not duplicate these values in Python.
- Top-level command modules own `main()` and command orchestration. Keep reusable logic
  in subpackages and use `python -m binderopt.<command>` entry points.
- `src/binderopt/optimization/` owns the ensemble, objective, Biohub/PGD optimizers, seeds,
  batch execution, and reference predictions. Preserve batch-lane identity and separate
  ordinary and tail losses.
- `src/binderopt/ranking/` owns SolubleMPNN, judge inference, DockQ, pooled screening, and
  optional cohort composites. Run redesign and judges in isolated spawned processes.
- `src/binderopt/structure/` owns target entities, masks, and backend-specific features.
- `src/binderopt/assets/` owns model and MSA caches. Model weights do not belong in the repository.
- `src/binderopt/outputs.py` owns atomic final artifacts and resolved scientific provenance.
  Do not add checkpoints or generic schemas without a demonstrated need.

Prefer small functions and explicit backend paths over registries or plugin systems.
Keep package initializers lightweight and heavy JAX/Mosaic imports inside execution paths
so caches can be configured before importing models. Command modules assemble the pipeline;
subpackages implement its stages.

## Scientific invariants

- Design one protein binder chain against all target protein chains combined. Retain ligands
  in structures but exclude them from protein losses, iPTM, and `IPSAE_min`.
- Preserve the design objective, no-Cys constraint, stochastic two-model ensemble, per-lane
  iPTM tail selection, selected step, seed derivation, and generation order.
- Predict both final design-model references before publishing a cell. Interrupted design
  cells are recomputed; completed cells must not be silently overwritten.
- For each judge, choose the iPsae-winning seed and take DockQ from that same seed. Record
  DockQ's geometric receptor/ligand assignment and target-aligned binder backbone RMSD.
  Save each judge's winning structure. `ranking_score` and `ipsae_ranking_score` are the
  mean best-over-seed iPsae across judges; `rank_loss == -ranking_score`.
- Reviewed ranking recipes define the supported model sets and scientific validity.
  Publication runs use the reference design profile and full ranking recipe.
- Outputs are atomic and final-only. Preserve output identity and resolved configuration;
  do not combine cells from incompatible experiments or configurations.

Ask before changing the objective, ranking formula, target semantics, supported model set,
or output identity. Add regression coverage when changing an invariant. Do not test exact
experiment names, counts, or sweep layouts: those are recipes, not software contracts.

## Analysis

Match paired arms on their recorded `design_key`. A shared base seed pairs initialization
and per-step stochastic draws only when `init_scale` also matches. Different base seeds
provide independent designs. Optional target-cohort composites use explicit analysis weights
and remain separate from primary cell rankings and `result.json` files.

## Development

Use Python 3.12 and `uv`, installing exactly one of the `jax-cpu` and `jax-cuda` groups.
Run the smallest relevant tests and Ruff after code changes. Run the full non-hardware suite
for cross-cutting changes, scientific invariants, config composition, outputs, orchestration,
and before an important commit or release:

```sh
uv run ruff check .
JAX_PLATFORMS=cpu uv run pytest -m "not hardware"
```

Documentation-only edits do not require tests. Hardware and integration tests need matching
assets and explicit relevance; do not start downloads or model runs implicitly. Preview
resolved Hydra configurations before running an experiment.
