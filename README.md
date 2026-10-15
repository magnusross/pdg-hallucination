# binderopt

Protein-binder design built on [Mosaic](https://github.com/escalante-bio/mosaic).
Binderopt optimizes a binder sequence with Biohub or projected gradient descent,
redesigns it with SolubleMPNN, and ranks predictions from ESMFold2, ESMFold2Fast,
and ProtenixV2. All target protein chains are scored together; ligands are excluded
from protein interface metrics.

## Setup

Requires Linux, Python 3.12, and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen --group jax-cuda
source .venv/bin/activate
export MOSAIC_CACHE_DIR=/path/to/mosaic-cache
export JAX_COMPILATION_CACHE_DIR=/path/to/jax-cache
```

For configuration work and tests, install `--group jax-cpu` instead.
Model weights are not included. Stage the checkpoints listed in
`configs/profile/reference.yaml` and `configs/ranking/full.yaml`, plus Protenix
supporting assets, under the cache's `huggingface/` and `protenix/` directories.
Follow each upstream project's license and access terms.

Check assets and prepare target MSAs:

```sh
python -m binderopt.cache check profile=reference ranking=full
python -m binderopt.msa prefetch target=4ZQK
python -m binderopt.msa check target=4ZQK
```

## Design and rank

Run from the repository root, using matching target, optimizer, and experiment settings:

```sh
python -m binderopt.design target=4ZQK optimizer=pgd profile=reference
python -m binderopt.rank target=4ZQK optimizer=pgd profile=reference ranking=full
```

Use `optimizer=biohub` for logit gradient descent. Targets, model recipes, optimizer
settings, and experiments live in `configs/`. Preview a resolved configuration with
`--cfg job --resolve`. The reference profile and full ranking recipe are the
publication settings; other recipes record their scientific validity in the config.

Results are written to the configured `cell_dir` under `outputs/`: `result.json`
contains scores and resolved settings, alongside FASTA sequences, structures, and
optimization trajectories. Ranking uses the mean of each judge's best-over-seed
iPsae; DockQ is taken from the same winning seeds. Completed design cells are
protected from overwriting, and interrupted designs restart the cell.

## Reproducing the analysis


All experimental data are available at [doi:10.5522/04/34493751](https://doi.org/10.5522/04/34493751)
as a `tar.zst` archive of the paper dataset. It includes sequences, predicted structures,
rankings, trajectories, settings, timings, and analysis tables and figures in `results/`.
Extract the archive with `tar --zstd -xf /path/to/archive.tar.zst`, then set `PAPER_DATA`
to the extracted folder containing `main_experiment/` and `pgd_75_screened/`.

`analysis/optimizer_analysis.py` compares paired optimizer runs;
`analysis/structural_uniqueness.py` measures structural diversity, and
`analysis/pgd_allocation.py` compares short screened runs with longer runs.

Run from the repository root (CPU analysis does not require model downloads):

```sh
uv sync --frozen --group jax-cpu --group analysis
export PAPER_DATA=/path/to/extracted/paper
uv run python -m analysis.optimizer_analysis "$PAPER_DATA/main_experiment" \
  --references biohub --challengers pgd --ipsae-weight 4 --dockq-weight 1 \
  --output analysis/artifacts/paper_main
uv run python -m analysis.pgd_allocation \
  --control-roots "$PAPER_DATA/main_experiment" \
  --short-root "$PAPER_DATA/pgd_75_screened" \
  --source-root "$PAPER_DATA/pgd_75_source" \
  --output-dir analysis/artifacts/paper_pgd_allocation
```

Use `--help` for further
options; structural uniqueness also requires Foldseek. 
