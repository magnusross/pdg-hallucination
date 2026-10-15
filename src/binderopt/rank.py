"""Redesign and rank one complete design cell with isolated judges."""

from __future__ import annotations

import logging
import multiprocessing
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from binderopt.outputs import atomic_json, utc_now
from binderopt.ranking.cells import (
    aggregate_cell_ranking,
    design_cell_result,
    process_failure,
    requested_seeds,
    select_designs_for_ranking,
)


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="rank",
)
def main(cfg: DictConfig) -> None:
    """Run SolubleMPNN and isolated judges for one completed design cell."""
    from binderopt.assets.cache import configure_process_caches
    from binderopt.ranking.models import run_ranking_model_worker
    from binderopt.ranking.mpnn import run_soluble_mpnn_worker

    started = time.perf_counter()
    started_at_utc = utc_now()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    cell_dir = Path(str(resolved["cell_dir"])).resolve()
    cell_result = design_cell_result(
        cell_dir,
        experiment_id=str(resolved["experiment_id"]),
        target_id=str(resolved["target"]["target_id"]),
        optimizer=str(resolved["optimizer"]["kind"]),
    )
    recipe = resolved["ranking"]
    selected_designs, preselection = select_designs_for_ranking(
        cell_result, resolved.get("ranking_preselection")
    )
    selected_indices = [int(item["design_index"]) for item in selected_designs]
    if preselection is not None:
        atomic_json(cell_dir / "ranking" / "preselection.json", preselection)
    profile = resolved["profile"]
    cache = profile["cache"]
    configure_process_caches(Path(str(cache["root"])), offline=bool(cache["offline"]))
    logger = logging.getLogger("binderopt.rank")
    logger.info("resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(
        "experiment=%s cell=%s candidates=%d selected=%d models=%s",
        cell_result["experiment_id"],
        cell_dir,
        int(cell_result["num_designs"]),
        len(selected_indices),
        [spec["name"] for spec in recipe["models"]],
    )

    base_seed = int(resolved["ranking_seed"])
    context = multiprocessing.get_context("spawn")
    mpnn_process = context.Process(
        target=run_soluble_mpnn_worker,
        kwargs={
            "cell_dir": cell_dir,
            "recipe": dict(recipe),
            "ranking_seed": base_seed,
            "cache_root": Path(str(cache["root"])),
            "offline": bool(cache["offline"]),
            "design_indices": selected_indices,
        },
        name="rank-soluble-mpnn",
    )
    mpnn_process.start()
    mpnn_process.join()
    mpnn_exit_code = mpnn_process.exitcode
    mpnn_process.close()
    if mpnn_exit_code != 0:
        raise process_failure("SolubleMPNN", mpnn_exit_code)

    seeds = requested_seeds(recipe, base_seed)
    for spec in recipe["models"]:
        process = context.Process(
            target=run_ranking_model_worker,
            kwargs={
                "cell_dir": cell_dir,
                "recipe_name": str(recipe["name"]),
                "model_spec": dict(spec),
                "seeds": seeds,
                "base_seed": base_seed,
                "cache_root": Path(str(cache["root"])),
                "offline": bool(cache["offline"]),
                "self_consistency_spec": dict(recipe["self_consistency"]),
                "gpu_count": int(profile["resources"]["gpus_per_task"]),
            },
            name=f"rank-{spec['name']}",
        )
        process.start()
        process.join()
        exit_code = process.exitcode
        process.close()
        if exit_code != 0:
            raise process_failure(f"ranking model {spec['name']}", exit_code)
    aggregate_cell_ranking(
        cell_dir,
        recipe,
        base_seed=base_seed,
        design_indices=selected_indices,
        preselection=preselection,
        ranking_started_at_utc=started_at_utc,
        ranking_wall_time_s=time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
