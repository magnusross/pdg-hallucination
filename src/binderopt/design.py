"""Design a complete binder cell and publish its final artifacts."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from binderopt.assets.cache import configure_process_caches
from binderopt.optimization.cell import (
    make_batch_runner,
    predict_design_references,
    run_design_cell,
)
from binderopt.outputs import git_state, utc_now, write_design_outputs
from binderopt.structure.targets import target_from_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("binderopt.design")


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="design")
def main(cfg: DictConfig) -> None:
    process_started = time.perf_counter()
    started_at_utc = utc_now()
    resolved_container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_container, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    resolved = cast(dict[str, Any], resolved_container)
    cache = resolved["profile"]["cache"]
    configure_process_caches(Path(str(cache["root"])), offline=bool(cache["offline"]))

    import jax

    from binderopt.optimization.models import load_design_ensemble
    from binderopt.optimization.seeding import design_keys, key_words
    from binderopt.structure.features import build_esm_binder_features

    target = target_from_config(resolved["target"])
    out_dir = Path(str(resolved["cell_dir"])).resolve()
    profile_design = resolved["profile"]["design"]
    model_ids = [item["checkpoint"] for item in profile_design["structure_models"]]
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    log.info(
        "git=%s devices=%s jax_compilation_cache=%s",
        git_state(REPO_ROOT),
        jax.devices(),
        getattr(jax.config, "jax_compilation_cache_dir", None)
        or "disabled (set JAX_COMPILATION_CACHE_DIR)",
    )
    log.info(
        "target=%s chains=%d ligands=%s models=%s cache=%s offline=%s output=%s",
        target.target_id,
        len(target.chains),
        [(item.ccd, item.count) for item in target.ligands],
        model_ids,
        cache["root"],
        cache["offline"],
        out_dir,
    )

    ensemble = load_design_ensemble(profile_design)
    prepared = build_esm_binder_features(
        ensemble.features_model,
        int(resolved["binder_length"]),
        target,
        supports_msa=bool(profile_design["supports_msa"]),
    )
    identity = {
        "experiment_id": resolved["experiment_id"],
        "target_id": target.target_id,
        "objective_id": resolved["objective_id"],
        "model_id": "+".join(model_ids),
        "ranking_model_id": None,
        "optimizer_kind": resolved["optimizer"]["kind"],
        "binder_length": resolved["binder_length"],
        "num_designs_requested": resolved["num_designs"],
        "base_seed": resolved["seed"],
        "batch_size": resolved["batch_size"],
        "scientific_validity": resolved["profile"]["scientific_validity"],
    }
    runner = make_batch_runner(resolved, ensemble, prepared)
    keys = design_keys(int(resolved["seed"]), int(resolved["num_designs"]))
    words = key_words(keys)
    started = time.perf_counter()
    setup_wall_time_s = started - process_started
    completed = run_design_cell(
        keys=keys,
        batch_size=int(resolved["batch_size"]),
        run_batch=runner,
    )
    optimization_wall_time_s = time.perf_counter() - started
    designs: list[dict[str, Any]] = []
    pssms: list[NDArray[np.float32]] = []
    trajectory: list[dict[str, Any]] = []
    for design_index, batch, lane, item in completed:
        designs.append(
            {
                "design_index": design_index,
                "optimizer_sequence": item.sequence,
                "argmax_loss": item.argmax_loss,
                "tail_ranking_loss": item.tail_ranking_loss,
                "tail_iptm": -item.tail_ranking_loss,
                "design_objective": item.design_objective,
                "optimizer_esmc_log_perplexity": item.esmc_log_perplexity,
                "optimizer_esmc_perplexity": float(np.exp(item.esmc_log_perplexity)),
                "selected_step": item.selected_step,
                # Identical across arms under one base seed: the pairing join key.
                "design_key": [int(word) for word in words[design_index]],
                "batch": batch,
                "batch_lane": lane,
                "design_time_s": item.design_time_s,
                "metrics": item.metrics,
            }
        )
        pssms.append(item.pssm)
        trajectory.extend(point | {"design_index": design_index} for point in item.trajectory)

    reference_started = time.perf_counter()
    references = predict_design_references(
        ensemble=ensemble,
        prepared=prepared,
        pssms=pssms,
        design_keys_array=keys,
        model_specs=profile_design["structure_models"],
        inference=profile_design["inference"],
    )
    reference_prediction_wall_time_s = time.perf_counter() - reference_started

    write_design_outputs(
        out_dir,
        identity=identity,
        config=resolved,
        git_sha=git_state(REPO_ROOT),
        hydra_overrides=list(HydraConfig.get().overrides.task),
        designs=designs,
        pssms=pssms,
        trajectory=trajectory,
        references=references,
        process_started=process_started,
        started_at_utc=started_at_utc,
        setup_wall_time_s=setup_wall_time_s,
        optimization_wall_time_s=optimization_wall_time_s,
        reference_prediction_wall_time_s=reference_prediction_wall_time_s,
        gpu_count=int(resolved["profile"]["resources"]["gpus_per_task"]),
    )


if __name__ == "__main__":
    main()
