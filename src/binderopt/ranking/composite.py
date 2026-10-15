"""Post-hoc target-cohort composite ranking."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from binderopt.outputs import utc_now
from binderopt.ranking.cells import design_cell_result, discover_design_cells


def _normalize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("composite ranking requires finite measurements")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=0))
    z_scores = (
        np.zeros_like(array) if standard_deviation == 0 else (array - mean) / standard_deviation
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "count": len(array),
        "zero_variance": standard_deviation == 0,
    }, z_scores


def _resolve_experiment_dirs(experiment_dirs: Path | Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(experiment_dirs, (str, Path)):
        resolved = (Path(experiment_dirs).resolve(),)
    else:
        resolved = tuple(Path(path).resolve() for path in experiment_dirs)
    if not resolved:
        raise ValueError("composite ranking requires at least one experiment directory")
    if len(set(resolved)) != len(resolved):
        raise ValueError("composite ranking input directories must be unique")
    return resolved


def compute_composite_ranking(
    experiment_dirs: Path | Sequence[Path],
    *,
    ipsae_weight,
    dockq_weight,
    exclude_targets=(),
    optimizers=None,
):
    """Pool compatible ranked cells by target across one or more experiments."""
    experiment_dirs = _resolve_experiment_dirs(experiment_dirs)
    ipsae_weight = float(ipsae_weight)
    dockq_weight = float(dockq_weight)
    excluded = {str(target) for target in exclude_targets}
    included_optimizers = None if optimizers is None else {str(item) for item in optimizers}
    if ipsae_weight <= 0 or dockq_weight <= 0:
        raise ValueError("composite weights must be positive")

    cohorts = defaultdict(list)
    signatures = {}
    experiment_ids = set()
    for experiment_dir, cell_dir in (
        (source_dir, cell_dir)
        for source_dir in experiment_dirs
        for cell_dir in discover_design_cells(source_dir, allow_mixed_experiments=True)
    ):
        result = design_cell_result(cell_dir)
        target_id = str(result["target_id"])
        if target_id in excluded:
            continue
        optimizer_config = result.get("config", {}).get("optimizer", {})
        optimizer = str(optimizer_config.get("optimizer_id") or result["optimizer_kind"])
        if included_optimizers is not None and optimizer not in included_optimizers:
            continue
        if result.get("ranking_model_id") is None:
            raise ValueError(f"composite ranking requires a ranked cell: {cell_dir}")
        experiment_id = str(result["experiment_id"])
        experiment_ids.add(experiment_id)
        signature = json.dumps(
            {
                "target": result.get("config", {}).get("target"),
                "binder_length": result.get("binder_length"),
                "objective_id": result.get("objective_id"),
                "model_id": result.get("model_id"),
                "ranking_model_id": result.get("ranking_model_id"),
            },
            sort_keys=True,
        )
        previous = signatures.setdefault(target_id, signature)
        if signature != previous:
            raise ValueError(f"composite cohort mixes incompatible cells: {target_id}")
        for design in result["designs"]:
            try:
                models = design["ranking"]["models"]
                model_names = tuple(sorted(models))
                ipsae = {name: float(models[name]["best_ipsae_min"]) for name in model_names}
                dockq = {name: float(models[name]["winning_dockq"]) for name in model_names}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"ranked design has invalid measurements: {cell_dir}") from error
            if not model_names or not np.all(np.isfinite([*ipsae.values(), *dockq.values()])):
                raise ValueError(f"ranked design has invalid measurements: {cell_dir}")
            primary_score = sum(ipsae.values()) / len(ipsae)
            if not np.isclose(float(design["ranking_score"]), primary_score):
                raise ValueError(f"primary ranking is not the iPsae endpoint: {cell_dir}")
            cohorts[target_id].append(
                {
                    "experiment_id": experiment_id,
                    "input": str(experiment_dir),
                    "cell": str(cell_dir.relative_to(experiment_dir)),
                    "optimizer_kind": str(result["optimizer_kind"]),
                    "design_index": int(design["design_index"]),
                    "design_key": design.get("design_key"),
                    "sequence": str(design["sequence"]),
                    "ranking_score": primary_score,
                    "model_names": model_names,
                    "ipsae": ipsae,
                    "dockq": dockq,
                }
            )

    targets = {}
    for target_id, designs in sorted(cohorts.items()):
        model_names = designs[0]["model_names"]
        if any(design["model_names"] != model_names for design in designs):
            raise ValueError(f"composite cohort mixes ranking model sets: {target_id}")
        statistics = {}
        z_scores = {}
        for metric in ("ipsae", "dockq"):
            for model_name in model_names:
                term = f"{model_name}.{metric}"
                statistics[term], z_scores[term] = _normalize(
                    [design[metric][model_name] for design in designs]
                )

        denominator = len(model_names) * (ipsae_weight + dockq_weight)
        ranked = []
        for index, design in enumerate(designs):
            ipsae_z = {name: float(z_scores[f"{name}.ipsae"][index]) for name in model_names}
            dockq_z = {name: float(z_scores[f"{name}.dockq"][index]) for name in model_names}
            composite_score = (
                ipsae_weight * sum(ipsae_z.values()) + dockq_weight * sum(dockq_z.values())
            ) / denominator
            ranked.append(
                {key: value for key, value in design.items() if key != "model_names"}
                | {
                    "z_scores": {"ipsae": ipsae_z, "dockq": dockq_z},
                    "composite_score": composite_score,
                }
            )
        ranked.sort(
            key=lambda item: (
                -item["composite_score"],
                item["experiment_id"],
                item["cell"],
                item["design_index"],
            )
        )
        for rank, design in enumerate(ranked, start=1):
            design["composite_rank"] = rank
        targets[target_id] = {
            "statistics": statistics,
            "designs": ranked,
        }

    sorted_experiment_ids = sorted(experiment_ids)
    return {
        "experiment_id": sorted_experiment_ids[0] if len(sorted_experiment_ids) == 1 else None,
        "experiment_ids": sorted_experiment_ids,
        "inputs": [str(path) for path in experiment_dirs],
        "created_at_utc": utc_now(),
        "weights": {"ipsae": ipsae_weight, "dockq": dockq_weight},
        "targets": targets,
    }
