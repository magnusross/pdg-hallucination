"""Cell-local ranking orchestration."""

from __future__ import annotations

import json
import signal
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from binderopt.outputs import atomic_json, atomic_text, fasta_text, utc_now


def process_failure(label: str, exit_code: int | None) -> RuntimeError:
    if exit_code is not None and exit_code < 0:
        try:
            signal_name = signal.Signals(-exit_code).name
        except ValueError:
            signal_name = f"signal {-exit_code}"
        return RuntimeError(f"{label} terminated by {signal_name} (exit code {exit_code})")
    return RuntimeError(f"{label} failed with exit code {exit_code}")


def requested_seeds(recipe: Mapping[str, Any], base_seed: int) -> list[int]:
    offsets = [int(value) for value in recipe["seed_offsets"]]
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("ranking seed_offsets must be non-empty and unique")
    self_consistency = recipe["self_consistency"]
    if len(offsets) != int(self_consistency["required_seed_count"]):
        raise ValueError("ranking seed count differs from the self-consistency requirement")
    if recipe["scientific_validity"] != self_consistency["validity_level"]:
        raise ValueError("ranking and self-consistency validity levels differ")
    seeds = [base_seed + offset for offset in offsets]
    if min(seeds) < 0:
        raise ValueError("ranking seeds must be non-negative")
    return seeds


def select_designs_for_ranking(
    result: Mapping[str, Any], preselection: Mapping[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Select a deterministic subset while retaining every optimizer candidate."""
    candidates = [dict(item) for item in result.get("design_candidates", result["designs"])]
    indices = [int(item["design_index"]) for item in candidates]
    if indices != list(range(len(candidates))):
        raise ValueError("ranking candidates must be contiguous and generation ordered")
    if preselection is None:
        return candidates, None

    metric = str(preselection["metric"])
    tie_breaker = str(preselection["tie_breaker"])
    expected = int(preselection["expected_candidates"])
    count = int(preselection["count"])
    if metric != "tail_iptm" or tie_breaker != "design_index":
        raise ValueError("ranking preselection must use tail_iptm then design_index")
    if len(candidates) != expected:
        raise ValueError(
            f"ranking preselection expected {expected} candidates, found {len(candidates)}"
        )
    if count <= 0 or count > len(candidates):
        raise ValueError("ranking preselection count must be between one and candidate count")

    scored = []
    for design in candidates:
        score = float(design[metric])
        if not np.isfinite(score):
            raise ValueError("ranking preselection requires finite tail_iptm values")
        scored.append((design, score))
    ordered = sorted(scored, key=lambda item: (-item[1], int(item[0]["design_index"])))
    selected_indices = {int(item[0]["design_index"]) for item in ordered[:count]}
    selected = [item for item in candidates if int(item["design_index"]) in selected_indices]
    manifest = {
        "metric": metric,
        "tie_breaker": tie_breaker,
        "candidate_count": len(candidates),
        "selected_count": count,
        "candidates": [
            {
                "design_index": int(design["design_index"]),
                "score": score,
                "selection_rank": rank,
                "selected": rank <= count,
            }
            for rank, (design, score) in enumerate(ordered, start=1)
        ],
    }
    return selected, manifest


def design_cell_result(
    cell_dir: Path,
    *,
    experiment_id: str | None = None,
    target_id: str | None = None,
    optimizer: str | None = None,
) -> dict[str, Any]:
    """Read one complete design cell and verify its experiment identity."""
    result_path = cell_dir / "result.json"
    if not result_path.is_file():
        raise ValueError(f"ranking design cell does not exist: {cell_dir}")
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    if result["num_designs"] != result["num_designs_requested"]:
        raise ValueError(f"ranking requires a completed design cell: {cell_dir}")
    if experiment_id is not None and result.get("experiment_id") != experiment_id:
        raise ValueError(f"ranking design cell experiment differs: {cell_dir}")
    if target_id is not None and result.get("target_id") != target_id:
        raise ValueError(f"ranking design cell target differs: {cell_dir}")
    if optimizer is not None and result.get("optimizer_kind") != optimizer:
        raise ValueError(f"ranking design cell optimizer differs: {cell_dir}")
    return result


def discover_design_cells(
    experiment_dir: Path, *, allow_mixed_experiments: bool = False
) -> list[Path]:
    """Discover complete cells; analysis may explicitly pool original experiment IDs."""
    if not experiment_dir.is_dir():
        raise ValueError(f"ranking experiment directory does not exist: {experiment_dir}")
    paths = [
        path.parent
        for path in experiment_dir.rglob("result.json")
        if "_launches" not in path.relative_to(experiment_dir).parts
    ]
    if not paths:
        raise ValueError(f"ranking found no completed design cells in {experiment_dir}")
    results = [(cell, design_cell_result(cell)) for cell in sorted(set(paths))]
    experiment_ids = {str(result["experiment_id"]) for _, result in results}
    if not allow_mixed_experiments and len(experiment_ids) != 1:
        raise ValueError(f"ranking input mixes experiments: {sorted(experiment_ids)}")
    return [cell for cell, _ in results]


def _records_by_index(
    path: Path, expected_indices: set[int]
) -> tuple[dict[str, Any], dict[int, Any]]:
    shard = json.loads(path.read_text(encoding="utf-8"))
    records = {int(item["design_index"]): item for item in shard["designs"]}
    if set(records) != expected_indices:
        raise ValueError(f"ranking shard has missing or extra designs: {path}")
    return shard, records


def aggregate_cell_ranking(
    cell_dir: Path,
    recipe: Mapping[str, Any],
    *,
    base_seed: int,
    design_indices: Sequence[int] | None = None,
    preselection: Mapping[str, Any] | None = None,
    ranking_started_at_utc: str | None = None,
    ranking_wall_time_s: float | None = None,
) -> dict[str, Any]:
    """Combine one cell's final shards using the iPsae-only ranking endpoint."""
    model_specs = list(recipe["models"])
    model_names = [str(spec["name"]) for spec in model_specs]
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("ranking model names must be non-empty and unique")
    seeds = requested_seeds(recipe, base_seed)
    self_consistency = recipe["self_consistency"]
    result = design_cell_result(cell_dir)
    candidates = list(result.get("design_candidates", result["designs"]))
    candidate_indices = {int(item["design_index"]) for item in candidates}
    requested_indices = (
        [int(item["design_index"]) for item in candidates]
        if design_indices is None
        else [int(index) for index in design_indices]
    )
    expected_indices = set(requested_indices)
    if not expected_indices or len(expected_indices) != len(requested_indices):
        raise ValueError("ranked design indices must be non-empty and unique")
    if not expected_indices.issubset(candidate_indices):
        raise ValueError("ranked design indices are not present in the design cell")
    if preselection is not None:
        manifest_indices = {
            int(item["design_index"]) for item in preselection["candidates"] if item["selected"]
        }
        if manifest_indices != expected_indices:
            raise ValueError("ranking preselection differs from ranked design indices")
    mpnn_shard, mpnn_records = _records_by_index(
        cell_dir / "ranking" / "soluble_mpnn.json", expected_indices
    )
    expected_mpnn_provenance = {
        "recipe": recipe["name"],
        "model": dict(self_consistency["mpnn"]),
        "ranking_seed": base_seed,
    }
    if mpnn_shard.get("provenance") != expected_mpnn_provenance:
        raise ValueError(f"SolubleMPNN provenance differs: {cell_dir}")

    model_records: dict[str, dict[int, Any]] = {}
    model_timings: dict[str, Any] = {}
    for spec in model_specs:
        name = str(spec["name"])
        path = cell_dir / "ranking" / f"{name}.json"
        shard, records = _records_by_index(path, expected_indices)
        expected_provenance = {
            "recipe": recipe["name"],
            "model": dict(spec),
            "ranking_seed": base_seed,
            "self_consistency": dict(self_consistency),
        }
        if shard.get("model_name") != name or shard.get("requested_seeds") != seeds:
            raise ValueError(f"ranking model identity or seeds differ: {path}")
        if shard.get("provenance") != expected_provenance:
            raise ValueError(f"ranking model provenance differs: {path}")
        model_records[name] = records
        model_timings[name] = shard.get("ranking_model_timing", {})

    ranked: list[dict[str, Any]] = []
    for optimizer_design in candidates:
        design_index = int(optimizer_design["design_index"])
        if design_index not in expected_indices:
            continue
        mpnn = mpnn_records[design_index]
        if mpnn["optimizer_sequence"] != optimizer_design["optimizer_sequence"]:
            raise ValueError("SolubleMPNN optimizer sequence differs from design result")
        models: dict[str, Any] = {}
        structures: dict[str, str] = {}
        ranking_time = 0.0
        for name in model_names:
            record = model_records[name][design_index]
            if record["sequence"] != mpnn["sequence"]:
                raise ValueError(f"ranking model sequence differs: {name}")
            seed_scores = record["seed_scores"]
            if [int(item["seed"]) for item in seed_scores] != seeds:
                raise ValueError(f"ranking model seed results differ: {name}")
            winner = min(
                seed_scores, key=lambda item: (-float(item["ipsae_min"]), int(item["seed"]))
            )
            if int(winner["seed"]) != int(record["winning_seed"]):
                raise ValueError(f"ranking model winning seed is inconsistent: {name}")
            if not all(np.isfinite(float(seed_score["ipsae_min"])) for seed_score in seed_scores):
                raise ValueError(f"ranking model has non-finite iPsae measurements: {name}")
            if not np.isfinite(float(winner["dockq"])):
                raise ValueError(
                    f"ranking model has no finite winning self-consistency measurement: {name}"
                )
            ranking_time += float(record.get("feature_prep_wall_time_s", 0.0))
            ranking_time += sum(float(item["prediction_wall_time_s"]) for item in seed_scores)
            models[name] = {
                "seed_scores": seed_scores,
                "best_ipsae_min": float(winner["ipsae_min"]),
                "winning_dockq": float(winner["dockq"]),
                "winning_seed": int(winner["seed"]),
            }
            if record.get("structure") is not None:
                structures[name] = str(record["structure"])
        ranking_score = sum(item["best_ipsae_min"] for item in models.values()) / len(models)
        ranked.append(
            dict(optimizer_design)
            | {
                "sequence": str(mpnn["sequence"]),
                "mpnn": mpnn,
                "ranking_time_s": ranking_time,
                "ranking": {
                    "recipe": recipe["name"],
                    "base_seed": base_seed,
                    "models": models,
                },
                "structures": structures,
                "structure": structures.get("esmfold2_full") or structures.get("esmfold2_fast"),
                "ipsae_ranking_score": ranking_score,
                "ranking_score": ranking_score,
                "rank_loss": -ranking_score,
            }
        )

    ranked.sort(key=lambda item: (float(item["rank_loss"]), int(item["design_index"])))
    result.update(
        {
            "ranking_model_id": recipe["name"],
            "designs": ranked,
            "num_ranked_designs": len(ranked),
            "summary": {
                "ranking_score_best": max(float(item["ranking_score"]) for item in ranked),
                "rank_loss_best": min(float(item["rank_loss"]) for item in ranked),
                "ranking_time_s": sum(float(item["ranking_time_s"]) for item in ranked),
            },
            "ranking_timing": {
                "started_at_utc": ranking_started_at_utc,
                "completed_at_utc": utc_now(),
                "cell_wall_time_s": (
                    round(ranking_wall_time_s, 3) if ranking_wall_time_s is not None else None
                ),
                "models": model_timings,
            },
        }
    )
    if preselection is not None:
        result["design_candidates"] = candidates
        result["ranking_preselection"] = dict(preselection)
    atomic_text(cell_dir / "designs.fasta", fasta_text(ranked))
    atomic_json(cell_dir / "result.json", result)
    return result
