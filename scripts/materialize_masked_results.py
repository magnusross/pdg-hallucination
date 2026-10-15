"""Create a symlink-backed result tree with winning-seed target masking backfilled."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from binderopt.outputs import atomic_json, utc_now
from binderopt.ranking.self_consistency import score_self_consistency
from binderopt.structure.targets import target_from_config


def _load_mapping(path: Path) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping: {path}")
    return value


def _without_self_consistency(target: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in target.items() if key != "self_consistency"}


def _symlink_tree(source: Path, destination: Path) -> None:
    destination.mkdir()
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir()
        else:
            target = os.path.relpath(source_path, destination_path.parent)
            destination_path.symlink_to(target)


def _winning_score(model: dict[str, Any]) -> dict[str, Any]:
    winning_seed = int(model["winning_seed"])
    matches = [item for item in model["seed_scores"] if int(item["seed"]) == winning_seed]
    if len(matches) != 1:
        raise ValueError(f"expected one score for winning seed {winning_seed}")
    return matches[0]


def _backfill_cell(
    cell_dir: Path,
    *,
    target_config: dict[str, Any],
    source_cell: str,
    created_at_utc: str,
) -> int:
    result_path = cell_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recorded_target = result.get("config", {}).get("target")
    if not isinstance(recorded_target, dict):
        raise ValueError(f"cell has no resolved target config: {cell_dir}")
    if _without_self_consistency(recorded_target) != _without_self_consistency(target_config):
        raise ValueError(f"target identity differs from requested config: {cell_dir}")

    target = target_from_config(target_config)
    if target.self_consistency_residue_ranges is None:
        raise ValueError("requested target config has no self-consistency residue mask")
    expected_ranges = [
        [[int(start), int(end)] for start, end in chain]
        for chain in target.self_consistency_residue_ranges
    ]
    if recorded_target.get("self_consistency") == target_config.get("self_consistency"):
        return 0

    shards: dict[str, tuple[Path, dict[str, Any], dict[int, dict[str, Any]]]] = {}
    model_names = tuple(sorted(result["designs"][0]["ranking"]["models"]))
    for model_name in model_names:
        shard_path = cell_dir / "ranking" / f"{model_name}.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        records = {int(item["design_index"]): item for item in shard["designs"]}
        shards[model_name] = (shard_path, shard, records)

    count = 0
    for design in result["designs"]:
        design_index = int(design["design_index"])
        reference = cell_dir / str(design["selected_reference"]["structure"])
        for model_name in model_names:
            model = design["ranking"]["models"][model_name]
            winner = _winning_score(model)
            shard_path, shard, records = shards[model_name]
            del shard_path
            atom_names = shard["provenance"]["self_consistency"]["target_aligned_rmsd"][
                "atom_names"
            ]
            prediction = cell_dir / str(design["structures"][model_name])
            scored = score_self_consistency(
                reference,
                prediction,
                binder_length=int(result["binder_length"]),
                target_sequences=[chain.sequence for chain in target.chains],
                atom_names=atom_names,
                target_residue_ranges=target.self_consistency_residue_ranges,
            )
            if scored["self_consistency_target_residue_ranges"] != expected_ranges:
                raise ValueError(f"masked score records unexpected residue ranges: {cell_dir}")
            winner.update(scored)
            model["winning_dockq"] = float(scored["dockq"])

            shard_model = records[design_index]
            shard_winner = _winning_score(shard_model)
            shard_winner.update(scored)
            shard_model["winning_dockq"] = float(scored["dockq"])
            count += 1

    result["config"]["target"] = copy.deepcopy(target_config)
    provenance = {
        "created_at_utc": created_at_utc,
        "scope": "each judge's iPsae-winning seed only",
        "source_cell": source_cell,
        "target_id": target.target_id,
        "target_residue_ranges": expected_ranges,
    }
    result["self_consistency_backfill"] = provenance
    for shard_path, shard, _ in shards.values():
        shard["self_consistency_backfill"] = provenance
        atomic_json(shard_path, shard)
    atomic_json(result_path, result)
    return count


def materialize_results(
    source: Path, destination: Path, *, target_config_path: Path
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source result directory does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    target_config = _load_mapping(target_config_path)
    target_id = str(target_config["target_id"])
    created_at_utc = utc_now()
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    temporary.rmdir()
    try:
        _symlink_tree(source, temporary)
        cells = sorted(temporary.glob(f"{target_id}_*/result.json"))
        if not cells:
            raise ValueError(f"source has no {target_id} cells")
        updated_cells = 0
        updated_scores = 0
        for result_path in cells:
            count = _backfill_cell(
                result_path.parent,
                target_config=target_config,
                source_cell=str(result_path.parent.relative_to(temporary)),
                created_at_utc=created_at_utc,
            )
            if count:
                updated_cells += 1
                updated_scores += count
        manifest = {
            "created_at_utc": created_at_utc,
            "source": str(source),
            "target_config": str(target_config_path.resolve()),
            "target_id": target_id,
            "updated_cells": updated_cells,
            "updated_winning_scores": updated_scores,
        }
        atomic_json(temporary / "self_consistency_backfill.json", manifest)
        temporary.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target-config", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize_results(
        args.source, args.destination, target_config_path=args.target_config
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
