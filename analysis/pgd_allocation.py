"""Compare short, screened PGD runs with long PGD runs using finalized cells.

Run from the repository root with ``uv run python -m analysis.pgd_allocation``.
Analysis artifacts are separate from the immutable experiment results.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analysis.optimizer_analysis import (
    TARGET_LABELS,
    _cell_compute,
    _design_key,
    _optimizer_id,
    _publication_style,
    _target_order,
    _validated_model_metrics,
)
from binderopt.outputs import atomic_json
from binderopt.ranking.cells import select_designs_for_ranking
from binderopt.ranking.screening import _digest, verify_ranked_cell

ARMS = ("long", "short")
PLOT_ARMS = (*ARMS, "biohub")
COLORS = {"long": "#332288", "short": "#B58A24", "biohub": "#CC6677"}
LINE_STYLES = {"long": "-", "short": "-", "biohub": ":"}


def _scientific_config(result: dict) -> dict:
    """Compare scientific settings, allowing only allocation-policy differences."""
    config = copy.deepcopy(result["config"])
    for field in (
        "cell_dir",
        "cell_optimizer_id",
        "experiment_id",
        "launch_id",
        "num_designs",
        "output_root",
        "seed",
        "profile",
        "ranking_preselection",
    ):
        config.pop(field, None)
    for field in ("n_steps", "tail_select_steps"):
        config["design"].pop(field)
    for phase in config["optimizer"].get("phases", []):
        phase.pop("n_steps", None)
    return config


def _load_shard_allocation(
    control_roots: list[Path],
    short_root: Path,
    *,
    targets: list[str],
    seeds: list[int],
    short_config: dict,
    ranking_recipe: dict,
) -> tuple[dict[str, dict[str, list[dict]]], list[dict]]:
    """Keep complete target cohorts; never use partial shards or unequal coverage."""
    indexed: dict[str, dict[tuple[str, int], tuple[Path, dict]]] = {a: {} for a in ARMS}
    for arm, roots in (("long", control_roots), ("short", [short_root])):
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(root)
            for path in sorted(root.glob("*/result.json")):
                result = json.loads(path.read_text())
                if _optimizer_id(result) != "pgd" or result["target_id"] not in targets:
                    continue
                identity = (result["target_id"], int(result["base_seed"]))
                if identity[1] not in seeds:
                    continue
                if identity in indexed[arm]:
                    raise ValueError(f"duplicate {arm} target/seed: {identity}")
                indexed[arm][identity] = (path.parent, result)

    pools: dict[str, dict[str, list[dict]]] = {}
    coverage = []
    expected_candidates = int(short_config["num_designs"])
    expected_ranked = int(short_config["ranking_preselection"]["count"])
    expected_judges = {item["name"] for item in ranking_recipe["models"]}
    expected_seed_count = int(ranking_recipe["self_consistency"]["required_seed_count"])
    judge_seed_sets: set[tuple[int, ...]] = set()
    for target in targets:
        target_cells: dict[str, list[dict]] = {a: [] for a in ARMS}
        seen_keys: dict[str, set[tuple[int, ...]]] = {a: set() for a in ARMS}
        reasons = []
        for seed in seeds:
            pair = {}
            for arm in ARMS:
                found = indexed[arm].get((target, seed))
                if found is None:
                    reasons.append(f"{arm} seed {seed}: missing cell")
                    continue
                cell, result = found
                config = result["config"]
                candidate_count = expected_candidates if arm == "short" else expected_ranked
                candidates = result.get("design_candidates", result["designs"])
                if (
                    len(candidates) != candidate_count
                    or int(result["num_designs_requested"]) != candidate_count
                    or int(config["num_designs"]) != candidate_count
                    or sorted(int(d["design_index"]) for d in candidates)
                    != list(range(candidate_count))
                ):
                    raise ValueError(f"incomplete design candidates: {cell}")
                for field in ("n_steps", "tail_select_steps"):
                    expected = int(short_config["design"][field]) * (1 if arm == "short" else 2)
                    if int(config["design"][field]) != expected:
                        raise ValueError(f"unexpected {field}: {cell}")
                keys = {
                    int(d["design_index"]): _design_key(d["design_key"], cell=cell)
                    for d in candidates
                }
                if len(set(keys.values())) != len(keys) or seen_keys[arm].intersection(
                    keys.values()
                ):
                    raise ValueError(f"duplicate candidate design keys: {cell}")
                seen_keys[arm].update(keys.values())
                if arm == "short":
                    if config["ranking_preselection"] != short_config["ranking_preselection"]:
                        raise ValueError(f"unexpected screening policy: {cell}")
                elif config.get("ranking_preselection") is not None:
                    raise ValueError(f"control must rank all candidates: {cell}")
                if not result.get("ranking_model_id"):
                    reasons.append(f"{arm} seed {seed}: unranked")
                    continue
                if result["ranking_model_id"] != ranking_recipe["name"]:
                    raise ValueError(f"unexpected judge recipe: {cell}")
                designs = result["designs"]
                if len(designs) != expected_ranked or any(
                    "ranking_score" not in d for d in designs
                ):
                    reasons.append(f"{arm} seed {seed}: incomplete ranking")
                    continue
                selected, manifest = (
                    select_designs_for_ranking(result, config["ranking_preselection"])
                    if arm == "short"
                    else (candidates, None)
                )
                if {d["design_index"] for d in selected} != {d["design_index"] for d in designs}:
                    raise ValueError(f"ranked set differs from screening selection: {cell}")
                if manifest != result.get("ranking_preselection"):
                    raise ValueError(f"screening manifest mismatch: {cell}")
                scores = []
                for design in designs:
                    if _design_key(design["design_key"], cell=cell) != keys[design["design_index"]]:
                        raise ValueError(f"ranked design key differs from candidate: {cell}")
                    models, score, _, _ = _validated_model_metrics(design, cell=cell)
                    if set(models) != expected_judges:
                        raise ValueError(f"unexpected judges: {cell}")
                    seed_sets = []
                    for model in design["ranking"]["models"].values():
                        recorded = [int(item["seed"]) for item in model["seed_scores"]]
                        if len(recorded) != expected_seed_count or len(set(recorded)) != len(
                            recorded
                        ):
                            raise ValueError(f"incomplete judge seeds: {cell}")
                        seed_sets.append(tuple(sorted(recorded)))
                    if len(set(seed_sets)) != 1:
                        raise ValueError(f"judge seed sets differ: {cell}")
                    judge_seed_sets.update(seed_sets)
                    if len(judge_seed_sets) != 1:
                        raise ValueError(f"judge seed sets differ across cells: {cell}")
                    if not all(
                        np.isclose(float(design[field]), value)
                        for field, value in (
                            ("ranking_score", score),
                            ("ipsae_ranking_score", score),
                            ("rank_loss", -score),
                        )
                    ):
                        raise ValueError(f"ranking identity mismatch: {cell}")
                    scores.append(score)
                design_hours, ranking_hours = _cell_compute(result, cell=cell)
                if design_hours + ranking_hours <= 0:
                    raise ValueError(f"non-positive GPU time: {cell}")
                target_cells[arm].append(
                    {
                        "target": target,
                        "arm": arm,
                        "seed": seed,
                        "cell": str(cell.resolve()),
                        "scores": np.asarray(scores),
                        "sequences": [d["sequence"] for d in designs],
                        "design_gpu_hours": design_hours,
                        "ranking_gpu_hours": ranking_hours,
                        "git_sha": result.get("git_sha"),
                    }
                )
                pair[arm] = result
            if len(pair) == 2:
                if _scientific_config(pair["long"]) != _scientific_config(pair["short"]):
                    raise ValueError(f"scientific configurations differ: {target} seed {seed}")
                long_keys = {
                    int(d["design_index"]): _design_key(d["design_key"], cell=Path(target))
                    for d in pair["long"]["designs"]
                }
                short_keys = {
                    int(d["design_index"]): _design_key(d["design_key"], cell=Path(target))
                    for d in pair["short"]["design_candidates"]
                }
                if any(short_keys.get(index) != key for index, key in long_keys.items()):
                    raise ValueError(f"shared design keys differ: {target} seed {seed}")
        coverage.append({"target": target, "included": not reasons, "reasons": reasons})
        if not reasons:
            pools[target] = target_cells
    if not pools:
        raise ValueError("no complete target cohorts")
    return pools, coverage


def _scores(result: dict, cell: Path, recipe: dict, ranking_seed: int) -> np.ndarray:
    """Validate the primary score and the complete configured judge seed sets."""
    if result.get("ranking_model_id") != recipe["name"]:
        raise ValueError(f"unexpected judge recipe: {cell}")
    expected_models = {model["name"] for model in recipe["models"]}
    expected_seeds = sorted(ranking_seed + int(offset) for offset in recipe["seed_offsets"])
    scores = []
    for design in result["designs"]:
        if design["ranking"]["base_seed"] != ranking_seed:
            raise ValueError(f"ranking seed differs: {cell}")
        models, score, _, _ = _validated_model_metrics(design, cell=cell)
        if set(models) != expected_models:
            raise ValueError(f"unexpected judges: {cell}")
        for model in design["ranking"]["models"].values():
            if sorted(int(s["seed"]) for s in model["seed_scores"]) != expected_seeds:
                raise ValueError(f"unexpected judge seeds: {cell}")
        for field, value in (
            ("ranking_score", score),
            ("ipsae_ranking_score", score),
            ("rank_loss", -score),
        ):
            if not np.isclose(design[field], value):
                raise ValueError(f"ranking identity mismatch: {cell}")
        scores.append(score)
    return np.asarray(scores)


def _screen_hours(root: Path, seed: int, manifest: dict) -> tuple[float, float]:
    """Charge MPNN and one-seed screening for every candidate, including discards."""
    mpnn = json.loads((root / "mpnn" / f"seed{seed}.json").read_text())
    screen = json.loads((root / "screening" / f"seed{seed}.json").read_text())
    candidates = [d for d in manifest["candidates"] if d["base_seed"] == seed]
    if (
        screen["base_seed"] != seed
        or mpnn["base_seed"] != seed
        or screen["provenance"] != manifest["provenance"]
        or mpnn["provenance"] != manifest["provenance"]
        or screen["mpnn_sha256"] != _digest(mpnn)
    ):
        raise ValueError(f"screening provenance differs: {root} seed {seed}")
    by_index = {d["design_index"]: d for d in candidates}
    redesigned = {d["design_index"]: d for d in mpnn["shard"]["designs"]}
    if (
        len(screen["designs"]) != len(candidates)
        or set(redesigned) != set(by_index)
        or {d["design_index"] for d in screen["designs"]} != set(by_index)
    ):
        raise ValueError(f"incomplete all-candidate screening: {root} seed {seed}")
    mpnn_digest = _digest(mpnn)
    for scored in screen["designs"]:
        selected = by_index[scored["design_index"]]
        if (
            any(
                scored[k] != selected[k]
                for k in ("design_key", "sequence", "optimizer_sequence", "ipsae_min")
            )
            or scored["sequence"] != redesigned[scored["design_index"]]["sequence"]
            or selected["mpnn_sha256"] != mpnn_digest
        ):
            raise ValueError(f"screening candidate identity differs: {root} seed {seed}")
    try:
        mpnn_task_seconds = float(mpnn["shard"]["timing"]["task_wall_time_s"])
    except KeyError:
        try:
            mpnn_task_seconds = float(mpnn["shard"]["timing"]["wall_time_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"missing MPNN timing in pooled artifact: {root} seed {seed}"
            ) from error
        warnings.warn(
            "MPNN shard has no task_wall_time_s; falling back to wall_time_s, which "
            "omits worker startup and model setup. Pooled short-run GPU-hours may be "
            "understated; adjust runtime comparisons accordingly.",
            UserWarning,
            stacklevel=2,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid parent-side MPNN task timing: {root} seed {seed}"
        ) from error
    seconds = [mpnn_task_seconds, screen["wall_time_s"]]
    if any(not np.isfinite(s) or s <= 0 for s in seconds):
        raise ValueError(f"invalid screening timing: {root} seed {seed}")
    if any(int(d["config"]["profile"]["resources"]["gpus_per_task"]) != 1 for d in (mpnn, screen)):
        raise ValueError("screening timing requires the recorded single-GPU task")
    return seconds[0] / 3600, seconds[1] / 3600


def load_allocation(
    control_roots, short_root, *, targets, seeds, short_config, ranking_recipe, source_root=None
):
    """Load complete target cohorts from shard selections or global screening."""
    if not any(short_root.glob("*/selection.json")):
        return _load_shard_allocation(
            control_roots,
            short_root,
            targets=targets,
            seeds=seeds,
            short_config=short_config,
            ranking_recipe=ranking_recipe,
        )
    controls = {}
    for root in control_roots:
        for path in root.glob("*/result.json"):
            result = json.loads(path.read_text())
            identity = (result["target_id"], int(result["base_seed"]))
            if _optimizer_id(result) != "pgd" or identity[0] not in targets:
                continue
            if identity in controls:
                raise ValueError(f"duplicate PGD control: {identity}")
            controls[identity] = (path.parent, result)
    pools, coverage = {}, []
    expected_per_seed = int(short_config["ranking_preselection"]["count"])
    for target in targets:
        root = short_root / target
        path = root / "selection.json"
        if not path.is_file():
            coverage.append(
                {"target": target, "included": False, "reasons": ["missing global selection"]}
            )
            continue
        manifest = json.loads(path.read_text())
        candidates = manifest["candidates"]
        ordered = sorted(
            candidates, key=lambda d: (-d["ipsae_min"], d["base_seed"], d["design_index"])
        )
        expected_count = expected_per_seed * len(seeds)
        if (
            manifest["selected_count"] != expected_count
            or manifest["candidate_count"] != int(short_config["num_designs"]) * len(seeds)
            or candidates != ordered
            or len(candidates) != manifest["candidate_count"]
            or any(
                d["selected"] != (i < expected_count) or d["selection_rank"] != i + 1
                for i, d in enumerate(candidates)
            )
        ):
            raise ValueError(f"invalid global selection: {path}")
        reasons = []
        for seed in seeds:
            if not (root / f"{target}_pgd_seed{seed}" / "result.json").is_file():
                reasons.append(f"short seed {seed}: missing pooled ranking")
            if (target, seed) not in controls:
                reasons.append(f"long seed {seed}: missing control")
        available_seeds = [
            seed
            for seed in seeds
            if (root / f"{target}_pgd_seed{seed}" / "result.json").is_file()
            and (target, seed) in controls
        ]
        coverage.append(
            {
                "target": target,
                "included": bool(available_seeds),
                "complete": not reasons,
                "reasons": reasons,
                "included_seeds": available_seeds,
            }
        )
        if not available_seeds:
            continue
        arms = {"long": [], "short": []}
        seen_keys = set()
        for seed in available_seeds:
            cell = root / f"{target}_pgd_seed{seed}"
            result = json.loads((cell / "result.json").read_text())
            cfg = result["pool_config"].copy()
            cfg["pool_dir"] = str(root)
            if cfg["ranking"] != ranking_recipe or cfg["source_seeds"] != seeds:
                raise ValueError(f"pooled recipe or seed cohort differs: {cell}")
            verify_ranked_cell(cfg, manifest, seed)
            control_cell, control = controls[target, seed]
            source = {"config": result["source_config"]}
            if _scientific_config(source) != _scientific_config(control):
                raise ValueError(f"scientific configurations differ: {cell}")
            for field in ("n_steps", "tail_select_steps"):
                if (
                    source["config"]["design"][field] != short_config["design"][field]
                    or control["config"]["design"][field] != 2 * short_config["design"][field]
                ):
                    raise ValueError(f"unexpected {field}: {cell}")
            generated = result["design_candidates"]
            if len(generated) != int(short_config["num_designs"]) or sorted(
                d["design_index"] for d in generated
            ) != list(range(int(short_config["num_designs"]))):
                raise ValueError(f"incomplete generation: {cell}")
            keys = {d["design_index"]: d["design_key"] for d in generated}
            key_set = {_design_key(key, cell=cell) for key in keys.values()}
            if len(key_set) != len(generated) or seen_keys.intersection(key_set):
                raise ValueError(f"duplicate candidate keys: {cell}")
            seen_keys.update(key_set)
            if (
                len(control["designs"]) != expected_per_seed
                or sorted(d["design_index"] for d in control["designs"])
                != list(range(expected_per_seed))
                or control["config"].get("ranking_preselection") is not None
                or any(keys.get(d["design_index"]) != d["design_key"] for d in control["designs"])
            ):
                raise ValueError(f"control coverage or paired keys differ: {control_cell}")
            mpnn_hours, screening_hours = _screen_hours(root, seed, manifest)
            for arm, data, directory in (("long", control, control_cell), ("short", result, cell)):
                design_hours, rank_hours = _cell_compute(data, cell=directory)
                extra = mpnn_hours + screening_hours if arm == "short" else 0
                arms[arm].append(
                    {
                        "target": target,
                        "arm": arm,
                        "seed": seed,
                        "cell": str(directory.resolve()),
                        "scores": _scores(data, directory, ranking_recipe, cfg["ranking_seed"]),
                        "sequences": [d["sequence"] for d in data["designs"]],
                        "design_keys": [d["design_key"] for d in data["designs"]],
                        "design_indices": [d["design_index"] for d in data["designs"]],
                        "design_gpu_hours": design_hours,
                        "ranking_gpu_hours": rank_hours + extra,
                        "full_ranking_gpu_hours": rank_hours,
                        "mpnn_gpu_hours": mpnn_hours if arm == "short" else 0,
                        "screening_gpu_hours": screening_hours if arm == "short" else 0,
                        "git_sha": data.get("git_sha"),
                        "global_selection": arm == "short",
                    }
                )
        pools[target] = arms
        # Global selection depended on the entire source pool. Charge generation,
        # MPNN and screening even for shards whose full ranking is unavailable.
        for seed in set(seeds) - set(available_seeds):
            mpnn = json.loads((root / "mpnn" / f"seed{seed}.json").read_text())
            source_dir = (
                Path(source_root) if source_root is not None else Path(mpnn["config"]["source_dir"])
            )
            source = json.loads(
                (source_dir / f"{target}_pgd_seed{seed}" / "result.json").read_text()
            )
            design = source["design_timing"]
            source_hours = design["process_wall_time_s"] * design["gpu_count"] / 3600
            mpnn_hours, screen_hours = _screen_hours(root, seed, manifest)
            first = arms["short"][0]
            first["design_gpu_hours"] += source_hours
            first["ranking_gpu_hours"] += mpnn_hours + screen_hours
            first["mpnn_gpu_hours"] += mpnn_hours
            first["screening_gpu_hours"] += screen_hours
            first.setdefault("unranked_source_seeds", []).append(seed)
    if not pools:
        raise ValueError("no complete target cohorts")
    return pools, coverage


def _top_mean(values: np.ndarray, fraction: float) -> float:
    count = math.ceil(len(values) * fraction)
    return float(np.partition(values, -count)[-count:].mean())


def add_logit_gd(pools: dict, control_roots: list[Path]) -> None:
    """Add complete Logit GD cohorts paired to the saved long-PGD controls."""
    indexed = {}
    for root in control_roots:
        for path in sorted(root.glob("*/result.json")):
            result = json.loads(path.read_text())
            if _optimizer_id(result) != "biohub" or result["target_id"] not in pools:
                continue
            identity = (result["target_id"], int(result["base_seed"]))
            if identity in indexed:
                raise ValueError(f"duplicate Logit GD target/seed: {identity}")
            indexed[identity] = (path.parent, result)
    expected = {(t, c["seed"]) for t, arms in pools.items() for c in arms["long"]}
    if not expected.issubset(indexed):
        raise ValueError("Logit GD does not match the complete target/seed cohort")
    for target, arms in pools.items():
        cells = []
        for reference in arms["long"]:
            cell, result = indexed[target, reference["seed"]]
            control = json.loads((Path(reference["cell"]) / "result.json").read_text())
            config = _scientific_config(result)
            control_config = _scientific_config(control)
            config.pop("optimizer")
            control_config.pop("optimizer")
            if (
                config != control_config
                or result["config"]["design"] != control["config"]["design"]
                or result["config"]["profile"].get("design")
                != control["config"]["profile"].get("design")
            ):
                raise ValueError(f"Logit GD scientific configuration differs: {cell}")
            designs = result["designs"]
            if (
                result.get("ranking_model_id") != control["ranking_model_id"]
                or len(designs) != len(control["designs"])
                or len(designs) != int(result["num_designs_requested"])
            ):
                raise ValueError(f"incomplete Logit GD ranking: {cell}")
            keys = {
                int(d["design_index"]): _design_key(d["design_key"], cell=cell) for d in designs
            }
            control_keys = {
                int(d["design_index"]): _design_key(d["design_key"], cell=cell)
                for d in control["designs"]
            }
            if len(keys) != len(designs) or keys != control_keys:
                raise ValueError(f"Logit GD paired design keys differ: {cell}")
            control_models = control["designs"][0]["ranking"]["models"]
            scores = []
            for design in designs:
                models, score, _, _ = _validated_model_metrics(design, cell=cell)
                if set(models) != set(control_models):
                    raise ValueError(f"Logit GD judges differ: {cell}")
                for name in models:
                    observed = sorted(
                        s["seed"] for s in design["ranking"]["models"][name]["seed_scores"]
                    )
                    expected_seeds = sorted(s["seed"] for s in control_models[name]["seed_scores"])
                    if observed != expected_seeds:
                        raise ValueError(f"Logit GD judge seeds differ: {cell}")
                if not all(
                    np.isclose(design[field], value)
                    for field, value in (
                        ("ranking_score", score),
                        ("ipsae_ranking_score", score),
                        ("rank_loss", -score),
                    )
                ):
                    raise ValueError(f"Logit GD ranking identity differs: {cell}")
                scores.append(score)
            design_hours, ranking_hours = _cell_compute(result, cell=cell)
            if design_hours + ranking_hours <= 0:
                raise ValueError(f"non-positive Logit GD GPU time: {cell}")
            cells.append(
                {
                    "target": target,
                    "arm": "biohub",
                    "seed": reference["seed"],
                    "cell": str(cell.resolve()),
                    "scores": np.asarray(scores),
                    "sequences": [d["sequence"] for d in designs],
                    "design_keys": [d["design_key"] for d in designs],
                    "design_indices": [d["design_index"] for d in designs],
                    "design_gpu_hours": design_hours,
                    "ranking_gpu_hours": ranking_hours,
                    "git_sha": result.get("git_sha"),
                }
            )
        arms["biohub"] = cells


def match_pool_sizes(pools: dict, coverage: list[dict]) -> None:
    """Match control sample sizes by paired keys, without consulting judge scores."""
    for target, arms in pools.items():
        count = sum(len(c["scores"]) for c in arms["short"])
        controls = arms["long"]
        available = sum(len(c["scores"]) for c in controls)
        if available < count:
            raise ValueError(f"insufficient matched controls: {target}")
        row = next(r for r in coverage if r["target"] == target)
        row.update(ranked_per_arm=count, available_controls=available)
        if available == count:
            continue
        keys = sorted(tuple(k) for c in controls for k in c["design_keys"])
        if len(set(keys)) != available:
            raise ValueError(f"duplicate control keys: {target}")
        selected = set(keys[:count])
        row["control_subset"] = "lowest design_key tuples, shared by PGD and Logit GD"
        for arm in ("long", "biohub"):
            for cell in arms[arm]:
                mask = np.asarray([tuple(key) in selected for key in cell["design_keys"]])
                cell["measured_designs"] = len(cell["scores"])
                cell["scores"] = cell["scores"][mask]
                for field in ("sequences", "design_keys", "design_indices"):
                    cell[field] = [
                        value for value, keep in zip(cell[field], mask, strict=True) if keep
                    ]
            observed = {tuple(k) for c in arms[arm] for k in c["design_keys"]}
            if observed != selected:
                raise ValueError(f"matched control subset differs: {target} {arm}")


def summarize(
    pools: dict,
    thresholds: np.ndarray,
    *,
    top_fraction: float,
    n_bootstrap: int,
    analysis_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Paired shard bootstrap; pooled yield is sum(hits) / sum(GPU-hours)."""
    if not 0 < top_fraction <= 1 or n_bootstrap < 0:
        raise ValueError("invalid top fraction or bootstrap count")
    if n_bootstrap and any(
        c.get("global_selection")
        for arms in pools.values()
        for cells in arms.values()
        for c in cells
    ):
        raise ValueError("seed-shard bootstrap is invalid after global pool selection")
    rng = np.random.default_rng(analysis_seed)
    arm_names = tuple(next(iter(pools.values())))
    counts = {a: np.zeros(len(thresholds)) for a in arm_names}
    unique_counts = {a: np.zeros(len(thresholds)) for a in arm_names}
    hours = dict.fromkeys(arm_names, 0.0)
    boot_counts = {a: np.zeros((n_bootstrap, len(thresholds))) for a in arm_names}
    boot_hours = {a: np.zeros(n_bootstrap) for a in arm_names}
    endpoints, compute, target_yields = [], [], []
    for target, arms in pools.items():
        n_shards = len(arms["long"])
        if set(arms) != set(arm_names) or any(
            [c["seed"] for c in arms["long"]] != [c["seed"] for c in arms[a]] for a in arm_names
        ):
            raise ValueError("bootstrap requires matched seed shards")
        draws = rng.integers(0, n_shards, size=(n_bootstrap, n_shards))
        bootstrap_means = {}
        target_endpoints = {}
        for arm in arm_names:
            cells = arms[arm]
            values = np.concatenate([c["scores"] for c in cells])
            cell_hours = np.asarray([c["design_gpu_hours"] + c["ranking_gpu_hours"] for c in cells])
            # Strict exceedance, including ties at the threshold consistently.
            cell_counts = np.asarray(
                [(c["scores"][:, None] > thresholds).sum(axis=0) for c in cells]
            )
            counts[arm] += cell_counts.sum(axis=0)
            best_by_sequence: dict[str, float] = {}
            for cell in cells:
                for sequence, score in zip(cell["sequences"], cell["scores"], strict=True):
                    best_by_sequence[sequence] = max(best_by_sequence.get(sequence, -np.inf), score)
            unique_scores = np.asarray(list(best_by_sequence.values()))
            unique_counts[arm] += (unique_scores[:, None] > thresholds).sum(axis=0)
            hours[arm] += float(cell_hours.sum())
            weights = np.stack([(draws == i).sum(axis=1) for i in range(n_shards)], axis=1)
            sampled_counts = weights @ cell_counts
            sampled_hours = weights @ cell_hours
            boot_counts[arm] += sampled_counts
            boot_hours[arm] += sampled_hours
            rate_low, rate_high = (
                np.quantile(sampled_counts / sampled_hours[:, None], [0.025, 0.975], axis=0)
                if n_bootstrap
                else ([None] * len(thresholds), [None] * len(thresholds))
            )
            target_counts = cell_counts.sum(axis=0)
            target_unique = (unique_scores[:, None] > thresholds).sum(axis=0)
            target_hours = float(cell_hours.sum())
            for i, threshold in enumerate(thresholds):
                target_yields.append(
                    {
                        "target": target,
                        "arm": arm,
                        "threshold": float(threshold),
                        "hits": int(target_counts[i]),
                        "gpu_hours": target_hours,
                        "hits_per_gpu_hour": float(target_counts[i] / target_hours),
                        "unique_hits": int(target_unique[i]),
                        "unique_hits_per_gpu_hour": float(target_unique[i] / target_hours),
                        "ci_low": float(rate_low[i]) if n_bootstrap else None,
                        "ci_high": float(rate_high[i]) if n_bootstrap else None,
                    }
                )
            sampled = (
                np.stack([c["scores"] for c in cells])[draws].reshape(n_bootstrap, values.size)
                if n_bootstrap
                else np.empty((0, values.size))
            )
            n_top = math.ceil(sampled.shape[1] * top_fraction)
            means = np.partition(sampled, -n_top, axis=1)[:, -n_top:].mean(axis=1)
            bootstrap_means[arm] = means
            low, high = np.quantile(means, [0.025, 0.975]) if n_bootstrap else (None, None)
            endpoints.append(
                {
                    "target": target,
                    "arm": arm,
                    "mean": _top_mean(values.ravel(), top_fraction),
                    "ci_low": float(low) if n_bootstrap else None,
                    "ci_high": float(high) if n_bootstrap else None,
                    "ranked": values.size,
                    "top_count": n_top,
                    "short_minus_long": None,
                    "difference_ci_low": None,
                    "difference_ci_high": None,
                }
            )
            target_endpoints[arm] = endpoints[-1]
            compute.append(
                {
                    "target": target,
                    "arm": arm,
                    "shards": n_shards,
                    "ranked": values.size,
                    "unique_sequences": len({s for c in cells for s in c["sequences"]}),
                    "design_gpu_hours": sum(c["design_gpu_hours"] for c in cells),
                    "ranking_gpu_hours": sum(c["ranking_gpu_hours"] for c in cells),
                    "mpnn_gpu_hours": sum(c.get("mpnn_gpu_hours", 0) for c in cells),
                    "screening_gpu_hours": sum(c.get("screening_gpu_hours", 0) for c in cells),
                    "full_ranking_gpu_hours": sum(
                        c.get("full_ranking_gpu_hours", c["ranking_gpu_hours"]) for c in cells
                    ),
                    "total_gpu_hours": float(cell_hours.sum()),
                }
            )
        difference = bootstrap_means["short"] - bootstrap_means["long"]
        low, high = np.quantile(difference, [0.025, 0.975]) if n_bootstrap else (None, None)
        for arm in ARMS:
            row = target_endpoints[arm]
            row["short_minus_long"] = (
                target_endpoints["short"]["mean"] - target_endpoints["long"]["mean"]
            )
            row["difference_ci_low"] = float(low) if n_bootstrap else None
            row["difference_ci_high"] = float(high) if n_bootstrap else None
    yield_rows = []
    for arm in arm_names:
        rates = boot_counts[arm] / boot_hours[arm][:, None]
        low, high = (
            np.quantile(rates, [0.025, 0.975], axis=0)
            if n_bootstrap
            else ([None] * len(thresholds), [None] * len(thresholds))
        )
        for i, threshold in enumerate(thresholds):
            yield_rows.append(
                {
                    "target": "Total",
                    "arm": arm,
                    "threshold": float(threshold),
                    "hits": int(counts[arm][i]),
                    "gpu_hours": hours[arm],
                    "hits_per_gpu_hour": float(counts[arm][i] / hours[arm]),
                    "unique_hits": int(unique_counts[arm][i]),
                    "unique_hits_per_gpu_hour": float(unique_counts[arm][i] / hours[arm]),
                    "ci_low": float(low[i]) if n_bootstrap else None,
                    "ci_high": float(high[i]) if n_bootstrap else None,
                }
            )
    return yield_rows + target_yields, endpoints, compute


def plot_allocation(
    yield_rows: list[dict],
    *,
    labels: dict[str, str],
    output_dir: Path,
) -> Any:
    """Plot pooled yield from the original ranked pools, above threshold 0.5."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    _publication_style(plt)
    fig, axis = plt.subplots(figsize=(2.6, 1.7))
    fig.subplots_adjust(left=0.195, right=0.98, bottom=0.26, top=0.97)
    plotted = [r for r in yield_rows if r["target"] == "Total" and 0.5 <= r["threshold"] <= 0.9]
    if not plotted:
        raise ValueError("pooled yield requires Total rows at thresholds >= 0.5")
    maximum = max(r["hits_per_gpu_hour"] for r in plotted)
    plot_arms = [arm for arm in PLOT_ARMS if arm in labels]
    for arm in plot_arms:
        selected = sorted([r for r in plotted if r["arm"] == arm], key=lambda r: r["threshold"])
        if not selected:
            raise ValueError(f"missing pooled yield arm: {arm}")
        x = [r["threshold"] for r in selected]
        axis.plot(
            x,
            [r["hits_per_gpu_hour"] for r in selected],
            color=COLORS[arm],
            lw=1.1,
            ls=LINE_STYLES[arm],
        )
    axis.set(
        xlim=(0.5, 0.9),
        ylim=(0, maximum * 1.03 if maximum else 1),
        xlabel="iPsae threshold",
        ylabel="Hits / GPU-hour",
    )
    axis.set_xticks(np.linspace(0.5, 0.9, 5))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(length=2.5, pad=2)
    axis.grid(axis="y", color=".92", lw=0.35)
    axis.legend(
        [Line2D([], [], color=COLORS[a], lw=1.1, ls=LINE_STYLES[a]) for a in plot_arms],
        [labels[a] for a in plot_arms],
        loc="upper right",
        ncol=1,
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"pgd_allocation_value.{suffix}", dpi=300)
    return fig


def hit_increase_table(
    yield_rows: list[dict], *, targets: list[str], threshold: float
) -> list[dict]:
    """Compare rates, keeping raw counts/hours and explicit excluded targets."""
    table = []
    for target in targets:
        selected = {
            r["arm"]: r
            for r in yield_rows
            if r["target"] == target and np.isclose(r["threshold"], threshold, atol=1e-12, rtol=0)
        }
        if selected and set(selected) != set(PLOT_ARMS):
            raise ValueError(f"table requires all three campaigns: {target}")
        if not selected and any(r["target"] == target for r in yield_rows):
            raise ValueError(f"threshold {threshold} is not present: {target}")
        row = {"target": target, "threshold": threshold, "included": bool(selected)}
        for arm in PLOT_ARMS:
            values = selected.get(arm, {})
            row[f"{arm}_hits"] = values.get("hits")
            row[f"{arm}_gpu_hours"] = values.get("gpu_hours")
            row[f"{arm}_hits_per_gpu_hour"] = values.get("hits_per_gpu_hour")
        for baseline in ("long", "biohub"):
            rate = row[f"{baseline}_hits_per_gpu_hour"]
            row[f"short_vs_{baseline}_increase_pct"] = (
                100 * (row["short_hits_per_gpu_hour"] / rate - 1) if rate else None
            )
        table.append(row)
    return table


def write_hit_table(
    table: list[dict],
    output_dir: Path,
    *,
    threshold: float,
    short_label: str = "PGD, 75 steps + iPTM",
) -> None:
    stem = "hit_increase_" + f"{threshold:g}".replace(".", "p")
    with (output_dir / f"{stem}.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        f"Hits per GPU-hour for primary iPsae > {threshold:g}.",
        "",
        "| Target | Logit GD, 150 | PGD, 150 | PGD, 75 + screen | Increase vs PGD 150 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table:
        label = TARGET_LABELS.get(row["target"], row["target"]).replace(r"$\alpha$", "\u03b1")
        values = [
            f"{row[f'{arm}_hits_per_gpu_hour']:.2f}" if row["included"] else "—"
            for arm in ("biohub", "long", "short")
        ]
        gain = row["short_vs_long_increase_pct"]
        values.append(f"{gain:+.1f}%" if gain is not None else "—")
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines.extend(
        [
            "",
            "Increase compares compute-normalized yields. Incomplete targets use matched",
            "sample sizes and recorded compute; all-candidate screening costs remain included.",
            "Raw counts and GPU-hours are in the CSV.",
        ]
    )
    (output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")
    (output_dir / f"{stem}.tex").write_text(
        latex_hit_table(table, threshold=threshold, short_label=short_label)
    )


def latex_hit_table(
    table: list[dict],
    *,
    threshold: float,
    short_label: str = "PGD, 75 steps + iPTM",
) -> str:
    """Transpose target yields and bold the highest unrounded rate per target."""
    headers = [TARGET_LABELS.get(r["target"], r["target"].replace("_", r"\_")) for r in table]
    lines = [
        r"% Requires \usepackage{booktabs}.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "r" * len(table) + "}",
        r"\toprule",
        "Method & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for arm, label in (
        ("biohub", "Logit GD, 150 steps"),
        ("long", "PGD, 150 steps"),
        ("short", short_label),
    ):
        values = []
        for row in table:
            if not row["included"]:
                values.append("--")
                continue
            rate = row[f"{arm}_hits_per_gpu_hour"]
            best = max(row[f"{a}_hits_per_gpu_hour"] for a in PLOT_ARMS)
            value = f"{rate:.2f}"
            if np.isclose(rate, best, atol=1e-12, rtol=0):
                value = r"\textbf{" + value + "}"
            values.append(value)
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.append(r"\midrule")
    gains = [
        f"{r['short_vs_long_increase_pct']:+.1f}" + r"\%"
        if r["short_vs_long_increase_pct"] is not None
        else "--"
        for r in table
    ]
    lines.extend(
        [
            "PGD 75 gain vs PGD 150 & " + " & ".join(gains) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Hits per GPU-hour for primary iPsae $> "
            + f"{threshold:g}"
            + r"$. Bold marks the highest yield for each target. The final row gives the "
            r"compute-adjusted percentage change from PGD 150 to PGD 75. "
            r"Dashes indicate incomplete target rankings.}",
            r"\label{tab:pgd-hit-yield-" + f"{threshold:g}".replace(".", "p") + "}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short-root", type=Path, default=Path("outputs/pgd_75_pool1024_of2048_fast_ipsae_v2")
    )
    parser.add_argument(
        "--control-roots",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/eight_target_optimizer_512_v1_trem2_masked"),
            Path("outputs/ten_target_biohub_pgd_512_v1"),
        ],
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="relocated source generation cells; overrides the saved source_dir for unranked costs",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiment/pgd_75_top64_of128_2048_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/artifacts/pgd_allocation_pooled")
    )
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--analysis-seed", type=int, default=20261005)
    parser.add_argument("--table-threshold", type=float, default=0.65)
    args = parser.parse_args()
    config = yaml.safe_load(args.experiment_config.read_text())
    params = config["hydra"]["sweeper"]["params"]
    targets = _target_order(str(params["target"]).split(","))
    seeds = [int(s) for s in str(params["seed"]).split(",")]
    recipe = yaml.safe_load(Path("configs/ranking/full.yaml").read_text())
    pools, coverage = load_allocation(
        args.control_roots,
        args.short_root,
        targets=targets,
        seeds=seeds,
        short_config=config,
        ranking_recipe=recipe,
        source_root=args.source_root,
    )
    add_logit_gd(pools, args.control_roots)
    match_pool_sizes(pools, coverage)
    thresholds = np.linspace(0, 1, 101)
    yields, endpoints, compute = summarize(
        pools,
        thresholds,
        top_fraction=0.10,
        n_bootstrap=args.n_bootstrap,
        analysis_seed=args.analysis_seed,
    )
    pooled = any(
        c.get("global_selection")
        for arms in pools.values()
        for cells in arms.values()
        for c in cells
    )
    labels = {
        "long": "PGD, 150 steps",
        "short": "PGD, 75 + pooled screen" if pooled else "PGD, 75 steps + screen",
        "biohub": "Logit GD, 150 steps",
    }
    plot_allocation(yields, labels=labels, output_dir=args.output_dir)
    table = hit_increase_table(yields, targets=targets, threshold=args.table_threshold)
    write_hit_table(
        table,
        args.output_dir,
        threshold=args.table_threshold,
        short_label="PGD, 75 steps + pooled iPsae" if pooled else "PGD, 75 steps + iPTM",
    )
    for name, rows in (
        ("threshold_yield", yields),
        ("top10_quality", endpoints),
        ("compute", compute),
    ):
        with (args.output_dir / f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(
        args.output_dir / "analysis.json",
        {
            "control_roots": [str(p.resolve()) for p in args.control_roots],
            "short_root": str(args.short_root.resolve()),
            "source_root": (
                str(args.source_root.resolve()) if args.source_root is not None else None
            ),
            "coverage": coverage,
            "n_bootstrap": args.n_bootstrap,
            "analysis_seed": args.analysis_seed,
            "top_fraction": 0.10,
            "threshold_operator": ">",
            "threshold_spacing": float(thresholds[1] - thresholds[0]),
            "table_threshold": args.table_threshold,
            "figure": {
                "pooling": "sum hits / sum GPU-hours across matched target cohorts",
                "threshold_min": 0.5,
                "threshold_max": 0.9,
                "xscale": "linear",
                "yscale": "linear",
                "size_inches": [2.6, 1.7],
                "uncertainty_bands": False,
            },
            "compute": (
                "design process + all-candidate MPNN + all-candidate Fast screen + full ranking"
                if pooled
                else "allocated GPU-hours: design process + ranking cell wall time"
            ),
            "compute_exclusions": "queue delays, failed attempts and node packing idle time",
            "selection": "global one-seed Fast iPsae" if pooled else "per-shard tail iPTM",
            "pooling": (
                "sum hits / sum GPU-hours within each target; Total rows pool matched cohorts"
            ),
            "incomplete_cohorts": (
                "Use available paired seed shards; match control counts by score-blind keys. "
                "Keep recorded control compute, and all source-pool design/MPNN/screen compute."
            ),
            "bootstrap": (
                "paired seed shards, independently resampled within each fixed target"
                if args.n_bootstrap
                else None
            ),
            "cells": [
                {k: v for k, v in c.items() if k not in ("scores", "sequences")}
                for arms in pools.values()
                for cells in arms.values()
                for c in cells
            ],
            "historical_source_limit": (
                "Dirty historical revisions are not fully recoverable from git hashes."
            ),
        },
    )
    print(json.dumps({"coverage": coverage, "compute": compute}, indent=2))


if __name__ == "__main__":
    main()
