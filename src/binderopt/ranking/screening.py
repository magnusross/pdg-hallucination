"""Screen a complete target pool, then fully rank its survivors in source-cell lanes.

Source cells are immutable inputs. SolubleMPNN redesigns the entire pool before
screening. Publication judges reuse those exact sequences after global selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from binderopt.outputs import atomic_json, git_state, utc_now
from binderopt.ranking.cells import (
    aggregate_cell_ranking,
    design_cell_result,
    discover_design_cells,
    process_failure,
    requested_seeds,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fast_model(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Use the reviewed fast judge settings without inventing a ranking recipe."""
    if recipe["name"] != "full" or recipe["scientific_validity"] != "publication":
        raise ValueError("pooled publication ranking requires the full recipe")
    requested_seeds(recipe, 0)
    return dict(next(spec for spec in recipe["models"] if spec["name"] == "esmfold2_fast"))


def audit_pool(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Require every configured seed and compatible generation provenance."""
    expected = [int(seed) for seed in cfg["source_seeds"]]
    count = int(cfg["candidates_per_cell"])
    if not expected or len(set(expected)) != len(expected) or min(expected) < 0 or count <= 0:
        raise ValueError("source seeds must be non-empty, unique and non-negative")
    if not 0 < int(cfg["selection_count"]) <= len(expected) * count:
        raise ValueError("selection_count must fit the complete target pool")
    fast_model(cfg["ranking"])
    if int(cfg["screening_seed"]) < 0 or int(cfg["ranking_seed"]) < 0:
        raise ValueError("screening and ranking seeds must be non-negative")
    source_root = Path(str(cfg["source_dir"])).resolve()
    pool_root = Path(str(cfg["pool_dir"])).resolve()
    if pool_root == source_root or source_root in pool_root.parents:
        raise ValueError("pool outputs must be separate from source experiment")
    if cfg["experiment_id"] == cfg["source_experiment_id"]:
        raise ValueError("pooled experiment requires a new identity")
    cells: list[dict[str, Any]] = []
    signatures = set()
    seen_keys = set()
    for path in discover_design_cells(source_root):
        result = design_cell_result(path, experiment_id=str(cfg["source_experiment_id"]))
        if result["target_id"] != cfg["target"]["target_id"]:
            continue
        config = result["config"]
        if config["target"] != cfg["target"] or result["optimizer_kind"] != "pgd":
            raise ValueError(f"source target or optimizer differs: {path}")
        seed = int(result["base_seed"])
        if seed != int(config["seed"]):
            raise ValueError(f"source seed provenance differs: {path}")
        designs = list(result.get("design_candidates", result["designs"]))
        if (
            int(result["num_designs"]) != count
            or len(designs) != count
            or [d["design_index"] for d in designs] != list(range(count))
        ):
            raise ValueError(f"source candidates are incomplete or not generation ordered: {path}")
        for design in designs:
            key = tuple(design["design_key"])
            if len(key) != 2 or key in seen_keys:
                raise ValueError(f"source design keys missing or repeated: {path}")
            seen_keys.add(key)
            sequence = design["optimizer_sequence"]
            if len(sequence) != result["binder_length"] or "C" in sequence:
                raise ValueError(f"invalid optimizer sequence: {path}")
            references = design["design_references"]
            for reference in references.values():
                if not (path / reference["structure"]).is_file():
                    raise ValueError(f"source reference is missing: {path}")
        signatures.add(
            _digest(
                {
                    "target": config["target"],
                    "optimizer": config["optimizer"],
                    "design": config["design"],
                    "models": config["profile"]["design"],
                    "binder_length": result["binder_length"],
                    "batch_size": result["batch_size"],
                    "objective_id": result["objective_id"],
                    "model_id": result["model_id"],
                    "scientific_validity": result["scientific_validity"],
                }
            )
        )
        cells.append({"path": path, "result": result, "seed": seed, "designs": designs})
    if sorted(cell["seed"] for cell in cells) != sorted(expected):
        raise ValueError("target pool must contain exactly the configured source seeds")
    if len(signatures) != 1:
        raise ValueError("target pool mixes scientific design configurations")
    return sorted(cells, key=lambda cell: cell["seed"])


def pool_provenance(cfg: Mapping[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_id": cfg["experiment_id"],
        "source_experiment_id": cfg["source_experiment_id"],
        "target": cfg["target"],
        "sources": [
            {
                "cell": str(cell["path"]),
                "base_seed": cell["seed"],
                "result_sha256": _digest(cell["result"]),
            }
            for cell in cells
        ],
        "screening": {
            "model": fast_model(cfg["ranking"]),
            "seed": cfg["screening_seed"],
            "sequence_field": "sequence",
            "metric": "ipsae_min",
        },
        "selection_count": cfg["selection_count"],
        "tie_breaker": ["base_seed", "design_index"],
        "ranking": cfg["ranking"],
        "ranking_seed": cfg["ranking_seed"],
    }


def read_mpnn(
    cfg: Mapping[str, Any], cells: list[dict[str, Any]], cell: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(str(cfg["pool_dir"])) / "mpnn" / f"seed{cell['seed']}.json"
    artifact: dict[str, Any] = json.loads(path.read_text())
    recipe = cfg["ranking"]
    expected = {
        "recipe": recipe["name"],
        "model": recipe["self_consistency"]["mpnn"],
        "ranking_seed": int(cfg["ranking_seed"]),
    }
    shard = artifact["shard"]
    if (
        artifact["provenance"] != pool_provenance(cfg, cells)
        or artifact["base_seed"] != cell["seed"]
        or shard["provenance"] != expected
    ):
        raise ValueError(f"MPNN provenance differs: {path}")
    if len(shard["designs"]) != len(cell["designs"]):
        raise ValueError(f"MPNN shard is incomplete: {path}")
    for redesigned, design in zip(shard["designs"], cell["designs"], strict=True):
        if any(redesigned[k] != design[k] for k in ("design_index", "optimizer_sequence")):
            raise ValueError(f"MPNN generation identity differs: {path}")
        if (
            redesigned["source_structure"] != design["selected_reference"]["structure"]
            or redesigned["source_model"] != design["selected_reference"]["model_name"]
        ):
            raise ValueError(f"MPNN reference differs: {path}")
        if (
            len(redesigned["sequence"]) != cell["result"]["binder_length"]
            or "C" in redesigned["sequence"]
        ):
            raise ValueError(f"invalid MPNN sequence: {path}")
    return artifact


def redesign_cell(cfg: Mapping[str, Any], cells: list[dict[str, Any]]) -> None:
    """Use the existing all-candidate MPNN key assignment once per source cell."""
    from binderopt.ranking.mpnn import run_soluble_mpnn_worker

    cell = next(cell for cell in cells if cell["seed"] == int(cfg["cell_seed"]))
    root = Path(str(cfg["pool_dir"]))
    destination = root / "mpnn" / f"seed{cell['seed']}.json"
    if destination.exists():
        raise FileExistsError(f"completed MPNN already exists: {destination}")
    work_root = root / "_launches"
    work_root.mkdir(parents=True, exist_ok=True)
    cache = cfg["profile"]["cache"]
    with tempfile.TemporaryDirectory(prefix=".mpnn-", dir=work_root) as directory:
        work = Path(directory)
        atomic_json(work / "result.json", cell["result"])
        # Workers read references but never modify them.
        (work / "design_references").symlink_to(
            cell["path"] / "design_references", target_is_directory=True
        )
        task_started = time.perf_counter()
        _spawn_worker(
            run_soluble_mpnn_worker,
            cell_dir=work,
            recipe=cfg["ranking"],
            ranking_seed=int(cfg["ranking_seed"]),
            cache_root=Path(cache["root"]),
            offline=bool(cache["offline"]),
        )
        shard = json.loads((work / "ranking" / "soluble_mpnn.json").read_text())
        shard["timing"]["task_wall_time_s"] = round(time.perf_counter() - task_started, 3)
        publish_json(
            destination,
            {
                "provenance": pool_provenance(cfg, cells),
                "base_seed": cell["seed"],
                "shard": shard,
                "config": dict(cfg),
                "git_sha": git_state(Path.cwd()),
            },
        )


def publish_json(path: Path, value: Any) -> None:
    """Publish a complete artifact without replacing even a concurrent writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=path.parent) as directory:
        temporary = Path(directory) / "artifact.json"
        atomic_json(temporary, value)
        os.link(temporary, path)


def screen_candidates(
    designs: list[dict[str, Any]], predict_score: Callable[[str], float]
) -> list[dict[str, Any]]:
    """Score every redesigned sequence; judge failures invalidate the entire shard."""
    records = []
    for design in designs:
        started = time.perf_counter()
        score = float(predict_score(str(design["sequence"])))
        if not math.isfinite(score):
            raise ValueError("screening requires finite protein IPSAE_min")
        records.append(
            {
                "design_index": design["design_index"],
                "design_key": design["design_key"],
                "optimizer_sequence": design["optimizer_sequence"],
                "sequence": design["sequence"],
                "ipsae_min": score,
                "prediction_wall_time_s": time.perf_counter() - started,
            }
        )
    return records


def screen_cell(cfg: Mapping[str, Any], cells: list[dict[str, Any]]) -> None:
    """Run one fixed fast-model seed on all candidates in one source cell."""
    provenance = pool_provenance(cfg, cells)
    cell = next(cell for cell in cells if cell["seed"] == int(cfg["cell_seed"]))
    destination = Path(str(cfg["pool_dir"])) / "screening" / f"seed{cell['seed']}.json"
    if destination.exists():
        raise FileExistsError(f"completed screening already exists: {destination}")
    mpnn = read_mpnn(cfg, cells, cell)
    from binderopt.assets.cache import configure_process_caches

    cache = cfg["profile"]["cache"]
    configure_process_caches(Path(str(cache["root"])), offline=bool(cache["offline"]))
    import jax
    import jax.numpy as jnp
    from hydra.utils import get_method
    from mosaic.common import TOKENS

    from binderopt.ranking.models import protein_ipsae_min
    from binderopt.structure.features import build_esm_ranking_features
    from binderopt.structure.targets import target_from_config

    started = time.perf_counter()
    spec = fast_model(cfg["ranking"])
    model = get_method(str(spec["factory"]))()
    target = target_from_config(cell["result"]["config"]["target"])
    key = jax.random.key(int(cfg["screening_seed"]))

    def predict(sequence: str) -> float:
        prepared = build_esm_ranking_features(
            model, sequence, target, supports_msa=bool(spec["use_target_msa"])
        )
        one_hot = jax.nn.one_hot(jnp.asarray([TOKENS.index(a) for a in sequence]), len(TOKENS))
        prediction = model.predict(
            PSSM=one_hot,
            features=prepared.features,
            writer=prepared.writer,
            recycling_steps=int(spec["recycling_steps"]),
            sampling_steps=int(spec["sampling_steps"]),
            key=key,
        )
        return protein_ipsae_min(one_hot, prediction.model_output, prepared.masks, key=key)

    designs = [
        dict(d) | {"design_key": original["design_key"]}
        for d, original in zip(mpnn["shard"]["designs"], cell["designs"], strict=True)
    ]
    records = screen_candidates(designs, predict)
    publish_json(
        destination,
        {
            "provenance": provenance,
            "base_seed": cell["seed"],
            "mpnn_sha256": _digest(mpnn),
            "designs": records,
            "git_sha": git_state(Path.cwd()),
            "config": dict(cfg),
            "completed_at_utc": utc_now(),
            "wall_time_s": time.perf_counter() - started,
        },
    )


def select_pool(cfg: Mapping[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Select globally, with no quota or threshold specific to an individual shard."""
    provenance = pool_provenance(cfg, cells)
    records = []
    for cell in cells:
        path = Path(str(cfg["pool_dir"])) / "screening" / f"seed{cell['seed']}.json"
        shard = json.loads(path.read_text())
        mpnn = read_mpnn(cfg, cells, cell)
        if shard["mpnn_sha256"] != _digest(mpnn):
            raise ValueError(f"screening MPNN provenance differs: {path}")
        if shard["provenance"] != provenance or shard["base_seed"] != cell["seed"]:
            raise ValueError(f"screening provenance differs: {path}")
        if len(shard["designs"]) != len(cell["designs"]):
            raise ValueError(f"screening shard is incomplete: {path}")
        for scored, design, redesigned in zip(
            shard["designs"], cell["designs"], mpnn["shard"]["designs"], strict=True
        ):
            if scored["sequence"] != redesigned["sequence"]:
                raise ValueError(f"screened sequence differs from MPNN: {path}")
            if any(
                scored[k] != design[k] for k in ("design_index", "design_key", "optimizer_sequence")
            ):
                raise ValueError(f"screening design identity differs: {path}")
            if not math.isfinite(float(scored["ipsae_min"])):
                raise ValueError(f"screening score is non-finite: {path}")
            records.append(dict(scored) | {"base_seed": cell["seed"], "mpnn_sha256": _digest(mpnn)})
    records.sort(key=lambda d: (-d["ipsae_min"], d["base_seed"], d["design_index"]))
    for rank, record in enumerate(records, 1):
        record.update(selection_rank=rank, selected=rank <= int(cfg["selection_count"]))
    return {
        "provenance": provenance,
        "candidate_count": len(records),
        "selected_count": int(cfg["selection_count"]),
        "candidates": records,
    }


def _spawn_worker(worker: Callable[..., Any], **kwargs: Any) -> None:
    process = multiprocessing.get_context("spawn").Process(target=worker, kwargs=kwargs)
    process.start()
    process.join()
    exit_code = process.exitcode
    process.close()
    if exit_code != 0:
        raise process_failure(worker.__name__, exit_code)


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def rank_survivors(
    cfg: Mapping[str, Any], cells: list[dict[str, Any]], manifest: Mapping[str, Any]
) -> None:
    """Rank selected source indices without renumbering or modifying source cells."""
    from binderopt.ranking.models import run_ranking_model_worker

    cell = next(cell for cell in cells if cell["seed"] == int(cfg["cell_seed"]))
    local = [d for d in manifest["candidates"] if d["base_seed"] == cell["seed"]]
    indices = sorted(d["design_index"] for d in local if d["selected"])
    if not indices:
        print(f"seed {cell['seed']}: no pooled survivors; no ranking required")
        return
    root = Path(str(cfg["pool_dir"]))
    destination = root / f"{cfg['target']['target_id']}_pgd_seed{cell['seed']}"
    if destination.exists():
        raise FileExistsError(f"pooled ranked cell already exists: {destination}")
    work_root = root / "_launches"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rank-", dir=work_root) as directory:
        work = Path(directory)
        result = copy.deepcopy(cell["result"])
        # Restore all candidates in generation order before selecting survivors.
        result["designs"] = result.pop("design_candidates", cell["designs"])
        for field in ("ranking_preselection", "ranking_timing", "summary", "num_ranked_designs"):
            result.pop(field, None)
        result.update(experiment_id=cfg["experiment_id"], ranking_model_id=None)
        result["source_generation"] = {
            "cell": str(cell["path"]),
            "result_sha256": _digest(cell["result"]),
            "experiment_id": cfg["source_experiment_id"],
        }
        result["source_config"] = copy.deepcopy(cell["result"]["config"])
        result["pool_config"] = dict(cfg)
        result["pool_git_sha"] = git_state(Path.cwd())
        result["config"].update(
            experiment_id=cfg["experiment_id"],
            cell_dir=str(destination),
            output_root=cfg["output_root"],
            ranking_preselection=None,
        )
        atomic_json(work / "result.json", result)
        shutil.copytree(
            cell["path"] / "design_references",
            work / "design_references",
            copy_function=_link_or_copy,
        )
        for name in (
            "pssms.npz",
            "optimizer_designs.fasta",
            "opt_trajectory.json",
            "opt_trajectory_pssms.npz",
        ):
            source = cell["path"] / name
            if source.exists():
                _link_or_copy(str(source), str(work / name))
        recipe = cfg["ranking"]
        cache = cfg["profile"]["cache"]
        started = time.perf_counter()
        started_at = utc_now()
        mpnn = read_mpnn(cfg, cells, cell)
        if any(d["mpnn_sha256"] != _digest(mpnn) for d in local):
            raise ValueError("selected MPNN artifact has changed")
        shard = copy.deepcopy(mpnn["shard"])
        shard["designs"] = [d for d in shard["designs"] if d["design_index"] in indices]
        atomic_json(work / "ranking" / "soluble_mpnn.json", shard)
        for spec in recipe["models"]:
            _spawn_worker(
                run_ranking_model_worker,
                cell_dir=work,
                recipe_name=recipe["name"],
                model_spec=spec,
                seeds=requested_seeds(recipe, int(cfg["ranking_seed"])),
                base_seed=int(cfg["ranking_seed"]),
                cache_root=Path(cache["root"]),
                offline=bool(cache["offline"]),
                self_consistency_spec=recipe["self_consistency"],
                gpu_count=int(cfg["profile"]["resources"]["gpus_per_task"]),
            )
        preselection = {
            "metric": "ipsae_min",
            "scope": "target_pool",
            "pool_manifest": "../selection.json",
            "candidate_count": len(local),
            "selected_count": len(indices),
            "candidates": local,
        }
        atomic_json(work / "ranking" / "preselection.json", preselection)
        aggregate_cell_ranking(
            work,
            recipe,
            base_seed=int(cfg["ranking_seed"]),
            design_indices=indices,
            preselection=preselection,
            ranking_started_at_utc=started_at,
            ranking_wall_time_s=time.perf_counter() - started,
        )
        # Publish the whole finished cell at once; interrupted work is recomputed.
        work.rename(destination)


def verify_ranked_cell(cfg: Mapping[str, Any], manifest: Mapping[str, Any], seed: int) -> int:
    """Validate a final cell against its frozen survivor identities before reuse."""
    root = Path(str(cfg["pool_dir"]))
    expected = {
        d["design_index"]: d
        for d in manifest["candidates"]
        if d["base_seed"] == seed and d["selected"]
    }
    path = root / f"{cfg['target']['target_id']}_pgd_seed{seed}"
    if not expected:
        if path.exists():
            raise ValueError(f"unexpected ranked cell without survivors: {path}")
        return 0
    result = design_cell_result(
        path,
        experiment_id=str(cfg["experiment_id"]),
        target_id=str(cfg["target"]["target_id"]),
        optimizer="pgd",
    )
    source = next(s for s in manifest["provenance"]["sources"] if s["base_seed"] == seed)
    expected_generation = {
        "cell": source["cell"],
        "result_sha256": source["result_sha256"],
        "experiment_id": cfg["source_experiment_id"],
    }
    if (
        result["base_seed"] != seed
        or result.get("source_generation") != expected_generation
        or result.get("pool_config", {}).get("ranking") != cfg["ranking"]
        or result.get("pool_config", {}).get("ranking_seed") != cfg["ranking_seed"]
    ):
        raise ValueError(f"ranked cell provenance differs: {path}")
    designs = result["designs"]
    if (
        result.get("ranking_model_id") != cfg["ranking"]["name"]
        or len(designs) != len(expected)
        or {d["design_index"] for d in designs} != set(expected)
    ):
        raise ValueError(f"ranked survivor set differs: {path}")
    for design in designs:
        selected = expected[design["design_index"]]
        if any(design[k] != selected[k] for k in ("design_key", "sequence")):
            raise ValueError(f"ranked survivor identity differs: {path}")
    return len(designs)


def verify_ranked_pool(cfg: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Require the entire selected pool, including shards with no survivors."""
    observed = sum(verify_ranked_cell(cfg, manifest, int(seed)) for seed in cfg["source_seeds"])
    if observed != int(cfg["selection_count"]):
        raise ValueError("ranked pool is incomplete")
    print(f"complete pooled ranking: {observed} survivors")
