"""Pooled selection contracts, exercised without model loads or accelerators."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from binderopt.outputs import atomic_json
from binderopt.ranking import screening as pool


@pytest.fixture
def experiment(tmp_path):
    recipe = yaml.safe_load(Path("configs/ranking/full.yaml").read_text())
    target = {"target_id": "toy", "chains": [{"sequence": "AAA"}]}
    cfg = {
        "target": target,
        "source_dir": str(tmp_path / "source"),
        "source_experiment_id": "old",
        "experiment_id": "pooled",
        "output_root": str(tmp_path),
        "pool_dir": str(tmp_path / "pooled" / "toy"),
        "source_seeds": [1, 2],
        "candidates_per_cell": 2,
        "selection_count": 2,
        "cell_seed": 1,
        "screening_seed": 1,
        "ranking_seed": 1,
        "ranking": recipe,
        "profile": {
            "cache": {"root": str(tmp_path), "offline": True},
            "resources": {"gpus_per_task": 1},
        },
    }
    for seed in cfg["source_seeds"]:
        cell = Path(cfg["source_dir"]) / f"toy_pgd_seed{seed}"
        designs = []
        for index in range(2):
            reference = f"design_references/design_{index:03d}/reference.pdb"
            (cell / reference).parent.mkdir(parents=True)
            (cell / reference).write_text("reference")
            record = {"structure": reference, "model_name": "reference"}
            designs.append(
                {
                    "design_index": index,
                    "design_key": [seed, index],
                    "optimizer_sequence": "AAA",
                    "selected_reference": record,
                    "design_references": {"reference": record},
                    "selected_step": 74,
                }
            )
        config = {
            "target": target,
            "optimizer": {"kind": "pgd"},
            "seed": seed,
            "design": {"n_steps": 75},
            "profile": {"design": {}},
            "experiment_id": "old",
        }
        result = {
            "experiment_id": "old",
            "target_id": "toy",
            "optimizer_kind": "pgd",
            "base_seed": seed,
            "num_designs": 2,
            "num_designs_requested": 2,
            "binder_length": 3,
            "batch_size": 1,
            "objective_id": "objective",
            "model_id": "ensemble",
            "scientific_validity": "reference",
            "config": config,
            "designs": designs,
        }
        if seed == 1:  # Old preselected ranking still retains all optimizer candidates.
            result.update(design_candidates=designs, designs=designs[:1])
        atomic_json(cell / "result.json", result)
    cells = pool.audit_pool(cfg)
    provenance = pool.pool_provenance(cfg, cells)
    for cell in cells:
        records = [
            {
                "design_index": d["design_index"],
                "optimizer_sequence": "AAA",
                "sequence": ["ARN", "RAN"][d["design_index"]],
                "source_structure": d["selected_reference"]["structure"],
                "source_model": "reference",
                "esmc_perplexity": 2.0,
            }
            for d in cell["designs"]
        ]
        shard = {
            "provenance": {
                "recipe": "full",
                "model": recipe["self_consistency"]["mpnn"],
                "ranking_seed": 1,
            },
            "timing": {"wall_time_s": 1.0},
            "designs": records,
        }
        artifact = {"provenance": provenance, "base_seed": cell["seed"], "shard": shard}
        atomic_json(Path(cfg["pool_dir"]) / "mpnn" / f"seed{cell['seed']}.json", artifact)
        scores = [0.9, 0.8] if cell["seed"] == 1 else [0.1, 0.2]
        screened = [
            dict(d) | {"sequence": records[i]["sequence"], "ipsae_min": scores[i]}
            for i, d in enumerate(cell["designs"])
        ]
        atomic_json(
            Path(cfg["pool_dir"]) / "screening" / f"seed{cell['seed']}.json",
            {
                "provenance": provenance,
                "base_seed": cell["seed"],
                "mpnn_sha256": pool._digest(artifact),
                "designs": screened,
            },
        )
    return cfg, cells


def test_global_selection_has_no_per_shard_quota(experiment):
    cfg, cells = experiment
    selected = [d for d in pool.select_pool(cfg, cells)["candidates"] if d["selected"]]
    assert [(d["base_seed"], d["design_index"]) for d in selected] == [(1, 0), (1, 1)]
    assert [d["sequence"] for d in selected] == ["ARN", "RAN"]


def test_ties_use_seed_then_generation_index(experiment):
    cfg, cells = experiment
    for seed in (1, 2):
        path = Path(cfg["pool_dir"]) / "screening" / f"seed{seed}.json"
        shard = json.loads(path.read_text())
        for d in shard["designs"]:
            d["ipsae_min"] = 0.5
        atomic_json(path, shard)
    assert [
        (d["base_seed"], d["design_index"]) for d in pool.select_pool(cfg, cells)["candidates"]
    ] == [(1, 0), (1, 1), (2, 0), (2, 1)]


@pytest.mark.parametrize("mutation", ["missing_seed", "mixed_config", "duplicate_key"])
def test_audit_rejects_invalid_pool(experiment, mutation):
    cfg, _ = experiment
    path = Path(cfg["source_dir"]) / "toy_pgd_seed2" / "result.json"
    result = json.loads(path.read_text())
    if mutation == "missing_seed":
        path.unlink()
    else:
        if mutation == "mixed_config":
            result["config"]["design"]["n_steps"] = 150
        else:
            result["designs"][0]["design_key"] = [1, 0]
        atomic_json(path, result)
    with pytest.raises(ValueError):
        pool.audit_pool(cfg)


@pytest.mark.parametrize("mutation", ["sequence", "key", "nonfinite", "missing", "provenance"])
def test_selection_rejects_invalid_screen(experiment, mutation):
    cfg, cells = experiment
    path = Path(cfg["pool_dir"]) / "screening" / "seed1.json"
    shard = json.loads(path.read_text())
    if mutation == "sequence":
        shard["designs"][0]["sequence"] = "AAA"
    elif mutation == "key":
        shard["designs"][0]["design_key"] = [99, 99]
    elif mutation == "nonfinite":
        shard["designs"][0]["ipsae_min"] = float("nan")
    elif mutation == "missing":
        shard["designs"].pop()
    else:
        shard["provenance"]["screening"]["seed"] = 2
    atomic_json(path, shard)
    with pytest.raises(ValueError):
        pool.select_pool(cfg, cells)


def test_screen_uses_redesigned_sequence_and_rejects_nan():
    designs = [
        {"design_index": 0, "design_key": [1, 2], "optimizer_sequence": "AAA", "sequence": "ARN"}
    ]
    seen = []
    records = pool.screen_candidates(designs, lambda sequence: seen.append(sequence) or 0.8)
    assert seen == ["ARN"]
    assert records[0]["ipsae_min"] == 0.8
    with pytest.raises(ValueError, match="finite"):
        pool.screen_candidates(designs, lambda _: float("nan"))


def test_final_ranking_reuses_mpnn_and_publishes_atomically(experiment, monkeypatch):
    cfg, cells = experiment
    manifest = pool.select_pool(cfg, cells)
    # Select a noncontiguous survivor via the frozen manifest.
    manifest["candidates"][0]["selected"] = False
    manifest["candidates"][2]["selected"] = True
    source_texts = [((c["path"] / "result.json").read_text()) for c in cells]
    calls = []

    def fake_worker(worker, **kwargs):
        assert worker.__name__ == "run_ranking_model_worker"  # No second MPNN draw.
        work = kwargs["cell_dir"]
        mpnn = json.loads((work / "ranking" / "soluble_mpnn.json").read_text())
        assert [(d["design_index"], d["sequence"]) for d in mpnn["designs"]] == [(1, "RAN")]
        assert not (Path(cfg["pool_dir"]) / "toy_pgd_seed1").exists()
        spec = kwargs["model_spec"]
        name = spec["name"]
        records = [
            {
                "design_index": 1,
                "sequence": "RAN",
                "winning_seed": 5,
                "seed_scores": [
                    {
                        "seed": seed,
                        "ipsae_min": seed / 10,
                        "dockq": seed / 20,
                        "prediction_wall_time_s": 1,
                    }
                    for seed in kwargs["seeds"]
                ],
            }
        ]
        atomic_json(
            work / "ranking" / f"{name}.json",
            {
                "model_name": name,
                "requested_seeds": kwargs["seeds"],
                "designs": records,
                "provenance": {
                    "recipe": "full",
                    "model": spec,
                    "ranking_seed": 1,
                    "self_consistency": cfg["ranking"]["self_consistency"],
                },
            },
        )
        calls.append(name)

    monkeypatch.setattr(pool, "_spawn_worker", fake_worker)
    pool.rank_survivors(cfg, cells, manifest)
    assert len(calls) == 3
    result = json.loads((Path(cfg["pool_dir"]) / "toy_pgd_seed1" / "result.json").read_text())
    assert result["experiment_id"] == "pooled"
    assert result["designs"][0]["design_key"] == [1, 1]
    assert result["designs"][0]["sequence"] == "RAN"
    assert result["designs"][0]["ranking_score"] == pytest.approx(0.5)
    assert len(result["design_candidates"]) == 2
    assert pool.verify_ranked_cell(cfg, manifest, 1) == 1
    assert source_texts == [(c["path"] / "result.json").read_text() for c in cells]
    with pytest.raises(FileExistsError):
        pool.rank_survivors(cfg, cells, manifest)


def test_failure_publishes_no_ranked_cell(experiment, monkeypatch):
    cfg, cells = experiment

    def fail(*args, **kwargs):
        raise RuntimeError("judge failed")

    monkeypatch.setattr(pool, "_spawn_worker", fail)
    with pytest.raises(RuntimeError, match="judge failed"):
        pool.rank_survivors(cfg, cells, pool.select_pool(cfg, cells))
    assert not (Path(cfg["pool_dir"]) / "toy_pgd_seed1").exists()
    assert not list((Path(cfg["pool_dir"]) / "_launches").glob(".rank-*"))


def test_zero_survivor_shard_and_incomplete_final_pool(experiment):
    cfg, cells = experiment
    manifest = pool.select_pool(cfg, cells)
    cfg["cell_seed"] = 2
    pool.rank_survivors(cfg, cells, manifest)
    assert not (Path(cfg["pool_dir"]) / "toy_pgd_seed2").exists()
    with pytest.raises(ValueError):
        pool.verify_ranked_pool(cfg, manifest)


def test_publishing_never_overwrites(tmp_path):
    path = tmp_path / "selection.json"
    pool.publish_json(path, {"selected": 1})
    with pytest.raises(FileExistsError):
        pool.publish_json(path, {"selected": 2})
    assert json.loads(path.read_text()) == {"selected": 1}


def test_changed_source_or_mpnn_invalidates_scores(experiment):
    cfg, cells = experiment
    changed = copy.deepcopy(cells)
    changed[0]["result"]["git_sha"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        pool.select_pool(cfg, changed)
    path = Path(cfg["pool_dir"]) / "mpnn" / "seed1.json"
    artifact = json.loads(path.read_text())
    artifact["shard"]["designs"][0]["sequence"] = "AAA"
    atomic_json(path, artifact)
    with pytest.raises(ValueError, match="MPNN provenance"):
        pool.select_pool(cfg, cells)


def test_mpnn_stage_processes_all_candidates_and_refuses_replacement(experiment, monkeypatch):
    cfg, cells = experiment
    path = Path(cfg["pool_dir"]) / "mpnn" / "seed1.json"
    original = json.loads(path.read_text())
    path.unlink()

    def fake_worker(worker, **kwargs):
        assert worker.__name__ == "run_soluble_mpnn_worker"
        assert "design_indices" not in kwargs
        work = kwargs["cell_dir"]
        source = json.loads((work / "result.json").read_text())
        assert len(source["design_candidates"]) == 2
        assert len(source["designs"]) == 1  # The worker receives only the selected candidates.
        reference = cells[0]["designs"][0]["selected_reference"]["structure"]
        assert (work / reference).read_text() == "reference"
        atomic_json(work / "ranking" / "soluble_mpnn.json", original["shard"])

    monkeypatch.setattr(pool, "_spawn_worker", fake_worker)
    pool.redesign_cell(cfg, cells)
    artifact = pool.read_mpnn(cfg, cells, cells[0])
    assert len(artifact["shard"]["designs"]) == 2
    with pytest.raises(FileExistsError):
        pool.redesign_cell(cfg, cells)


def test_staging_is_excluded_from_discovery(experiment, monkeypatch):
    cfg, cells = experiment

    def inspect_then_fail(worker, **kwargs):
        with pytest.raises(ValueError, match="no completed design cells"):
            pool.discover_design_cells(Path(cfg["pool_dir"]))
        raise RuntimeError("stop before publication")

    monkeypatch.setattr(pool, "_spawn_worker", inspect_then_fail)
    with pytest.raises(RuntimeError):
        pool.rank_survivors(cfg, cells, pool.select_pool(cfg, cells))


@pytest.mark.parametrize("mutation", ["sequence", "source", "ranking_seed"])
def test_reuse_rejects_completed_cell_with_different_identity(experiment, mutation):
    cfg, cells = experiment
    manifest = pool.select_pool(cfg, cells)
    cell = cells[0]
    result = copy.deepcopy(cell["result"])
    result.update(experiment_id=cfg["experiment_id"], ranking_model_id="full")
    result["source_generation"] = {
        "cell": str(cell["path"]),
        "result_sha256": pool._digest(cell["result"]),
        "experiment_id": cfg["source_experiment_id"],
    }
    result["pool_config"] = copy.deepcopy(cfg)
    result["designs"] = [dict(d) for d in manifest["candidates"] if d["base_seed"] == 1]
    path = Path(cfg["pool_dir"]) / "toy_pgd_seed1" / "result.json"
    atomic_json(path, result)
    assert pool.verify_ranked_cell(cfg, manifest, 1) == 2
    if mutation == "sequence":
        result["designs"][0]["sequence"] = "AAA"
    elif mutation == "source":
        result["source_generation"]["result_sha256"] = "changed"
    else:
        result["pool_config"]["ranking_seed"] = 99
    atomic_json(path, result)
    with pytest.raises(ValueError):
        pool.verify_ranked_cell(cfg, manifest, 1)
