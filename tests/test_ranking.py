from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binderopt.composite import main as composite_main
from binderopt.ranking.cells import (
    aggregate_cell_ranking,
    design_cell_result,
    discover_design_cells,
    select_designs_for_ranking,
)
from binderopt.ranking.composite import compute_composite_ranking
from binderopt.ranking.models import Prediction, protein_ipsae_min, run_model
from binderopt.structure.features import EntityMasks

SELF_CONSISTENCY = {
    "required_seed_count": 2,
    "validity_level": "publication",
    "mpnn": {
        "factory": "fixed",
        "checkpoint": "soluble",
        "temperature": 0.1,
        "excluded_amino_acids": ["C"],
    },
    "reference_selection": "max_protein_iptm_yaml_order_tiebreak",
    "dockq": {
        "implementation": "DockQ",
        "residue_mapping": "positional_numbering",
        "target_grouping": "merged",
        "hetero_atoms": "excluded",
    },
    "target_aligned_rmsd": {"atom_names": ["N", "CA", "C", "O"]},
}


def _metric(value: float) -> dict:
    return {
        "dockq": value,
        "fnat": value,
        "fnonnat": 0.0,
        "contact_f1": value,
        "interface_rmsd": 1.0,
        "dockq_ligand_rmsd": 2.0,
        "dockq_receptor": "target",
        "dockq_ligand": "binder",
        "target_aligned_binder_backbone_rmsd": 2.0,
        "interface_contact_counts": {
            "reference": 2,
            "prediction": 2,
            "recovered": 1,
            "non_native": 0,
        },
        "clashes": 0,
        "target_chain_mapping": [{"reference": "B", "prediction": "B"}],
    }


def _cell(
    path: Path,
    *,
    optimizer: str = "pgd",
    design_key: list[int] | None = None,
    design_count: int = 1,
) -> None:
    designs = []
    for design_index in range(design_count):
        reference = path / f"design_references/design_{design_index:03d}/design_a.pdb"
        reference.parent.mkdir(parents=True)
        reference.write_text("MODEL\n")
        designs.append(
            {
                "design_index": design_index,
                "optimizer_sequence": "ARN",
                "optimizer_esmc_perplexity": 2.0,
                "tail_iptm": 0.5 + design_index / 10,
                "design_key": design_key or [1, 2],
                "selected_reference": {
                    "model_name": "design_a",
                    "checkpoint": "a",
                    "protein_iptm": 0.7,
                    "structure": f"design_references/design_{design_index:03d}/design_a.pdb",
                },
            }
        )
    result = {
        "experiment_id": "test",
        "target_id": "target",
        "objective_id": "esm_binder",
        "model_id": "design_models",
        "ranking_model_id": None,
        "optimizer_kind": optimizer,
        "binder_length": 3,
        "num_designs_requested": design_count,
        "num_designs": design_count,
        "config": {
            "target": {
                "target_id": "target",
                "name": "Target",
                "default_binder_length": 3,
                "chains": [{"entity_id": "target", "sequence": "AAA", "use_msa": False}],
                "ligands": [],
            }
        },
        "designs": designs,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps(result))
    (path / "optimizer_designs.fasta").write_text(">design_000\nARN\n")


def _write_shards(
    cell: Path,
    recipe: dict,
    *,
    base_seed: int,
    scores: dict[str, list[tuple[float, float]]],
    design_indices: list[int] | None = None,
) -> None:
    ranking = cell / "ranking"
    ranking.mkdir(exist_ok=True)
    source = json.loads((cell / "result.json").read_text())["designs"]
    requested = None if design_indices is None else set(design_indices)
    selected = source if requested is None else [
        item for item in source if int(item["design_index"]) in requested
    ]
    mpnn = {
        "model_name": "soluble_mpnn",
        "provenance": {
            "recipe": recipe["name"],
            "model": recipe["self_consistency"]["mpnn"],
            "ranking_seed": base_seed,
        },
        "designs": [
            {
                "design_index": design["design_index"],
                "optimizer_sequence": design["optimizer_sequence"],
                "sequence": "DDD",
                "source_structure": design["selected_reference"]["structure"],
                "source_model": "design_a",
                "esmc_log_perplexity": 1.0,
                "esmc_perplexity": float(np.e),
            }
            for design in selected
        ],
    }
    (ranking / "soluble_mpnn.json").write_text(json.dumps(mpnn))
    seeds = [base_seed + offset for offset in recipe["seed_offsets"]]
    for spec in recipe["models"]:
        name = spec["name"]
        seed_scores = [
            {"seed": seed, "ipsae_min": ipsae, "prediction_wall_time_s": 0.1, **_metric(dockq)}
            for seed, (ipsae, dockq) in zip(seeds, scores[name], strict=True)
        ]
        winner = min(seed_scores, key=lambda item: (-item["ipsae_min"], item["seed"]))
        shard = {
            "model_name": name,
            "requested_seeds": seeds,
            "provenance": {
                "recipe": recipe["name"],
                "model": spec,
                "ranking_seed": base_seed,
                "self_consistency": recipe["self_consistency"],
            },
            "ranking_model_timing": {"model_load_wall_time_s": 1.0},
            "designs": [
                {
                    "design_index": design["design_index"],
                    "sequence": "DDD",
                    "seed_scores": seed_scores,
                    "best_ipsae_min": winner["ipsae_min"],
                    "winning_dockq": winner["dockq"],
                    "winning_seed": winner["seed"],
                    "structure": f"structures/design_{design['design_index']:03d}/{name}.pdb",
                }
                for design in selected
            ],
        }
        (ranking / f"{name}.json").write_text(json.dumps(shard))


def test_model_writes_only_ipsae_winner_after_all_seeds_complete(tmp_path) -> None:
    design = {"design_index": 0, "sequence": "ARN"}

    def interrupted(_design):
        def predict(seed: int) -> Prediction:
            if seed == 12:
                raise RuntimeError("interrupted")
            return Prediction(0.5, f"MODEL {seed}\n", _metric(0.4))

        return predict

    with pytest.raises(RuntimeError, match="interrupted"):
        run_model(
            tmp_path,
            model_name="esmfold2_fast",
            designs=[design],
            seeds=[11, 12],
            provenance={},
            make_predictor=interrupted,
        )
    assert not (tmp_path / "ranking/esmfold2_fast.json").exists()
    assert not (tmp_path / "structures/design_000/esmfold2_fast.pdb").exists()

    complete = run_model(
        tmp_path,
        model_name="esmfold2_fast",
        designs=[design],
        seeds=[11, 12],
        provenance={},
        make_predictor=lambda _design: (
            lambda seed: Prediction(
                0.5 if seed == 11 else 0.6,
                f"MODEL {seed}\n",
                _metric(0.2 if seed == 11 else 0.3),
            )
        ),
    )
    record = complete["designs"][0]
    assert record["winning_seed"] == 12
    assert record["winning_dockq"] == 0.3
    assert (tmp_path / "structures/design_000/esmfold2_fast.pdb").read_text() == "MODEL 12\n"


def test_discovery_requires_one_complete_experiment(tmp_path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    _cell(first)
    _cell(second, optimizer="biohub")
    assert discover_design_cells(tmp_path) == [first, second]
    assert design_cell_result(first, experiment_id="test")["target_id"] == "target"
    result = json.loads((second / "result.json").read_text())
    result["experiment_id"] = "other"
    (second / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="mixes experiments"):
        discover_design_cells(tmp_path)
    assert discover_design_cells(tmp_path, allow_mixed_experiments=True) == [first, second]


def test_tail_iptm_preselection_is_stable_and_retains_candidates() -> None:
    designs = [
        {"design_index": 0, "tail_iptm": 0.8},
        {"design_index": 1, "tail_iptm": 0.8},
        {"design_index": 2, "tail_iptm": 0.6},
        {"design_index": 3, "tail_iptm": 0.9},
    ]
    policy = {
        "metric": "tail_iptm",
        "tie_breaker": "design_index",
        "expected_candidates": 4,
        "count": 2,
    }
    selected, manifest = select_designs_for_ranking({"designs": designs}, policy)
    assert [item["design_index"] for item in selected] == [0, 3]
    assert manifest is not None
    assert [item["design_index"] for item in manifest["candidates"]] == [3, 0, 1, 2]
    assert [item["selected"] for item in manifest["candidates"]] == [
        True,
        True,
        False,
        False,
    ]

    rerun, _ = select_designs_for_ranking(
        {"designs": selected, "design_candidates": designs}, policy
    )
    assert [item["design_index"] for item in rerun] == [0, 3]


def test_preselected_ranking_preserves_all_optimizer_candidates(tmp_path) -> None:
    recipe = {
        "name": "full",
        "scientific_validity": "publication",
        "seed_offsets": [0, 1],
        "self_consistency": SELF_CONSISTENCY,
        "models": [{"name": "a"}],
    }
    _cell(tmp_path, design_count=3)
    selected, manifest = select_designs_for_ranking(
        design_cell_result(tmp_path),
        {
            "metric": "tail_iptm",
            "tie_breaker": "design_index",
            "expected_candidates": 3,
            "count": 2,
        },
    )
    indices = [item["design_index"] for item in selected]
    _write_shards(
        tmp_path,
        recipe,
        base_seed=7,
        scores={"a": [(0.8, 0.4), (0.7, 0.3)]},
        design_indices=indices,
    )
    ranked = aggregate_cell_ranking(
        tmp_path, recipe, base_seed=7, design_indices=indices, preselection=manifest
    )
    assert ranked["num_designs"] == 3
    assert ranked["num_ranked_designs"] == 2
    assert [item["design_index"] for item in ranked["design_candidates"]] == [0, 1, 2]
    assert {item["design_index"] for item in ranked["designs"]} == {1, 2}


def test_cell_ranking_is_ipsae_only_and_composite_is_post_hoc(tmp_path) -> None:
    recipe = {
        "name": "full",
        "scientific_validity": "publication",
        "seed_offsets": [0, 1],
        "self_consistency": SELF_CONSISTENCY,
        "models": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
    }
    pgd, biohub = tmp_path / "target_pgd", tmp_path / "target_biohub"
    _cell(pgd, optimizer="pgd", design_key=[9, 9])
    _cell(biohub, optimizer="biohub", design_key=[9, 9])
    _write_shards(
        pgd,
        recipe,
        base_seed=7,
        scores={name: [(0.8, 0.1), (0.7, 0.99)] for name in ("a", "b", "c")},
    )
    _write_shards(
        biohub,
        recipe,
        base_seed=7,
        scores={name: [(0.2, 0.9), (0.1, 0.0)] for name in ("a", "b", "c")},
    )
    aggregate_cell_ranking(pgd, recipe, base_seed=7)
    aggregate_cell_ranking(biohub, recipe, base_seed=7)
    pgd_result = json.loads((pgd / "result.json").read_text())
    design = pgd_result["designs"][0]
    assert design["optimizer_sequence"] == "ARN"
    assert design["sequence"] == "DDD"
    assert design["ranking"]["models"]["a"]["winning_dockq"] == 0.1
    assert design["ipsae_ranking_score"] == pytest.approx(0.8)
    assert design["ranking_score"] == pytest.approx(0.8)
    assert design["rank_loss"] == pytest.approx(-0.8)
    assert "ranking_normalization" not in design
    assert "final_score" not in design
    assert (pgd / "designs.fasta").read_text().endswith("DDD\n")

    manifest = compute_composite_ranking(tmp_path, ipsae_weight=4, dockq_weight=1)
    ranked = manifest["targets"]["target"]["designs"]
    assert ranked[0]["optimizer_kind"] == "pgd"
    assert ranked[0]["composite_score"] == pytest.approx((4 * 3 - 3) / 15)
    assert ranked[1]["composite_score"] == pytest.approx((-4 * 3 + 3) / 15)
    assert manifest["targets"]["target"]["statistics"]["a.ipsae"]["count"] == 2

    composite_main([str(tmp_path), "--ipsae-weight", "4", "--dockq-weight", "1"])
    written = json.loads((tmp_path / "analysis/composite_ranking.json").read_text())
    assert written["targets"]["target"]["designs"][0]["optimizer_kind"] == "pgd"


def test_cell_ranking_fails_on_nonfinite_measurements(tmp_path) -> None:
    recipe = {
        "name": "full",
        "scientific_validity": "publication",
        "seed_offsets": [0, 1],
        "self_consistency": SELF_CONSISTENCY,
        "models": [{"name": "a"}],
    }
    _cell(tmp_path)
    _write_shards(
        tmp_path,
        recipe,
        base_seed=7,
        scores={"a": [(np.nan, 0.5), (0.1, 0.5)]},
    )
    with pytest.raises(ValueError, match="finite"):
        aggregate_cell_ranking(tmp_path, recipe, base_seed=7)


@pytest.mark.parametrize("module", [
    "binderopt.design", "binderopt.rank", "binderopt.pool",
    "binderopt.composite", "binderopt.cache", "binderopt.msa",
    "binderopt.optimization.cell", "binderopt.ranking.cells",
    "binderopt.ranking.models", "binderopt.ranking.mpnn",
    "binderopt.structure.protenix",
])
def test_entry_points_and_workers_import_without_jax_or_mosaic(module: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {module}; "
            "assert 'jax' not in sys.modules; assert 'mosaic' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@dataclass(frozen=True)
class FakeOutput:
    full_sequence: jax.Array
    pae_logits: jax.Array
    pae_bins: jax.Array


def test_mosaic_ipsae_uses_combined_protein_target_and_excludes_ligand() -> None:
    masks = EntityMasks(
        binder=np.array([1, 1, 0, 0, 0, 0], dtype=bool),
        target=np.array([0, 0, 1, 1, 1, 0], dtype=bool),
        ligand=np.array([0, 0, 0, 0, 0, 1], dtype=bool),
    )
    logits = np.zeros((6, 6, 4), np.float32)
    bins = jnp.array([1.0, 5.0, 10.0, 20.0])
    output = FakeOutput(jnp.arange(6), jnp.asarray(logits), bins)
    sequence = jnp.zeros((2, 20))
    baseline = protein_ipsae_min(sequence, output, masks, key=jax.random.key(0))
    logits[5, :, :] = 1000
    logits[:, 5, :] = -1000
    changed = protein_ipsae_min(
        sequence,
        FakeOutput(jnp.arange(6), jnp.asarray(logits), bins),
        masks,
        key=jax.random.key(0),
    )
    assert changed == pytest.approx(baseline)
