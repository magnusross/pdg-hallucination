from __future__ import annotations

import json
import runpy
from pathlib import Path


def test_materialize_backfills_only_copied_winning_scores(tmp_path: Path) -> None:
    module = runpy.run_path(
        Path(__file__).parents[1] / "scripts" / "materialize_masked_results.py"
    )
    source = tmp_path / "source"
    cell = source / "TREM2_pgd_seed1"
    (cell / "ranking").mkdir(parents=True)
    (cell / "refs").mkdir()
    (cell / "structures").mkdir()
    (cell / "refs/reference.pdb").write_text("reference")
    (cell / "structures/judge.pdb").write_text("prediction")
    (source / "unchanged.txt").write_text("unchanged")

    target = {
        "target_id": "TREM2",
        "name": "TREM2",
        "default_binder_length": 1,
        "chains": [{"entity_id": "target_A", "sequence": "AA", "use_msa": False}],
        "ligands": [],
    }
    scores = [
        {"seed": 1, "ipsae_min": 0.8, "dockq": 0.2},
        {"seed": 2, "ipsae_min": 0.7, "dockq": 0.3},
    ]
    model = {
        "best_ipsae_min": 0.8,
        "winning_seed": 1,
        "winning_dockq": 0.2,
        "seed_scores": scores,
    }
    design = {
        "design_index": 0,
        "selected_reference": {"structure": "refs/reference.pdb"},
        "structures": {"judge": "structures/judge.pdb"},
        "ranking": {"models": {"judge": model}},
    }
    result = {
        "binder_length": 1,
        "config": {"target": target},
        "designs": [design],
    }
    (cell / "result.json").write_text(json.dumps(result))
    shard = {
        "provenance": {
            "self_consistency": {"target_aligned_rmsd": {"atom_names": ["CA"]}}
        },
        "designs": [
            {
                "design_index": 0,
                "winning_seed": 1,
                "winning_dockq": 0.2,
                "seed_scores": [dict(item) for item in scores],
            }
        ],
    }
    (cell / "ranking/judge.json").write_text(json.dumps(shard))
    target_config = tmp_path / "TREM2.yaml"
    target_config.write_text(
        "\n".join(
            [
                "target_id: TREM2",
                "name: TREM2",
                "default_binder_length: 1",
                "chains:",
                "  - entity_id: target_A",
                "    sequence: AA",
                "    use_msa: false",
                "ligands: []",
                "self_consistency:",
                "  primary_target_residue_ranges:",
                "    target_A: [[1, 1]]",
            ]
        )
    )

    def fake_score(reference, prediction, **kwargs):
        assert Path(reference).read_text() == "reference"
        assert Path(prediction).read_text() == "prediction"
        assert kwargs["target_residue_ranges"] == (((1, 1),),)
        return {
            "dockq": 0.9,
            "target_aligned_binder_backbone_rmsd": 2.0,
            "self_consistency_target_residue_ranges": [[[1, 1]]],
            "self_consistency_range_indexing": "one_based_inclusive",
            "unmasked_self_consistency": {
                "dockq": 0.2,
                "target_aligned_binder_backbone_rmsd": 10.0,
            },
        }

    module["_backfill_cell"].__globals__["score_self_consistency"] = fake_score
    destination = tmp_path / "destination"
    manifest = module["materialize_results"](
        source, destination, target_config_path=target_config
    )

    assert manifest["updated_cells"] == 1
    assert manifest["updated_winning_scores"] == 1
    assert (destination / "unchanged.txt").is_symlink()
    assert (destination / "TREM2_pgd_seed1/structures/judge.pdb").is_symlink()
    copied_result_path = destination / "TREM2_pgd_seed1/result.json"
    copied_shard_path = destination / "TREM2_pgd_seed1/ranking/judge.json"
    assert not copied_result_path.is_symlink()
    assert not copied_shard_path.is_symlink()
    copied_result = json.loads(copied_result_path.read_text())
    copied_model = copied_result["designs"][0]["ranking"]["models"]["judge"]
    assert copied_model["winning_dockq"] == 0.9
    assert copied_model["seed_scores"][0]["target_aligned_binder_backbone_rmsd"] == 2.0
    assert copied_model["seed_scores"][1] == scores[1]
    assert copied_result["config"]["target"]["self_consistency"] == {
        "primary_target_residue_ranges": {"target_A": [[1, 1]]}
    }
    original = json.loads((cell / "result.json").read_text())
    assert original["designs"][0]["ranking"]["models"]["judge"]["winning_dockq"] == 0.2
