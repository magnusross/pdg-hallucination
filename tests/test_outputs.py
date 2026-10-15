from __future__ import annotations

import json

import numpy as np
import pytest

import binderopt.outputs as outputs
from binderopt.outputs import write_design_outputs


def _identity() -> dict:
    return {
        "experiment_id": "exp",
        "target_id": "4ZQK",
        "objective_id": "esm_binder",
        "model_id": "esm_small",
        "ranking_model_id": None,
        "optimizer_kind": "pgd",
        "binder_length": 3,
        "num_designs_requested": 2,
    }


def _design(index: int) -> dict:
    return {
        "design_index": index,
        "optimizer_sequence": "ARN",
        "argmax_loss": 1.0 + index,
        "tail_ranking_loss": -0.5,
        "tail_iptm": 0.5,
        "design_objective": 2.5,
        "selected_step": 3,
        "seed": 7,
        "batch": 0,
        "batch_lane": index,
        "design_time_s": 0.2,
        "metrics": {},
    }


def _references(count: int) -> list[dict]:
    return [
        {
            "selected_model": "design_a",
            "models": [
                {
                    "model_name": "design_a",
                    "checkpoint": "checkpoint-a",
                    "protein_iptm": 0.7,
                    "settings": {},
                    "_structure_text": f"MODEL {index}\n",
                }
            ],
        }
        for index in range(count)
    ]


def test_atomic_replace_leaves_previous_json_on_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "result.json"
    outputs.atomic_json(path, {"complete": 1})

    def fail_replace(source, destination):
        raise OSError("simulated interruption")

    monkeypatch.setattr(outputs.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        outputs.atomic_json(path, {"partial": 2})
    assert json.loads(path.read_text()) == {"complete": 1}


def test_final_design_output_preserves_generation_identity(tmp_path) -> None:
    result = write_design_outputs(
        tmp_path,
        identity=_identity(),
        config={"resolved": True},
        git_sha="abc-dirty",
        hydra_overrides=["optimizer=pgd"],
        designs=[_design(0), _design(1)],
        pssms=[np.zeros((3, 20), np.float32), np.ones((3, 20), np.float32) / 20],
        references=_references(2),
        trajectory=[
            {
                "design_index": 0,
                "step": 1,
                "objective": 2.0,
                "pssm": np.full((3, 20), 0.05, np.float32),
            }
        ],
        process_started=0.0,
        started_at_utc="2026-01-01T00:00:00+00:00",
        setup_wall_time_s=0.0,
        optimization_wall_time_s=0.0,
        reference_prediction_wall_time_s=0.0,
        gpu_count=1,
    )
    assert [item["design_index"] for item in result["designs"]] == [0, 1]
    assert "status" not in result
    assert "schema_version" not in result
    assert "rank_loss" not in result["designs"][0]
    assert "summary" not in result
    assert result["design_timing"]["setup_wall_time_s"] == 0.0
    assert result["design_timing"]["gpu_count"] == 1
    assert result["designs"][0]["optimizer_sequence"] == "ARN"
    assert result["designs"][0]["selected_reference"]["model_name"] == "design_a"
    assert not (tmp_path / "designs.fasta").exists()
    assert (tmp_path / "optimizer_designs.fasta").read_text().endswith("ARN\n")
    assert (tmp_path / "design_references/design_000/design_a.pdb").is_file()
    with np.load(tmp_path / "pssms.npz") as archive:
        np.testing.assert_array_equal(archive["design_index"], [0, 1])
        assert archive["pssm"][0].sum() == 0
    with np.load(tmp_path / "opt_trajectory_pssms.npz") as archive:
        np.testing.assert_array_equal(archive["design_index"], [0])
        np.testing.assert_array_equal(archive["step"], [1])
        np.testing.assert_allclose(archive["pssm"][0], np.full((3, 20), 0.05, np.float32))
    trajectory = json.loads((tmp_path / "opt_trajectory.json").read_text())["trajectory"]
    assert "pssm" not in trajectory[0]
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_design_output_refuses_to_replace_existing_result(tmp_path) -> None:
    kwargs = {
        "identity": _identity(),
        "config": {},
        "git_sha": "abc",
        "hydra_overrides": [],
        "designs": [_design(0), _design(1)],
        "pssms": [np.zeros((3, 20), np.float32)] * 2,
        "trajectory": [],
        "references": _references(2),
        "process_started": 0.0,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "setup_wall_time_s": 0.0,
        "optimization_wall_time_s": 0.0,
        "reference_prediction_wall_time_s": 0.0,
        "gpu_count": 1,
    }
    write_design_outputs(tmp_path, **kwargs)
    with pytest.raises(FileExistsError, match="already exists"):
        write_design_outputs(tmp_path, **kwargs)
