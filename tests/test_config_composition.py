from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def _compose(name: str, overrides: list[str] | None = None) -> dict:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name=name, overrides=overrides or [])
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def test_design_groups_compose_without_python_defaults(monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/binderopt-test-cache")
    config = _compose("design", ["target=RBX1", "optimizer=biohub", "profile=reference"])
    assert config["target"]["target_id"] == "RBX1"
    assert config["optimizer"] == {
        "kind": "biohub",
        "init_scale": 0.01,
        "learning_rate": 0.1,
        "temperature_min": 0.01,
        "beta": 0.0,
    }
    assert config["binder_length"] == 80
    assert config["batch_size"] == 1
    assert config["design"]["alphabet"] == "ARNDCQEGHILKMFPSTWYV"
    assert config["design"]["excluded_amino_acids"] == ["C"]
    assert config["design"]["tail_select_steps"] <= config["design"]["n_steps"]


def test_ranking_recipes_have_only_the_reviewed_model_sets(monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/binderopt-test-cache")
    expected = {
        "full": (["esmfold2_full", "esmfold2_fast", "protenix_v2"], [0, 1, 2, 3, 4]),
        "fast": (["esmfold2_full", "esmfold2_fast", "protenix_v2"], [0]),
        "l40s": (["protenix_v2"], [0]),
    }
    for recipe, (models, offsets) in expected.items():
        config = _compose("rank", [f"ranking={recipe}"])
        assert [model["name"] for model in config["ranking"]["models"]] == models
        assert config["ranking"]["seed_offsets"] == offsets
        assert all(model["samples_per_seed"] == 1 for model in config["ranking"]["models"])
        self_consistency = config["ranking"]["self_consistency"]
        assert self_consistency["mpnn"]["excluded_amino_acids"] == ["C"]
        assert self_consistency["dockq"] == {
            "implementation": "DockQ",
            "residue_mapping": "positional_numbering",
            "target_grouping": "merged",
            "hetero_atoms": "excluded",
        }
        assert "ranking_weights" not in self_consistency

    l40s = _compose("rank", ["ranking=l40s"])
    assert l40s["ranking"]["scientific_validity"] == "development_only"
    model = l40s["ranking"]["models"][0]
    assert model["checkpoint"] == "protenix_v2"
    assert model["recycling_steps"] == 10
    assert model["sampling_steps"] == 200
    assert model["samples_per_seed"] == 1
    assert model["use_target_msa"] is True


def test_reference_scientific_values_are_visible_in_yaml(monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/binderopt-test-cache")
    config = _compose("design", ["profile=reference"])
    inference = config["profile"]["design"]["inference"]
    assert inference == {
        "recycling_steps": 1,
        "sampling_steps": 50,
        "num_diffusion_samples": 1,
        "lm_dropout": 0.5,
        "msa_max_depth": 1024,
    }
    assert config["design"]["objective"] == {
        "within_binder_contact": {"weight": 0.5, "num_contacts_per_residue": 2},
        "inter_contact": {"weight": 0.5, "contact_distance": 22.0},
        "globularity": {"weight": 0.2},
        "esmc_pll": {"weight": 0.15, "masking_fraction": 0.15, "num_samples": 4},
    }


def test_pool_experiment_resolves_stage_overrides_and_source_paths(monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/binderopt-test-cache")
    config = _compose("pool", ["stage=screen", "target=RBX1", "cell_seed=3"])
    assert config["stage"] == "screen"
    assert config["target"]["target_id"] == "RBX1"
    assert config["cell_seed"] in config["source_seeds"]
    assert config["source_dir"] == f"outputs/{config['source_experiment_id']}"
    assert config["pool_dir"] == f"{config['output_root']}/{config['experiment_id']}/RBX1"
    assert config["source_experiment_id"] != config["experiment_id"]
    assert config["profile"]["name"] == "reference"
    assert (
        0 < config["selection_count"] <= len(config["source_seeds"]) * config["candidates_per_cell"]
    )


def test_pgd_step_budget_and_parameter_scan_overrides_compose(monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/binderopt-test-cache")
    config = _compose("design", [
        "optimizer=pgd", "optimizer.stepsize_scale=0.075", "optimizer.momentum=0.25",
        "design.n_steps=75", "design.tail_select_steps=10",
    ])
    assert config["optimizer"]["stepsize_scale"] == 0.075
    assert config["optimizer"]["momentum"] == 0.25
    assert config["design"]["n_steps"] == 75
    assert config["design"]["tail_select_steps"] == 10


@pytest.mark.parametrize("command", ["design", "rank", "pool"])
def test_command_entry_points_resolve_configs_without_running_models(command, monkeypatch) -> None:
    monkeypatch.setenv("MOSAIC_CACHE_DIR", "/tmp/binderopt-test-cache")
    completed = subprocess.run(
        [sys.executable, "-m", f"binderopt.{command}", "--cfg", "job", "--resolve",
         "target=RBX1", "profile=reference"],
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = yaml.safe_load(completed.stdout)
    assert resolved["target"]["target_id"] == "RBX1"
    assert resolved["profile"]["cache"]["root"] == "/tmp/binderopt-test-cache"


@pytest.mark.parametrize("command, arguments, expected_code, expected_text", [
    ("composite", ["--help"], 0, "--ipsae-weight"),
    ("cache", [], 1, "python -m binderopt.cache check"),
    ("msa", [], 1, "python -m binderopt.msa"),
])
def test_command_help_and_usage(command, arguments, expected_code, expected_text) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", f"binderopt.{command}", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == expected_code
    assert expected_text in completed.stdout + completed.stderr
