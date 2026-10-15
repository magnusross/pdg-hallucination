from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

_ANALYSIS = runpy.run_path(Path(__file__).parents[1] / "analysis" / "optimizer_analysis.py")
load_optimizer_experiment = _ANALYSIS["load_optimizer_experiment"]
main = _ANALYSIS["main"]
plot_pool_maxima = _ANALYSIS["_plot_pool_maxima"]
plot_survival = _ANALYSIS["_plot_survival"]
run_analysis = _ANALYSIS["run_analysis"]
summarize_endpoints = _ANALYSIS["summarize_endpoints"]
summarize_compute = _ANALYSIS["summarize_compute"]


def _model(ipsae: float, dockq: float) -> dict:
    return {
        "seed_scores": [
            {"seed": 11, "ipsae_min": ipsae, "dockq": dockq},
            {"seed": 12, "ipsae_min": ipsae - 0.1, "dockq": 0.99},
        ],
        "best_ipsae_min": ipsae,
        "winning_dockq": dockq,
        "winning_seed": 11,
    }


def _write_cell(
    path: Path,
    *,
    optimizer: str,
    optimizer_id: str | None = None,
    experiment_id: str = "optimizer_test",
    base_seed: int = 7,
    ipsae: tuple[float, float],
    dockq: tuple[float, float],
) -> None:
    designs = []
    for design_index, (ipsae_value, dockq_value) in enumerate(zip(ipsae, dockq, strict=True)):
        models = {
            "judge_a": _model(ipsae_value, dockq_value),
            "judge_b": _model(ipsae_value, dockq_value),
        }
        designs.append(
            {
                "design_index": design_index,
                "design_key": [base_seed, design_index],
                "optimizer_sequence": "AAA",
                "sequence": "DDD",
                "argmax_loss": 2.0 - ipsae_value,
                "optimizer_esmc_perplexity": 3.0,
                "mpnn": {"esmc_perplexity": 2.0},
                "ranking": {"models": models},
                "ipsae_ranking_score": ipsae_value,
                "ranking_score": ipsae_value,
                "rank_loss": -ipsae_value,
            }
        )
    result = {
        "experiment_id": experiment_id,
        "target_id": "target",
        "objective_id": "objective",
        "model_id": "design_models",
        "ranking_model_id": "full",
        "optimizer_kind": optimizer,
        "binder_length": 3,
        "num_designs_requested": 2,
        "num_designs": 2,
        "base_seed": base_seed,
        "batch_size": 2,
        "git_sha": "test",
        "design_timing": {"process_wall_time_s": 3600, "gpu_count": 2},
        "ranking_timing": {"cell_wall_time_s": 1800},
        "config": {
            "profile": {"resources": {"gpus_per_task": 1}},
            "optimizer": {
                "kind": optimizer,
                **({"optimizer_id": optimizer_id} if optimizer_id is not None else {}),
            },
            "target": {
                "target_id": "target",
                "name": "target",
                "default_binder_length": 3,
                "chains": [{"entity_id": "target", "sequence": "AAA", "use_msa": False}],
                "ligands": [],
            },
        },
        "designs": designs,
        "summary": {
            "ranking_score_best": max(ipsae),
            "rank_loss_best": -max(ipsae),
        },
    }
    path.mkdir(parents=True)
    (path / "result.json").write_text(json.dumps(result))


def _experiment(path: Path, *, experiment_id: str = "optimizer_test", base_seed: int = 7) -> None:
    _write_cell(
        path / "target_biohub_seed7",
        optimizer="biohub",
        experiment_id=experiment_id,
        base_seed=base_seed,
        ipsae=(0.2, 0.3),
        dockq=(0.8, 0.7),
    )
    _write_cell(
        path / "target_pgd_seed7",
        optimizer="pgd",
        experiment_id=experiment_id,
        base_seed=base_seed,
        ipsae=(0.8, 0.7),
        dockq=(0.1, 0.2),
    )
    _write_cell(
        path / "target_pgd_lr0p075_m0p25_seed7",
        optimizer="pgd_lr0p075_m0p25",
        experiment_id=experiment_id,
        base_seed=base_seed,
        ipsae=(0.6, 0.5),
        dockq=(0.5, 0.4),
    )


def test_loader_computes_dockq_composite_and_confirms_design_key_pairs(
    tmp_path: Path,
) -> None:
    _experiment(tmp_path)

    rows, cells, composite, optimizers = load_optimizer_experiment(
        tmp_path,
        optimizers=["biohub", "pgd", "pgd_lr0p075_m0p25"],
        ipsae_weight=4,
        dockq_weight=1,
    )

    assert len(rows) == 6
    assert len(cells) == 3
    assert optimizers == ("biohub", "pgd", "pgd_lr0p075_m0p25")
    assert {row["design_key"] for row in rows} == {(7, 0), (7, 1)}
    # The losing seed has DockQ 0.99; it must not leak into the endpoint.
    assert {row["mean_winning_dockq"] for row in rows if row["optimizer"] == "pgd"} == {0.1, 0.2}
    manifest_scores = {
        (item["optimizer_kind"], item["design_index"]): item["composite_score"]
        for item in composite["targets"]["target"]["designs"]
    }
    assert all(
        row["composite_score"]
        == pytest.approx(manifest_scores[(row["optimizer"], row["design_index"])])
        for row in rows
    )

    _, comparisons = summarize_endpoints(
        rows,
        optimizers=optimizers,
        top_fraction=1.0,
        n_bootstrap=200,
        analysis_seed=3,
    )
    composite_effects = [item for item in comparisons if item["metric"] == "composite_score"]
    assert len(composite_effects) == 3
    pgd_vs_biohub = next(
        item
        for item in composite_effects
        if item["optimizer_a"] == "biohub" and item["optimizer_b"] == "pgd"
    )
    assert pgd_vs_biohub["optimizer_b_minus_a"] > 0


def test_loader_rejects_dockq_not_taken_from_ipsae_winner(tmp_path: Path) -> None:
    _experiment(tmp_path)
    result_path = tmp_path / "target_pgd_seed7/result.json"
    result = json.loads(result_path.read_text())
    result["designs"][0]["ranking"]["models"]["judge_a"]["winning_dockq"] = 0.99
    result_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="iPsae-winning seed"):
        load_optimizer_experiment(
            tmp_path,
            optimizers=None,
            ipsae_weight=4,
            dockq_weight=1,
        )


def test_loader_distinguishes_recipes_with_the_same_optimizer_kind(tmp_path: Path) -> None:
    _experiment(tmp_path)
    _write_cell(
        tmp_path / "target_pgd_alt_seed7",
        optimizer="pgd",
        optimizer_id="pgd_alt",
        ipsae=(0.4, 0.5),
        dockq=(0.4, 0.3),
    )

    rows, _, _, optimizers = load_optimizer_experiment(
        tmp_path,
        optimizers=["biohub", "pgd", "pgd_alt", "pgd_lr0p075_m0p25"],
        ipsae_weight=4,
        dockq_weight=1,
    )

    assert optimizers == ("biohub", "pgd", "pgd_alt", "pgd_lr0p075_m0p25")
    assert {row["optimizer"] for row in rows} == set(optimizers)


def test_loader_can_select_a_subset_of_optimizer_arms(tmp_path: Path) -> None:
    _experiment(tmp_path)
    rows, _, _, optimizers = load_optimizer_experiment(
        tmp_path,
        optimizers=["biohub", "pgd"],
        ipsae_weight=4,
        dockq_weight=1,
    )

    assert optimizers == ("biohub", "pgd")
    assert {row["optimizer"] for row in rows} == {"biohub", "pgd"}


@pytest.mark.parametrize("merged", [False, True])
def test_loader_pools_distinct_experiments_and_preserves_sources(
    tmp_path: Path, merged: bool
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _experiment(first, experiment_id="first_experiment", base_seed=7)
    _experiment(second, experiment_id="second_experiment", base_seed=8)

    rows, cells, composite, optimizers = load_optimizer_experiment(
        tmp_path if merged else [first, second],
        optimizers=["biohub", "pgd"],
        ipsae_weight=4,
        dockq_weight=1,
    )

    assert len(rows) == 8
    assert len(cells) == 4
    assert optimizers == ("biohub", "pgd")
    assert composite["experiment_id"] is None
    assert composite["experiment_ids"] == ["first_experiment", "second_experiment"]
    inputs = [str(tmp_path)] if merged else [str(first), str(second)]
    assert composite["inputs"] == inputs
    assert {row["experiment_id"] for row in rows} == {
        "first_experiment",
        "second_experiment",
    }
    assert {row["input"] for row in rows} == set(inputs)
    assert {row["design_key"] for row in rows} == {
        (7, 0),
        (7, 1),
        (8, 0),
        (8, 1),
    }


def test_cli_pools_multiple_experiment_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "analysis"
    _experiment(first, experiment_id="first_experiment", base_seed=7)
    _experiment(second, experiment_id="second_experiment", base_seed=8)

    main(
        [
            str(first),
            str(second),
            "--references",
            "biohub",
            "--challengers",
            "pgd",
            "--ipsae-weight",
            "4",
            "--dockq-weight",
            "1",
            "--n-bootstrap",
            "20",
            "--output",
            str(output),
            "--no-plots",
        ]
    )

    result = json.loads((output / "analysis.json").read_text())
    assert result["experiment_id"] is None
    assert result["experiment_ids"] == ["first_experiment", "second_experiment"]
    assert result["input"] is None
    assert result["inputs"] == [str(first), str(second)]
    assert result["n_designs"] == 8
    assert "first_experiment" in (output / "cells.csv").read_text()
    assert "second_experiment" in (output / "cells.csv").read_text()


def test_loader_rejects_duplicate_design_keys_across_experiments(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _experiment(first, experiment_id="first_experiment")
    _experiment(second, experiment_id="second_experiment")

    with pytest.raises(ValueError, match="duplicate design_key"):
        load_optimizer_experiment(
            [first, second],
            optimizers=["biohub", "pgd"],
            ipsae_weight=4,
            dockq_weight=1,
        )


def test_run_analysis_writes_reusable_tables_without_results_fixture(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    _experiment(experiment)
    output = tmp_path / "analysis"

    result = run_analysis(
        experiment,
        output_dir=output,
        optimizers=None,
        ipsae_weight=4,
        dockq_weight=1,
        top_fraction=0.1,
        n_bootstrap=100,
        analysis_seed=3,
        make_plots=False,
    )

    assert result["n_paired_initializations"] == 2
    assert result["optimizers"] == ["biohub", "pgd", "pgd_lr0p075_m0p25"]
    assert result["composite_weights"] == {"ipsae": 4.0, "dockq": 1.0}
    assert result["plots"] == []
    assert result["total_gpu_hours"] == pytest.approx(7.5)
    assert "| target | 3 | 6 | 6.00 | 1.50 | 7.50 |" in (
        output / "compute_summary.md"
    ).read_text()
    assert "| Total | 3 | 6 | 6.00 | 1.50 | 7.50 |" in (
        output / "compute_summary.md"
    ).read_text()
    assert "total_gpu_hours" in (output / "compute_summary.csv").read_text()
    assert (output / "composite_ranking.json").is_file()
    assert "composite_score" in (output / "designs.csv").read_text()
    assert "optimizer_b_minus_a" in (output / "paired_comparisons.csv").read_text()


def test_compute_rollup_sums_targets_without_optimizer_breakdown() -> None:
    cells = [
        {"target": "A", "n_designs": 64, "design_gpu_hours": 2.0, "ranking_gpu_hours": 1.0},
        {"target": "A", "n_designs": 64, "design_gpu_hours": 3.0, "ranking_gpu_hours": 2.0},
        {"target": "B", "n_designs": 32, "design_gpu_hours": 1.0, "ranking_gpu_hours": 0.5},
    ]
    assert summarize_compute(cells) == [
        {"target": "A", "cells": 2, "designs": 128, "design_gpu_hours": 5.0,
         "ranking_gpu_hours": 3.0, "total_gpu_hours": 8.0},
        {"target": "B", "cells": 1, "designs": 32, "design_gpu_hours": 1.0,
         "ranking_gpu_hours": 0.5, "total_gpu_hours": 1.5},
        {"target": "Total", "cells": 3, "designs": 160, "design_gpu_hours": 6.0,
         "ranking_gpu_hours": 3.5, "total_gpu_hours": 9.5},
    ]


def test_compute_rejects_missing_timing(tmp_path: Path) -> None:
    _experiment(tmp_path)
    result_path = tmp_path / "target_pgd_seed7/result.json"
    result = json.loads(result_path.read_text())
    del result["ranking_timing"]
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="missing or invalid compute timing"):
        load_optimizer_experiment(tmp_path, ipsae_weight=4, dockq_weight=1)


def test_pool_maximum_plot_is_pdf(tmp_path: Path) -> None:
    rows = [
        {
            "target": target,
            "optimizer": optimizer,
            "design_key": [7, design_index],
            "base_seed": 7,
            "design_index": design_index,
            "ranking_score": score,
        }
        for target in ("target_a", "target_b")
        for optimizer, scores in (("biohub", (0.2, 0.4)), ("pgd", (0.3, 0.5)))
        for design_index, score in enumerate(scores)
    ]

    name = plot_pool_maxima(
        rows,
        optimizers=("biohub", "pgd"),
        output_dir=tmp_path,
    )

    assert name == "max_ipsae_vs_pool_size.pdf"
    assert (tmp_path / name).read_bytes().startswith(b"%PDF")


def test_survival_plot_includes_full_and_tail_primary_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    saved_curves = {}
    savefig = Figure.savefig

    def capture_figure(figure, path, **kwargs):
        axis = figure.axes[0]
        saved_curves[Path(path).name] = (axis.get_xlim(), axis.lines[0].get_data())
        return savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    rows = [
        {
            "target": f"target_{target}",
            "optimizer": optimizer,
            "ranking_score": 0.4 + 0.01 * design_index,
            "composite_score": -0.2 + 0.05 * design_index,
        }
        for target in range(10)
        for optimizer in ("biohub", "pgd")
        for design_index in range(4)
    ]

    names = plot_survival(
        rows,
        optimizers=("biohub", "pgd"),
        output_dir=tmp_path,
        n_bootstrap=20,
        analysis_seed=3,
        tail_max_fraction=0.5,
    )

    assert names == ["ranking_score_survival_tail.pdf", "ranking_score_survival.pdf"]
    assert all((tmp_path / name).read_bytes().startswith(b"%PDF") for name in names)
    limits, (thresholds, survival) = saved_curves["ranking_score_survival.pdf"]
    assert limits == (0, 1)
    np.testing.assert_allclose(thresholds, [0, .4, .41, .42, .43, 1])
    np.testing.assert_allclose(survival, [1, 1, .75, .5, .25, 0])
    tail_limits, _ = saved_curves["ranking_score_survival_tail.pdf"]
    np.testing.assert_allclose(tail_limits, [.415, .43])


def test_analysis_can_exclude_an_unranked_target(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    _experiment(experiment)
    _write_cell(
        experiment / "excluded_biohub_seed7",
        optimizer="biohub",
        ipsae=(0.2, 0.3),
        dockq=(0.8, 0.7),
    )
    result_path = experiment / "excluded_biohub_seed7/result.json"
    result = json.loads(result_path.read_text())
    result["target_id"] = "excluded"
    result["config"]["target"]["target_id"] = "excluded"
    result["ranking_model_id"] = None
    result_path.write_text(json.dumps(result))

    result = run_analysis(
        experiment,
        output_dir=tmp_path / "analysis",
        optimizers=None,
        exclude_targets=["excluded"],
        ipsae_weight=4,
        dockq_weight=1,
        top_fraction=0.1,
        n_bootstrap=100,
        analysis_seed=3,
        make_plots=False,
    )

    assert result["targets"] == ["target"]
    assert result["excluded_targets"] == ["excluded"]


def test_pssm_dynamics_preserves_design_step_identity(tmp_path: Path) -> None:
    # Both designs start on a face, then restore a previously excluded choice.
    pssms = np.array([
        [[1.0, 0.0]], [[0.5, 0.5]], [[1.0, 0.0]],
        [[0.0, 1.0]], [[0.5, 0.5]], [[0.0, 1.0]],
    ])
    order = np.array([4, 0, 5, 2, 1, 3])
    support, reentry = _ANALYSIS["_reduce_pssm_dynamics"](
        pssms[order], np.repeat([0, 1], 3)[order], np.tile([0, 1, 2], 2)[order],
        n_designs=2, n_steps=3, cell=tmp_path,
    )
    np.testing.assert_array_equal(support, [[1, 2, 1], [1, 2, 1]])
    assert np.isnan(reentry[:, 0]).all()
    np.testing.assert_array_equal(reentry[:, 1:], [[1, 0], [1, 0]])


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "invalid_probability"])
def test_pssm_dynamics_rejects_invalid_trajectories(tmp_path: Path, corruption: str) -> None:
    pssms = np.array([[[1.0, 0.0]], [[0.5, 0.5]], [[0.0, 1.0]]])
    indices, steps = np.zeros(3, dtype=int), np.arange(3)
    if corruption == "duplicate":
        steps[2] = 1
    elif corruption == "missing":
        pssms, indices, steps = pssms[:-1], indices[:-1], steps[:-1]
    else:
        pssms[0, 0, 0] = -1
    with pytest.raises(ValueError):
        _ANALYSIS["_reduce_pssm_dynamics"](
            pssms, indices, steps, n_designs=1, n_steps=3, cell=tmp_path,
        )


def _write_dynamics_cell(path: Path, n_designs: int, offset: float) -> dict:
    path.mkdir()
    pssms = np.tile(np.array([[[1.0, 0.0]], [[0.5, 0.5]], [[0.0, 1.0]]]),
                    (n_designs, 1, 1))
    result = {
        "config": {"design": {"n_steps": 3, "tail_select_steps": 2}},
        "designs": [{"design_index": index, "selected_step": 1,
                     "design_objective": offset + 2, "argmax_loss": offset + 2.25}
                    for index in range(n_designs)],
    }
    (path / "result.json").write_text(json.dumps(result))
    (path / "opt_trajectory.json").write_text(json.dumps({"trajectory": [
        {"design_index": index, "step": step, "objective": offset + 3 - step}
        for index in range(n_designs) for step in range(3)
    ]}))
    np.savez(path / "opt_trajectory_pssms.npz", pssm=pssms,
             design_index=np.repeat(np.arange(n_designs), 3), step=np.tile(np.arange(3), n_designs))
    np.savez(path / "pssms.npz", pssm=pssms[1::3], design_index=np.arange(n_designs))
    return {"input": str(path.parent), "cell": path.name, "base_seed": 1,
            "target": "target", "optimizer": "pgd"}


def test_dynamics_reduction_uses_selected_iterate_and_weights_designs(tmp_path: Path) -> None:
    cells = [_write_dynamics_cell(tmp_path / "one", 1, 0),
             _write_dynamics_cell(tmp_path / "two", 2, 3)]
    records, endpoints = _ANALYSIS["_summarize_dynamics"](cells, output_dir=tmp_path)
    assert [row["objective_change"] for row in records] == [0, -1, -2]
    assert [row["support"] for row in records] == [1, 2, 1]
    assert [row["reentry"] for row in records[1:]] == [1, 0]
    assert records[0]["n_designs"] == 3
    # Tail selection is step two, not the last saved iterate.
    assert {row["selected_step"] for row in endpoints} == {2}
    assert {row["selected_support"] for row in endpoints} == {2}
    assert {row["argmax_minus_soft"] for row in endpoints} == {.25}
    assert np.mean([row["soft_loss"] for row in endpoints]) == 4
    assert (tmp_path / "optimization_dynamics_curves.csv").exists()


@pytest.mark.parametrize("corruption", ["selected_pssm", "selected_objective", "missing_pssms"])
def test_dynamics_reduction_rejects_mismatched_selected_data(
    tmp_path: Path, corruption: str,
) -> None:
    cell = _write_dynamics_cell(tmp_path / "cell", 1, 0)
    if corruption == "selected_pssm":
        np.savez(tmp_path / "cell/pssms.npz", pssm=[[[1.0, 0.0]]], design_index=[0])
    elif corruption == "selected_objective":
        path = tmp_path / "cell/result.json"
        result = json.loads(path.read_text())
        result["designs"][0]["design_objective"] = 100
        path.write_text(json.dumps(result))
    else:
        other = _write_dynamics_cell(tmp_path / "other", 1, 0)
        (tmp_path / "other/opt_trajectory_pssms.npz").unlink()
        with pytest.raises(FileNotFoundError):
            _ANALYSIS["_summarize_dynamics"]([cell, other], output_dir=tmp_path)
        return
    with pytest.raises(ValueError, match="Selected"):
        _ANALYSIS["_summarize_dynamics"]([cell], output_dir=tmp_path)


def test_report_median_table_uses_whole_pool_medians(tmp_path: Path) -> None:
    summaries = [
        {"metric": metric, "target": "4ZQK", "optimizer": arm,
         "median": median, "estimate": 0.99}
        for metric in ("ranking_score", "mean_winning_dockq")
        for arm, median in (("biohub", 0.2), ("pgd", 0.3))
    ]
    _ANALYSIS["_write_report_median_table"](
        summaries, optimizers=("biohub", "pgd"), output_dir=tmp_path,
    )
    table = (tmp_path / "report_median_table.tex").read_text()
    assert "Median ipSAE & Logit GD & 0.200" in table
    assert "Median scDockQ & Logit GD & 0.200" in table
    assert table.count("PGD & 0.300") == 2
    assert "0.990" not in table
    assert "textbf" not in table
