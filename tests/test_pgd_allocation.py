from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from binderopt.ranking.cells import select_designs_for_ranking

with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(Path(__file__).parents[1]))
    _ANALYSIS = runpy.run_path(Path(__file__).parents[1] / "analysis/pgd_allocation.py")
load_allocation = _ANALYSIS["load_allocation"]
add_logit_gd = _ANALYSIS["add_logit_gd"]
plot_allocation = _ANALYSIS["plot_allocation"]
summarize = _ANALYSIS["summarize"]
hit_increase_table = _ANALYSIS["hit_increase_table"]
write_hit_table = _ANALYSIS["write_hit_table"]

_write_cell = runpy.run_path(Path(__file__).with_name("test_optimizer_analysis.py"))["_write_cell"]


@pytest.fixture
def allocation(tmp_path):
    roots = {a: tmp_path / a for a in ("long", "short")}
    spec = {
        "num_designs": 4,
        "design": {"n_steps": 2, "tail_select_steps": 1},
        "ranking_preselection": {
            "count": 2,
            "expected_candidates": 4,
            "metric": "tail_iptm",
            "tie_breaker": "design_index",
        },
    }
    for arm, root in roots.items():
        cell = root / "target_pgd_seed7"
        _write_cell(cell, optimizer="pgd", ipsae=(0.2, 0.8), dockq=(0.3, 0.4))
        path = cell / "result.json"
        result = json.loads(path.read_text())
        config = result["config"]
        config["design"] = {
            "n_steps": 2 if arm == "short" else 4,
            "tail_select_steps": 1 if arm == "short" else 2,
        }
        config["num_designs"] = 4 if arm == "short" else 2
        if arm == "short":
            config["ranking_preselection"] = spec["ranking_preselection"]
            result["num_designs_requested"] = 4
            candidates = []
            for index in range(4):
                design = copy.deepcopy(result["designs"][index % 2])
                design.update(design_index=index, design_key=[7, index], tail_iptm=1 - index / 5)
                candidates.append(design)
            result["design_candidates"] = candidates
            _, result["ranking_preselection"] = select_designs_for_ranking(
                result, spec["ranking_preselection"]
            )
        path.write_text(json.dumps(result))
    return roots, dict(
        targets=["target", "missing"],
        seeds=[7],
        short_config=spec,
        ranking_recipe={
            "name": "full",
            "models": [{"name": "judge_a"}, {"name": "judge_b"}],
            "self_consistency": {"required_seed_count": 2},
        },
    )


def test_complete_targets_only_and_shared_candidate_keys(allocation):
    roots, kwargs = allocation
    pools, coverage = load_allocation([roots["long"]], roots["short"], **kwargs)
    assert list(pools) == ["target"]
    assert coverage[1]["included"] is False
    assert len(coverage[1]["reasons"]) == 2
    path = roots["short"] / "target_pgd_seed7/result.json"
    result = json.loads(path.read_text())
    result["design_candidates"][0]["design_key"] = [999, 0]
    result["designs"][0]["design_key"] = [999, 0]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="shared design keys"):
        load_allocation([roots["long"]], roots["short"], **kwargs)


def test_unranked_shard_excludes_whole_target(allocation):
    roots, kwargs = allocation
    path = roots["short"] / "target_pgd_seed7/result.json"
    result = json.loads(path.read_text())
    result.pop("ranking_model_id")
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="no complete target cohorts"):
        load_allocation([roots["long"]], roots["short"], **kwargs)


def test_scientific_mismatch_is_not_silently_skipped(allocation):
    roots, kwargs = allocation
    path = roots["short"] / "target_pgd_seed7/result.json"
    result = json.loads(path.read_text())
    result["config"]["target"]["chains"][0]["sequence"] = "CCC"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="scientific configurations differ"):
        load_allocation([roots["long"]], roots["short"], **kwargs)


def test_pooled_rate_uses_sum_of_compute_and_strict_thresholds():
    pools = {}
    for target, hours in (("A", 1.0), ("B", 9.0)):
        pools[target] = {
            arm: [
                {
                    "seed": 1,
                    "scores": np.array([0.5, 0.8]),
                    "sequences": ["AAA", "BBB"],
                    "design_gpu_hours": hours,
                    "ranking_gpu_hours": 0.0,
                }
            ]
            for arm in ("long", "short")
        }
    yields, endpoints, _ = summarize(
        pools, np.array([0.5, 0.8]), top_fraction=0.1, n_bootstrap=20, analysis_seed=1
    )
    assert yields[0]["hits"] == 2
    assert yields[0]["hits_per_gpu_hour"] == pytest.approx(0.2)
    assert yields[1]["hits"] == 0
    a_rate = next(r for r in yields if r["target"] == "A" and r["threshold"] == 0.5)
    b_rate = next(r for r in yields if r["target"] == "B" and r["threshold"] == 0.5)
    assert a_rate["hits_per_gpu_hour"] == pytest.approx(1.0)
    assert b_rate["hits_per_gpu_hour"] == pytest.approx(1 / 9)
    assert all(row["mean"] == pytest.approx(0.8) for row in endpoints)
    assert all(row["difference_ci_low"] == 0 for row in endpoints)


def test_paired_shards_preserve_constant_score_difference():
    pools = {
        "A": {
            arm: [
                {
                    "seed": i,
                    "scores": np.array([i / 10 + delta]),
                    "sequences": ["AAA"],
                    "design_gpu_hours": 1.0,
                    "ranking_gpu_hours": 0.0,
                }
                for i in range(1, 5)
            ]
            for arm, delta in (("long", 0), ("short", 0.1))
        }
    }
    _, endpoints, _ = summarize(
        pools, np.array([0.2]), top_fraction=0.5, n_bootstrap=100, analysis_seed=1
    )
    assert endpoints[0]["difference_ci_low"] == pytest.approx(0.1)
    assert endpoints[0]["difference_ci_high"] == pytest.approx(0.1)


def test_unique_yield_counts_a_sequence_once_within_each_target():
    pools = {
        target: {
            arm: [
                {
                    "seed": 1,
                    "scores": np.array([0.6, 0.8]),
                    "sequences": ["AAA", "AAA"],
                    "design_gpu_hours": 1.0,
                    "ranking_gpu_hours": 0.0,
                }
            ]
            for arm in ("long", "short")
        }
        for target in ("A", "B")
    }
    yields, _, _ = summarize(
        pools, np.array([0.5]), top_fraction=0.1, n_bootstrap=10, analysis_seed=1
    )
    assert yields[0]["hits"] == 4
    assert yields[0]["unique_hits"] == 2
    assert yields[0]["unique_hits_per_gpu_hour"] == pytest.approx(1.0)


def test_without_bootstrap_preserves_point_estimates(allocation):
    roots, kwargs = allocation
    pools, _ = load_allocation([roots["long"]], roots["short"], **kwargs)
    thresholds = np.array([0.5, 0.8])
    plain, quality, compute = summarize(
        pools, thresholds, top_fraction=0.1, n_bootstrap=0, analysis_seed=1
    )
    boot, _, _ = summarize(pools, thresholds, top_fraction=0.1, n_bootstrap=20, analysis_seed=1)
    assert [r["hits_per_gpu_hour"] for r in plain] == [r["hits_per_gpu_hour"] for r in boot]
    assert all(r["ci_low"] is None and r["ci_high"] is None for r in plain + quality)
    assert quality[0]["top_count"] == 1
    assert len(compute) == 2


def _add_saved_logit_gd(roots):
    control = roots["long"] / "target_pgd_seed7/result.json"
    result = json.loads(control.read_text())
    result["optimizer_kind"] = "biohub"
    result["config"]["optimizer"] = {"kind": "biohub", "init_scale": 0.01}
    result["design_timing"]["process_wall_time_s"] = 7200
    path = roots["long"] / "target_biohub_seed7/result.json"
    path.parent.mkdir()
    path.write_text(json.dumps(result))
    return path


def test_logit_gd_uses_matching_cohort_and_its_own_compute(allocation, tmp_path):
    roots, kwargs = allocation
    _add_saved_logit_gd(roots)
    pools, _ = load_allocation([roots["long"]], roots["short"], **kwargs)
    add_logit_gd(pools, [roots["long"]])
    yields, quality, compute = summarize(
        pools, np.array([0.5, 0.8, 0.9]), top_fraction=0.1, n_bootstrap=0, analysis_seed=1
    )
    gd = next(r for r in yields if r["target"] == "Total" and r["arm"] == "biohub")
    assert gd["hits"] == 1
    assert gd["gpu_hours"] == 4.5
    assert gd["hits_per_gpu_hour"] == pytest.approx(1 / 4.5)
    assert len(compute) == 3
    assert next(r for r in quality if r["arm"] == "biohub")["short_minus_long"] is None
    import matplotlib.pyplot as plt

    fig = plot_allocation(
        yields, labels={a: a for a in ("long", "short", "biohub")}, output_dir=tmp_path
    )
    assert len(fig.axes[0].lines) == 3
    plt.close(fig)


def test_logit_gd_rejects_unpaired_design_keys(allocation):
    roots, kwargs = allocation
    path = _add_saved_logit_gd(roots)
    result = json.loads(path.read_text())
    result["designs"][0]["design_key"] = [999, 0]
    path.write_text(json.dumps(result))
    pools, _ = load_allocation([roots["long"]], roots["short"], **kwargs)
    with pytest.raises(ValueError, match="Logit GD paired design keys"):
        add_logit_gd(pools, [roots["long"]])


def test_figure_plots_only_pooled_yield_above_half(tmp_path):
    import matplotlib.pyplot as plt

    yields = [
        {
            "target": "Total",
            "arm": arm,
            "threshold": t,
            "hits_per_gpu_hour": max(0, 0.9 - t),
        }
        for arm in ("long", "short")
        for t in (0.0, 0.5, 0.9, 1.0)
    ]
    # Target-specific rows must not contribute curves or influence the scale.
    yields.append(
        {
            "target": "A",
            "arm": "long",
            "threshold": 0.5,
            "hits_per_gpu_hour": 100.0,
            "ci_low": 100.0,
            "ci_high": 100.0,
        }
    )
    fig = plot_allocation(yields, labels={"long": "long", "short": "short"}, output_dir=tmp_path)
    assert len(fig.axes) == 1
    axis = fig.axes[0]
    assert axis.get_xlim() == (0.5, 0.9)
    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "linear"
    assert axis.get_ylim()[0] == 0
    assert axis.get_ylim()[1] < 1.0
    assert len(axis.lines) == 2
    assert not axis.collections
    assert all(line.get_xdata().tolist() == [0.5, 0.9] for line in axis.lines)
    assert all(line.get_ydata()[-1] == 0 for line in axis.lines)
    assert (tmp_path / "pgd_allocation_value.pdf").exists()
    assert (tmp_path / "pgd_allocation_value.png").exists()
    plt.close(fig)


def test_hit_increase_is_compute_adjusted_and_excluded_targets_are_empty():
    yields = [
        {
            "target": "A",
            "arm": a,
            "threshold": 0.65,
            "hits": hits,
            "gpu_hours": hours,
            "hits_per_gpu_hour": hits / hours,
        }
        for a, hits, hours in (("long", 10, 10), ("short", 20, 40), ("biohub", 5, 20))
    ]
    table = hit_increase_table(yields, targets=["A", "missing"], threshold=0.65)
    assert table[0]["short_vs_long_increase_pct"] == -50
    assert table[0]["short_vs_biohub_increase_pct"] == 100
    assert table[1]["included"] is False
    assert table[1]["short_hits"] is None
    assert table[1]["short_vs_long_increase_pct"] is None


def test_hit_increase_leaves_zero_baseline_percentage_undefined():
    yields = [
        {
            "target": "A",
            "arm": a,
            "threshold": 0.65,
            "hits": hits,
            "gpu_hours": 1.0,
            "hits_per_gpu_hour": float(hits),
        }
        for a, hits in (("long", 0), ("short", 1), ("biohub", 0))
    ]
    table = hit_increase_table(yields, targets=["A"], threshold=0.65)
    assert table[0]["short_vs_long_increase_pct"] is None


def test_latex_table_transposes_targets_and_bolds_winners_including_ties(tmp_path):
    yields = [
        {
            "target": t,
            "arm": a,
            "threshold": 0.65,
            "hits": rate,
            "gpu_hours": 1.0,
            "hits_per_gpu_hour": rate,
        }
        for t, rates in (("A", (1.0, 0.5, 0.25)), ("B", (1.0, 2.0, 2.0)))
        for a, rate in zip(("long", "short", "biohub"), rates, strict=True)
    ]
    table = hit_increase_table(yields, targets=["A", "B", "missing"], threshold=0.65)
    write_hit_table(table, tmp_path, threshold=0.65)
    tex = (tmp_path / "hit_increase_0p65.tex").read_text()
    assert "Method & A & B & missing" in tex
    assert r"Logit GD, 150 steps & 0.25 & \textbf{2.00} & --" in tex
    assert r"PGD, 150 steps & \textbf{1.00} & 1.00 & --" in tex
    assert r"PGD, 75 steps + iPTM & 0.50 & \textbf{2.00} & --" in tex
    assert tex.count(r"\textbf{") == 3
    assert r"-50.0\% & +100.0\% & --" in tex


def test_global_survivor_counts_can_differ_between_seed_shards():
    pools = {
        "A": {
            arm: [
                {
                    "seed": seed,
                    "scores": np.asarray(scores),
                    "sequences": [f"{seed}-{i}" for i in range(len(scores))],
                    "design_gpu_hours": 1.0,
                    "ranking_gpu_hours": 2.0,
                    "global_selection": arm == "short",
                }
                for seed, scores in enumerate([[0.9], [0.5, 0.7, 0.8]], 1)
            ]
            for arm in ("long", "short")
        }
    }
    yields, endpoints, compute = summarize(
        pools, np.array([0.5]), top_fraction=0.5, n_bootstrap=0, analysis_seed=1
    )
    assert all(row["hits"] == 3 and row["gpu_hours"] == 6 for row in yields)
    assert all(row["mean"] == pytest.approx(0.85) for row in endpoints)
    assert all(row["ranked"] == 4 and row["shards"] == 2 for row in compute)
    with pytest.raises(ValueError, match="bootstrap is invalid"):
        summarize(pools, np.array([0.5]), top_fraction=0.5, n_bootstrap=10, analysis_seed=1)


def test_screen_compute_charges_discarded_candidates_and_validates_sequences(tmp_path):
    from binderopt.ranking.screening import _digest

    config = {"profile": {"resources": {"gpus_per_task": 1}}}
    designs = [
        {
            "design_index": i,
            "design_key": [i],
            "sequence": f"A{i}",
            "optimizer_sequence": f"B{i}",
            "ipsae_min": 0.8 - i * 0.2,
        }
        for i in range(2)
    ]
    mpnn = {
        "base_seed": 1,
        "config": config,
        "provenance": {},
        "shard": {
            "designs": designs,
            "timing": {"wall_time_s": 3600, "task_wall_time_s": 3600},
        },
    }
    digest = _digest(mpnn)
    screen = {
        "base_seed": 1,
        "config": config,
        "provenance": {},
        "mpnn_sha256": digest,
        "designs": designs,
        "wall_time_s": 1800,
    }
    manifest = {
        "provenance": {},
        "candidates": [
            dict(d, base_seed=1, selected=i == 0, mpnn_sha256=digest) for i, d in enumerate(designs)
        ],
    }
    for name, record in (("mpnn", mpnn), ("screening", screen)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "seed1.json").write_text(json.dumps(record))
    screen_hours = _ANALYSIS["_screen_hours"]
    assert screen_hours(tmp_path, 1, manifest) == (1.0, 0.5)
    screen["designs"][1]["sequence"] = "changed"
    (tmp_path / "screening/seed1.json").write_text(json.dumps(screen))
    with pytest.raises(ValueError, match="identity differs"):
        screen_hours(tmp_path, 1, manifest)


def test_reduced_pool_matches_paired_control_keys_without_selecting_on_scores():
    def cell(scores, keys):
        return {
            "scores": np.asarray(scores),
            "design_keys": [[k] for k in keys],
            "design_indices": list(range(len(keys))),
            "sequences": [str(k) for k in keys],
            "design_gpu_hours": 2.0,
            "ranking_gpu_hours": 3.0,
        }

    pools = {
        "A": {
            "short": [cell([0.8, 0.9], [8, 9])],
            "long": [cell([0.99, 0.1, 0.2], [3, 1, 2])],
            "biohub": [cell([0.01, 0.9, 0.8], [3, 1, 2])],
        }
    }
    coverage = [{"target": "A", "included": True}]
    _ANALYSIS["match_pool_sizes"](pools, coverage)
    for arm in ("long", "biohub"):
        record = pools["A"][arm][0]
        assert record["design_keys"] == [[1], [2]]
        assert record["design_indices"] == [1, 2]
        assert record["measured_designs"] == 3
        assert record["design_gpu_hours"] + record["ranking_gpu_hours"] == 5
    assert pools["A"]["long"][0]["scores"].tolist() == [0.1, 0.2]
    assert coverage[0]["ranked_per_arm"] == 2


def test_pooled_unranked_costs_use_relocated_source(allocation, tmp_path, monkeypatch):
    roots, kwargs = allocation
    kwargs["targets"] = ["target"]
    kwargs["seeds"] = [7, 8]
    kwargs["ranking_recipe"]["seed_offsets"] = [10, 11]
    for directory in roots.values():
        cell_path = directory / "target_pgd_seed7/result.json"
        cell_result = json.loads(cell_path.read_text())
        for design in cell_result["designs"]:
            design["ranking"]["base_seed"] = 1
        cell_path.write_text(json.dumps(cell_result))
    root = roots["short"] / "target"
    root.mkdir()
    old_cell = roots["short"] / "target_pgd_seed7"
    old_cell.rename(root / old_cell.name)
    path = root / old_cell.name / "result.json"
    result = json.loads(path.read_text())
    result["source_config"] = copy.deepcopy(result["config"])
    result["pool_config"] = {"ranking": kwargs["ranking_recipe"], "source_seeds": [7, 8],
                             "ranking_seed": 1}
    path.write_text(json.dumps(result))
    candidates = [
        {"base_seed": seed, "design_index": index, "ipsae_min": 1 - position / 10,
         "selected": position < 4, "selection_rank": position + 1}
        for position, (seed, index) in enumerate(
            (seed, index) for seed in [7, 8] for index in range(4)
        )
    ]
    (root / "selection.json").write_text(json.dumps({
        "selected_count": 4, "candidate_count": 8, "candidates": candidates,
    }))
    (root / "mpnn").mkdir()
    saved_source = tmp_path / "old_location"
    (root / "mpnn/seed8.json").write_text(json.dumps({
        "config": {"source_dir": str(saved_source)},
    }))
    source_root = tmp_path / "relocated"
    source_cell = source_root / "target_pgd_seed8"
    source_cell.mkdir(parents=True)
    source = {"design_timing": {"process_wall_time_s": 3600, "gpu_count": 2}}
    (source_cell / "result.json").write_text(json.dumps(source))
    monkeypatch.setitem(load_allocation.__globals__, "verify_ranked_cell", lambda *args: None)
    monkeypatch.setitem(load_allocation.__globals__, "_screen_hours", lambda *args: (0.25, 0.5))
    pools, coverage = load_allocation(
        [roots["long"]], roots["short"], source_root=source_root, **kwargs
    )
    short = pools["target"]["short"][0]
    assert short["design_gpu_hours"] == 4  # Ranked shard plus unranked source generation.
    assert short["ranking_gpu_hours"] == 2  # Full ranking plus both shards' MPNN/screening.
    assert short["unranked_source_seeds"] == [8]
    assert coverage[0]["included_seeds"] == [7]
    assert json.loads((root / "mpnn/seed8.json").read_text())["config"]["source_dir"] == str(
        saved_source
    )
    with pytest.raises(FileNotFoundError):
        load_allocation([roots["long"]], roots["short"], **kwargs)
