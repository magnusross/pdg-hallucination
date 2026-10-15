"""Paired optimizer analysis for complete ranked experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from binderopt.outputs import atomic_json, atomic_text, utc_now
from binderopt.ranking.cells import design_cell_result, discover_design_cells
from binderopt.ranking.composite import compute_composite_ranking

# The primary endpoint is deliberately kept separate from the analysis-only
# composite.  The latter is target-cohort normalized by binderopt.ranking.composite.
ENDPOINTS = (
    ("composite_score", "top_fraction_mean", True),
    ("ranking_score", "top_fraction_mean", True),
    ("mean_winning_dockq", "top_fraction_mean", True),
    ("mean_winning_rmsd", "mean", False),
    ("argmax_loss", "mean", False),
    ("optimizer_esmc_perplexity", "mean", False),
    ("mpnn_esmc_perplexity", "mean", False),
)
NATURALNESS_METRICS = (
    ("optimizer_esmc_perplexity", "optimizer sequence ESMC PPL"),
    ("mpnn_esmc_perplexity", "final sequence ESMC PPL"),
)
# Publication plotting vocabulary. Internal optimizer and target identifiers are
# retained in every table and analysis artifact; only figure labels are changed.
PLOT_COLORS = {"biohub": "#CC6677", "pgd": "#332288"}
PLOT_LABELS = {"biohub": "Logit GD", "pgd": "PGD"}
TARGET_LABELS = {
    "4ZQK": "PD-L1",
    "BBF14": "BBF-14",
    "BHRF1": "BHRF1",
    "IL7Ra": r"IL-7R$\alpha$",
    "MBP": "MBP",
    "RBX1": "RBX1",
    "TNFa": r"TNF-$\alpha$",
    "TREM2": "TREM2",
    "TrkA": "TrkA",
    "VEGFA": "VEGF-A",
}
TARGET_ORDER = (
    "4ZQK", "IL7Ra", "TREM2", "RBX1", "TrkA",
    "VEGFA", "TNFa", "BHRF1", "MBP", "BBF14",
)


def _target_order(values: Iterable[str]) -> list[str]:
    available = set(values)
    return [value for value in TARGET_ORDER if value in available] + sorted(
        available - set(TARGET_ORDER)
    )


def _plot_colors(optimizers: Sequence[str]) -> dict[str, str]:
    fallback = ("#117733", "#88CCEE", "#AA4499", "#DDCC77")
    return {
        name: PLOT_COLORS.get(name, fallback[index % len(fallback)])
        for index, name in enumerate(optimizers)
    }


def _plot_label(name: str) -> str:
    return PLOT_LABELS.get(name, name.replace("_", " ").title())


def _publication_style(plt: Any) -> None:
    from matplotlib import font_manager

    font = Path(__file__).resolve().parent / "fonts" / "lmsans10-regular.otf"
    family = "DejaVu Sans"
    if font.exists():
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )



def _finite(value: Any, *, field: str, cell: Path) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite {field} in {cell}")
    return result


def _design_key(value: Any, *, cell: Path) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"ranked design has no recorded design_key: {cell}")
    try:
        return tuple(int(word) for word in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"ranked design has invalid design_key: {cell}") from error


def _optimizer_id(result: Mapping[str, Any]) -> str:
    optimizer = result.get("config", {}).get("optimizer", {})
    return str(optimizer.get("optimizer_id") or result["optimizer_kind"])


def _cell_compute(result: Mapping[str, Any], *, cell: Path) -> tuple[float, float]:
    """Return allocated GPU-hours for the completed design and ranking tasks."""
    try:
        design = result["design_timing"]
        ranking = result["ranking_timing"]
        design_gpus = int(design["gpu_count"])
        ranking_gpus = int(result["config"]["profile"]["resources"]["gpus_per_task"])
        if design_gpus < 1 or ranking_gpus < 1:
            raise ValueError("GPU counts must be positive")
        design_seconds = _finite(design["process_wall_time_s"], field="design wall time", cell=cell)
        ranking_seconds = _finite(ranking["cell_wall_time_s"], field="ranking wall time", cell=cell)
        if design_seconds < 0 or ranking_seconds < 0:
            raise ValueError("wall times must be non-negative")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid compute timing in {cell}: {error}") from error
    return design_seconds * design_gpus / 3600, ranking_seconds * ranking_gpus / 3600


def summarize_compute(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sum allocated GPU-hours across selected cells, by target and overall."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell["target"])].append(cell)
    output = []
    for target, selected in [*sorted(grouped.items()), ("Total", cells)]:
        design_hours = math.fsum(float(cell["design_gpu_hours"]) for cell in selected)
        ranking_hours = math.fsum(float(cell["ranking_gpu_hours"]) for cell in selected)
        output.append(
            {
                "target": target,
                "cells": len(selected),
                "designs": sum(int(cell["n_designs"]) for cell in selected),
                "design_gpu_hours": design_hours,
                "ranking_gpu_hours": ranking_hours,
                "total_gpu_hours": design_hours + ranking_hours,
            }
        )
    return output


def _validated_model_metrics(
    design: Mapping[str, Any], *, cell: Path
) -> tuple[tuple[str, ...], float, float, float | None]:
    try:
        models = design["ranking"]["models"]
        model_names = tuple(sorted(models))
    except (KeyError, TypeError) as error:
        raise ValueError(f"ranked design has no judge measurements: {cell}") from error
    if not model_names:
        raise ValueError(f"ranked design has no judge measurements: {cell}")

    ipsae_values = []
    dockq_values = []
    rmsd_values = []
    for name in model_names:
        model = models[name]
        try:
            winning_seed = int(model["winning_seed"])
            seed_scores = model["seed_scores"]
            winners = [item for item in seed_scores if int(item["seed"]) == winning_seed]
            if len(winners) != 1:
                raise ValueError
            winner = winners[0]
            best_ipsae = _finite(model["best_ipsae_min"], field="best_ipsae_min", cell=cell)
            winning_dockq = _finite(model["winning_dockq"], field="winning_dockq", cell=cell)
            if not np.isclose(best_ipsae, float(winner["ipsae_min"])):
                raise ValueError
            if not np.isclose(winning_dockq, float(winner["dockq"])):
                raise ValueError
            expected = min(
                seed_scores, key=lambda item: (-float(item["ipsae_min"]), int(item["seed"]))
            )
            if int(expected["seed"]) != winning_seed:
                raise ValueError
            rmsd = winner.get("target_aligned_binder_backbone_rmsd")
            if rmsd is not None:
                rmsd_values.append(
                    _finite(rmsd, field="target_aligned_binder_backbone_rmsd", cell=cell)
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"judge {name} does not record DockQ from its iPsae-winning seed: {cell}"
            ) from error
        ipsae_values.append(best_ipsae)
        dockq_values.append(winning_dockq)
    mean_rmsd = float(np.mean(rmsd_values)) if len(rmsd_values) == len(model_names) else None
    return model_names, float(np.mean(ipsae_values)), float(np.mean(dockq_values)), mean_rmsd


def load_optimizer_experiment(
    experiment_dirs: Path | Sequence[Path],
    *,
    optimizers: Sequence[str] | None = None,
    exclude_targets: Sequence[str] = (),
    ipsae_weight: float,
    dockq_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], tuple[str, ...]]:
    """Load paired optimizer arms, validate ranking semantics, and attach composites."""
    if isinstance(experiment_dirs, (str, Path)):
        resolved_inputs = (Path(experiment_dirs).resolve(),)
    else:
        resolved_inputs = tuple(Path(path).resolve() for path in experiment_dirs)
    if not resolved_inputs:
        raise ValueError("optimizer analysis requires at least one experiment directory")
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("optimizer analysis input directories must be unique")
    cell_results = [
        (experiment_dir, cell_dir, design_cell_result(cell_dir))
        for experiment_dir in resolved_inputs
        for cell_dir in discover_design_cells(experiment_dir, allow_mixed_experiments=True)
    ]
    excluded = set(exclude_targets)
    cell_results = [
        (experiment_dir, cell_dir, result)
        for experiment_dir, cell_dir, result in cell_results
        if str(result["target_id"]) not in excluded
    ]
    discovered = {_optimizer_id(result) for _, _, result in cell_results}
    optimizer_names = tuple(optimizers) if optimizers is not None else tuple(sorted(discovered))
    if len(optimizer_names) < 2 or len(set(optimizer_names)) != len(optimizer_names):
        raise ValueError("optimizer comparison requires at least two unique optimizer names")
    missing = set(optimizer_names) - discovered
    if missing:
        raise ValueError(f"unknown optimizer arm(s): {sorted(missing)}")
    selected = set(optimizer_names)
    cell_results = [
        (experiment_dir, cell_dir, result)
        for experiment_dir, cell_dir, result in cell_results
        if _optimizer_id(result) in selected
    ]

    composite = compute_composite_ranking(
        resolved_inputs,
        ipsae_weight=ipsae_weight,
        dockq_weight=dockq_weight,
        exclude_targets=excluded,
        optimizers=optimizer_names,
    )
    composite_by_identity = {
        (
            target,
            item["experiment_id"],
            item["input"],
            item["cell"],
            int(item["design_index"]),
        ): item
        for target, cohort in composite["targets"].items()
        for item in cohort["designs"]
    }

    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for experiment_dir, cell_dir, result in cell_results:
        optimizer = _optimizer_id(result)
        design_gpu_hours, ranking_gpu_hours = _cell_compute(result, cell=cell_dir)
        relative_cell = str(cell_dir.relative_to(experiment_dir))
        experiment_id = str(result["experiment_id"])
        target = str(result["target_id"])
        cells.append(
            {
                "experiment_id": experiment_id,
                "input": str(experiment_dir),
                "cell": relative_cell,
                "target": target,
                "optimizer": optimizer,
                "base_seed": int(result["base_seed"]),
                "n_designs": len(result["designs"]),
                "design_gpu_hours": design_gpu_hours,
                "ranking_gpu_hours": ranking_gpu_hours,
                "ranking_recipe": str(result["ranking_model_id"]),
                "git_sha": str(result.get("git_sha", "unknown")),
            }
        )
        for design in result["designs"]:
            design_index = int(design["design_index"])
            model_names, mean_ipsae, mean_dockq, mean_rmsd = _validated_model_metrics(
                design, cell=cell_dir
            )
            ranking_score = _finite(design["ranking_score"], field="ranking_score", cell=cell_dir)
            if not np.isclose(ranking_score, mean_ipsae):
                raise ValueError(f"ranking_score is not mean best-over-seed iPsae: {cell_dir}")
            if not np.isclose(float(design["ipsae_ranking_score"]), mean_ipsae):
                raise ValueError(f"ipsae_ranking_score identity mismatch: {cell_dir}")
            if not np.isclose(float(design["rank_loss"]), -mean_ipsae):
                raise ValueError(f"rank_loss identity mismatch: {cell_dir}")
            composite_item = composite_by_identity[
                (target, experiment_id, str(experiment_dir), relative_cell, design_index)
            ]
            mpnn = design.get("mpnn", {})
            optimizer_perplexity = _finite(
                design["optimizer_esmc_perplexity"],
                field="optimizer_esmc_perplexity",
                cell=cell_dir,
            )
            mpnn_perplexity = _finite(
                mpnn["esmc_perplexity"], field="mpnn.esmc_perplexity", cell=cell_dir
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "input": str(experiment_dir),
                    "cell": relative_cell,
                    "target": target,
                    "optimizer": optimizer,
                    "base_seed": int(result["base_seed"]),
                    "design_index": design_index,
                    "design_key": _design_key(design.get("design_key"), cell=cell_dir),
                    "sequence": str(design["sequence"]),
                    "optimizer_sequence": str(design["optimizer_sequence"]),
                    "ranking_models": model_names,
                    "ranking_score": mean_ipsae,
                    "mean_winning_dockq": mean_dockq,
                    "composite_score": float(composite_item["composite_score"]),
                    "composite_rank": int(composite_item["composite_rank"]),
                    "mean_winning_rmsd": mean_rmsd,
                    "argmax_loss": _finite(
                        design["argmax_loss"], field="argmax_loss", cell=cell_dir
                    ),
                    "optimizer_esmc_perplexity": optimizer_perplexity,
                    "optimizer_esmc_log_perplexity": float(
                        design.get("optimizer_esmc_log_perplexity", np.log(optimizer_perplexity))
                    ),
                    "mpnn_esmc_perplexity": mpnn_perplexity,
                    "mpnn_esmc_log_perplexity": float(
                        mpnn.get("esmc_log_perplexity", np.log(mpnn_perplexity))
                    ),
                }
            )

    identities: dict[tuple[str, tuple[int, ...]], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = (str(row["target"]), row["design_key"])
        optimizer = str(row["optimizer"])
        if optimizer in identities[identity]:
            raise ValueError(f"duplicate design_key within optimizer arm: {identity}")
        identities[identity][optimizer] = row
    for identity, arms in identities.items():
        if set(arms) != set(optimizer_names):
            raise ValueError(f"unpaired design_key across optimizer arms: {identity}")
        recorded_identities = {(arm["base_seed"], arm["design_index"]) for arm in arms.values()}
        if len(recorded_identities) != 1:
            raise ValueError(f"paired design_key has inconsistent recorded identity: {identity}")
        judge_sets = {arm["ranking_models"] for arm in arms.values()}
        if len(judge_sets) != 1:
            raise ValueError(f"paired optimizer arms use different judge sets: {identity}")
    return rows, cells, composite, optimizer_names


def _rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode()).digest()
    offset = int.from_bytes(digest[:8], "little")
    return np.random.default_rng((seed + offset) % (2**64))


def _estimate(values: np.ndarray, statistic: str, top_fraction: float) -> tuple[float, int]:
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("statistics require a non-empty finite sample")
    if statistic == "mean":
        return float(values.mean()), len(values)
    if statistic != "top_fraction_mean":
        raise ValueError(f"unknown statistic: {statistic}")
    count = max(1, int(np.ceil(top_fraction * len(values))))
    return float(np.partition(values, -count)[-count:].mean()), count


def _bootstrap_estimates(
    values: np.ndarray,
    *,
    statistic: str,
    top_fraction: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    sampled = values[indices]
    if statistic == "mean":
        return sampled.mean(axis=1)
    if statistic != "top_fraction_mean":
        raise ValueError(f"unknown statistic: {statistic}")
    count = max(1, int(np.ceil(top_fraction * len(values))))
    return np.partition(sampled, -count, axis=1)[:, -count:].mean(axis=1)


def _paired_values(
    rows: Sequence[Mapping[str, Any]], *, target: str, metric: str, optimizers: Sequence[str]
) -> tuple[np.ndarray, ...]:
    pair_map: dict[tuple[int, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if str(row["target"]) == target:
            value = row.get(metric)
            if value is not None and np.isfinite(float(value)):
                pair_map[row["design_key"]][str(row["optimizer"])] = float(value)
    ordered = []
    for key in sorted(pair_map):
        if not set(optimizers).issubset(pair_map[key]):
            raise ValueError(f"metric {metric} is not paired for {target}, design_key={key}")
        ordered.append(pair_map[key])
    if not ordered:
        raise ValueError(f"no finite {metric} values for target {target}")
    return tuple(
        np.asarray([item[name] for item in ordered], dtype=np.float64) for name in optimizers
    )


def _paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    statistic: str,
    top_fraction: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(0, len(values_a), size=(n_bootstrap, len(values_a)))
    sampled_a = values_a[indices]
    sampled_b = values_b[indices]
    if statistic == "mean":
        return sampled_b.mean(axis=1) - sampled_a.mean(axis=1)
    count = max(1, int(np.ceil(top_fraction * len(values_a))))
    return np.partition(sampled_b, -count, axis=1)[:, -count:].mean(axis=1) - np.partition(
        sampled_a, -count, axis=1
    )[:, -count:].mean(axis=1)


def _mark_significant_best(
    summaries: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> None:
    grouped_summaries: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_comparisons: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        grouped_summaries[(str(item["target"]), str(item["metric"]))].append(item)
        item["significant_best"] = False
    for item in comparisons:
        grouped_comparisons[(str(item["target"]), str(item["metric"]))].append(item)
    for key, items in grouped_summaries.items():
        higher = bool(items[0]["higher_is_better"])
        best_value = (max if higher else min)(float(item["estimate"]) for item in items)
        candidates = [item for item in items if np.isclose(float(item["estimate"]), best_value)]
        pair_items = grouped_comparisons[key]
        for candidate in candidates:
            opponents = [item["optimizer"] for item in items if item is not candidate]
            wins = True
            for opponent in opponents:
                contrast = next(
                    (
                        item
                        for item in pair_items
                        if {item["optimizer_a"], item["optimizer_b"]}
                        == {candidate["optimizer"], opponent}
                    ),
                    None,
                )
                if contrast is None:
                    wins = False
                    break
                if candidate["optimizer"] == contrast["optimizer_b"]:
                    low = float(contrast["familywise_better_ci_low"])
                else:
                    low = float(-contrast["familywise_better_ci_high"])
                if low <= 0:
                    wins = False
                    break
            candidate["significant_best"] = wins


def summarize_endpoints(
    rows: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    top_fraction: float,
    n_bootstrap: int,
    analysis_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate summaries and all paired optimizer effects."""
    if len(optimizers) < 2 or len(set(optimizers)) != len(optimizers):
        raise ValueError("optimizer comparison requires at least two unique optimizer names")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    targets = sorted({str(row["target"]) for row in rows})
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for metric, statistic, higher_is_better in ENDPOINTS:
        if any(row.get(metric) is None for row in rows):
            continue
        for target in targets:
            metric_values = []
            for optimizer in optimizers:
                values = np.asarray(
                    [
                        row[metric]
                        for row in rows
                        if row["target"] == target and row["optimizer"] == optimizer
                    ],
                    dtype=np.float64,
                )
                estimate, n_used = _estimate(values, statistic, top_fraction)
                boot = _bootstrap_estimates(
                    values,
                    statistic=statistic,
                    top_fraction=top_fraction,
                    n_bootstrap=n_bootstrap,
                    rng=_rng(analysis_seed, f"summary:{metric}:{target}:{optimizer}"),
                )
                low, high = np.quantile(boot, [0.025, 0.975])
                metric_values.extend(values.tolist())
                summaries.append(
                    {
                        "metric": metric,
                        "statistic": statistic,
                        "target": target,
                        "optimizer": optimizer,
                        "n": len(values),
                        "n_used": n_used,
                        "estimate": estimate,
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "median": float(np.median(values)),
                        "best": float(values.max() if higher_is_better else values.min()),
                        "higher_is_better": higher_is_better,
                    }
                )
            target_sd = float(np.std(np.asarray(metric_values, dtype=np.float64), ddof=0))
            for optimizer_a, optimizer_b in combinations(optimizers, 2):
                values_a, values_b = _paired_values(
                    rows, target=target, metric=metric, optimizers=(optimizer_a, optimizer_b)
                )
                estimate_a, _ = _estimate(values_a, statistic, top_fraction)
                estimate_b, _ = _estimate(values_b, statistic, top_fraction)
                differences = _paired_bootstrap(
                    values_a,
                    values_b,
                    statistic=statistic,
                    top_fraction=top_fraction,
                    n_bootstrap=n_bootstrap,
                    rng=_rng(
                        analysis_seed, f"paired:{metric}:{target}:{optimizer_a}:{optimizer_b}"
                    ),
                )
                low, high = np.quantile(differences, [0.025, 0.975])
                better_estimate = estimate_b - estimate_a
                better_low, better_high = float(low), float(high)
                if not higher_is_better:
                    better_estimate = -better_estimate
                    better_low, better_high = -float(high), -float(low)
                normalized = better_estimate / target_sd if target_sd else 0.0
                normalized_low = better_low / target_sd if target_sd else 0.0
                normalized_high = better_high / target_sd if target_sd else 0.0
                comparisons.append(
                    {
                        "metric": metric,
                        "statistic": statistic,
                        "target": target,
                        "n_pairs": len(values_a),
                        "optimizer_a": optimizer_a,
                        "optimizer_b": optimizer_b,
                        "optimizer_a_estimate": estimate_a,
                        "optimizer_b_estimate": estimate_b,
                        "optimizer_b_minus_a": estimate_b - estimate_a,
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "higher_is_better": higher_is_better,
                        "target_sd": target_sd,
                        "better_effect": better_estimate,
                        "better_ci_low": better_low,
                        "better_ci_high": better_high,
                        "effect_sd_units": normalized,
                        "effect_sd_ci_low": normalized_low,
                        "effect_sd_ci_high": normalized_high,
                        "favored": None,
                        "conclusion": "inconclusive",
                    }
                )

    # Mark effects as significant using a Bonferroni familywise interval within
    # each target/metric set of optimizer comparisons.
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_key[(str(item["target"]), str(item["metric"]))].append(item)
    for key, items in by_key.items():
        alpha = 0.05 / len(items)
        q_low, q_high = alpha / 2, 1 - alpha / 2
        for item in items:
            values_a, values_b = _paired_values(
                rows,
                target=key[0],
                metric=key[1],
                optimizers=(item["optimizer_a"], item["optimizer_b"]),
            )
            differences = _paired_bootstrap(
                values_a,
                values_b,
                statistic=item["statistic"],
                top_fraction=top_fraction,
                n_bootstrap=n_bootstrap,
                rng=_rng(
                    analysis_seed,
                    f"familywise:{key[1]}:{key[0]}:{item['optimizer_a']}:{item['optimizer_b']}",
                ),
            )
            raw_low, raw_high = np.quantile(differences, [q_low, q_high])
            better_low, better_high = float(raw_low), float(raw_high)
            if not item["higher_is_better"]:
                better_low, better_high = -float(raw_high), -float(raw_low)
            item["familywise_ci_low"] = float(raw_low)
            item["familywise_ci_high"] = float(raw_high)
            item["familywise_better_ci_low"] = better_low
            item["familywise_better_ci_high"] = better_high
            target_sd = float(item["target_sd"])
            item["familywise_effect_sd_ci_low"] = better_low / target_sd if target_sd else 0.0
            item["familywise_effect_sd_ci_high"] = better_high / target_sd if target_sd else 0.0
            if better_low > 0:
                item["favored"] = (
                    item["optimizer_b"] if item["higher_is_better"] else item["optimizer_a"]
                )
                item["conclusion"] = f"{item['favored']} favored (familywise CI excludes zero)"
            elif better_high < 0:
                item["favored"] = (
                    item["optimizer_a"] if item["higher_is_better"] else item["optimizer_b"]
                )
                item["conclusion"] = f"{item['favored']} favored (familywise CI excludes zero)"

    _mark_significant_best(summaries, comparisons)
    return summaries, comparisons


def summarize_naturalness(
    rows: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    n_bootstrap: int,
    analysis_seed: int,
) -> list[dict[str, Any]]:
    """Summarize perplexity as geometric means because the stored quantity is log PPL."""
    output: list[dict[str, Any]] = []
    for metric, label in NATURALNESS_METRICS:
        for target in sorted({str(row["target"]) for row in rows}):
            for optimizer in optimizers:
                values = np.asarray(
                    [
                        row[metric]
                        for row in rows
                        if row["target"] == target and row["optimizer"] == optimizer
                    ],
                    dtype=np.float64,
                )
                if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
                    raise ValueError(f"invalid perplexity values for {target}/{optimizer}/{metric}")
                estimate = float(np.exp(np.mean(np.log(values))))
                indices = _rng(
                    analysis_seed, f"naturalness:{metric}:{target}:{optimizer}"
                ).integers(0, len(values), size=(n_bootstrap, len(values)))
                boot = np.exp(np.mean(np.log(values[indices]), axis=1))
                low, high = np.quantile(boot, [0.025, 0.975])
                output.append(
                    {
                        "metric": metric,
                        "label": label,
                        "target": target,
                        "optimizer": optimizer,
                        "n": len(values),
                        "estimate": estimate,
                        "ci_low": float(low),
                        "ci_high": float(high),
                    }
                )
    return output


def threshold_counts(
    rows: Sequence[Mapping[str, Any]], *, optimizers: Sequence[str], threshold: float
) -> list[dict[str, Any]]:
    output = []
    for target in sorted({str(row["target"]) for row in rows}):
        for optimizer in optimizers:
            values = np.asarray(
                [
                    row["ranking_score"]
                    for row in rows
                    if row["target"] == target and row["optimizer"] == optimizer
                ],
                dtype=np.float64,
            )
            count = int(np.count_nonzero(values >= threshold))
            output.append(
                {
                    "target": target,
                    "optimizer": optimizer,
                    "threshold": threshold,
                    "count": count,
                    "n": len(values),
                    "fraction": count / len(values),
                }
            )
    return output


def _csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    materialized = list(rows)
    if not materialized:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
    writer.writeheader()
    for row in materialized:
        writer.writerow(
            {
                key: json.dumps(value) if isinstance(value, (tuple, list, dict)) else value
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _interval(
    item: Mapping[str, Any],
    digits: int = 3,
    *,
    star: bool = False,
    dagger: bool = False,
) -> str:
    suffix = (" *" if star else "") + (" †" if dagger else "")
    return (
        f"{_fmt(item['estimate'], digits)} "
        f"[{_fmt(item['ci_low'], digits)}, {_fmt(item['ci_high'], digits)}]{suffix}"
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_compute_table(output_dir: Path, compute: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        (
            item["target"],
            item["cells"],
            item["designs"],
            _fmt(item["design_gpu_hours"], 2),
            _fmt(item["ranking_gpu_hours"], 2),
            _fmt(item["total_gpu_hours"], 2),
        )
        for item in compute
    ]
    atomic_text(
        output_dir / "compute_summary.md",
        "# Experiment compute\n\n"
        "Allocated GPU-hours from completed selected cells: design process wall time times "
        "design GPU count, plus ranking cell wall time times configured GPUs per task. "
        "Includes task setup "
        "and CPU work within those tasks; excludes queue time, failed attempts, and other "
        "optimizer recipes.\n\n"
        + _markdown_table(
            ("Target", "Cells", "Designs", "Design GPU-h", "Ranking GPU-h", "Total GPU-h"),
            rows,
        ),
    )


def _summary_lookup(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(item["target"]), str(item["optimizer"]), str(item["metric"])): item
        for item in summaries
    }


def _write_markdown_tables(
    output_dir: Path,
    *,
    summaries: Sequence[Mapping[str, Any]],
    naturalness: Sequence[Mapping[str, Any]],
    rmsd_summaries: Sequence[Mapping[str, Any]],
    counts: Sequence[Mapping[str, Any]],
    optimizers: Sequence[str],
) -> None:
    lookup = _summary_lookup(summaries)
    top_rows = []
    targets = sorted({str(item["target"]) for item in summaries})
    for target in targets:
        for optimizer in optimizers:
            top_rows.append(
                [
                    target,
                    optimizer,
                    lookup.get((target, optimizer, "ranking_score"), {}).get("n", "—"),
                    _interval(
                        lookup[(target, optimizer, "ranking_score")],
                        3,
                        star=bool(
                            lookup[(target, optimizer, "ranking_score")].get("significant_best")
                        ),
                        dagger=bool(
                            lookup[(target, optimizer, "ranking_score")].get(
                                "significant_vs_reference"
                            )
                        ),
                    )
                    if (target, optimizer, "ranking_score") in lookup
                    else "—",
                    _interval(
                        lookup[(target, optimizer, "composite_score")],
                        3,
                        star=bool(
                            lookup[(target, optimizer, "composite_score")].get("significant_best")
                        ),
                        dagger=bool(
                            lookup[(target, optimizer, "composite_score")].get(
                                "significant_vs_reference"
                            )
                        ),
                    )
                    if (target, optimizer, "composite_score") in lookup
                    else "—",
                    _interval(
                        lookup[(target, optimizer, "mean_winning_dockq")],
                        3,
                        star=bool(
                            lookup[(target, optimizer, "mean_winning_dockq")].get(
                                "significant_best"
                            )
                        ),
                        dagger=bool(
                            lookup[(target, optimizer, "mean_winning_dockq")].get(
                                "significant_vs_reference"
                            )
                        ),
                    )
                    if (target, optimizer, "mean_winning_dockq") in lookup
                    else "—",
                ]
            )
    top_text = (
        "# Top-10% score summary\n\n"
        "Values are top-10% means with paired-bootstrap 95% CIs. `†` marks a challenger "
        "significantly better than a selected reference; `*` marks a unique best that beats "
        "every other optimizer. Significance uses Bonferroni-corrected familywise intervals "
        "within each target and metric.\n\n"
        + _markdown_table(
            ("Target", "Optimizer", "N", "Ranking iPsae", "Composite", "DockQ"), top_rows
        )
    )
    atomic_text(output_dir / "top10_summary.md", top_text)

    natural_rows = []
    natural_lookup = {
        (str(item["target"]), str(item["optimizer"]), str(item["metric"])): item
        for item in naturalness
    }
    for target in sorted({str(item["target"]) for item in naturalness}):
        for optimizer in optimizers:
            natural_rows.append(
                [
                    target,
                    optimizer,
                    _interval(natural_lookup[(target, optimizer, "optimizer_esmc_perplexity")], 2),
                    _interval(natural_lookup[(target, optimizer, "mpnn_esmc_perplexity")], 2),
                ]
            )
    atomic_text(
        output_dir / "naturalness_summary.md",
        "# ESMC perplexity summary\n\n"
        "Geometric means with bootstrap 95% CIs; lower is better. "
        "No SolubleMPNN likelihood score is saved in these result files; "
        "only ESMC perplexity is available.\n\n"
        + _markdown_table(
            ("Target", "Optimizer", "Optimizer sequence", "Final MPNN sequence"), natural_rows
        ),
    )

    count_rows = [
        [
            item["target"],
            item["optimizer"],
            item["count"],
            item["n"],
            f"{100 * item['fraction']:.1f}%",
        ]
        for item in counts
    ]
    threshold = counts[0]["threshold"] if counts else None
    atomic_text(
        output_dir / "ranking_threshold_counts.md",
        f"# Ranking-score threshold counts\n\n"
        f"Number of designs with ranking score ≥ {threshold:.2f}.\n\n"
        + _markdown_table(("Target", "Optimizer", "Count", "N", "Fraction"), count_rows),
    )

    if rmsd_summaries:
        rmsd_rows = []
        for item in rmsd_summaries:
            rmsd_rows.append(
                [
                    item["target"],
                    item["optimizer"],
                    item["n"],
                    _interval(
                        {
                            "estimate": item["mean"],
                            "ci_low": item["mean_ci_low"],
                            "ci_high": item["mean_ci_high"],
                        },
                        2,
                    ),
                    _fmt(item["median"], 2),
                    item["top_n"],
                    _interval(
                        {
                            "estimate": item["top_mean"],
                            "ci_low": item["top_mean_ci_low"],
                            "ci_high": item["top_mean_ci_high"],
                        },
                        2,
                    ),
                    _fmt(item["top_median"], 2),
                ]
            )
        top_percent = 100 * float(rmsd_summaries[0]["top_fraction"])
        rmsd_text = (
            "# Redesign backbone consistency\n\n"
            "Each design's value is the mean target-aligned binder backbone RMSD across "
            "the three ranking judges, using each judge's iPsae-winning seed. Alignment "
            "uses each target's configured self-consistency residues (the full target unless "
            "a target-specific range is configured), and error uses the whole binder backbone "
            "(N/CA/C/O), not only interface residues. The population is strongly right-skewed, "
            f"The top subset is the top {top_percent:g}% by ranking score. Mean intervals "
            "are bootstrap 95% CIs.\n\n"
            + _markdown_table(
                (
                    "Target",
                    "Optimizer",
                    "N",
                    "All mean RMSD (Å)",
                    "All median",
                    "Top N",
                    "Top mean RMSD (Å)",
                    "Top median",
                ),
                rmsd_rows,
            )
        )
    else:
        rmsd_text = (
            "# Redesign backbone consistency\n\n"
            "No target-aligned RMSD fields were present in the input results.\n"
        )
    atomic_text(output_dir / "rmsd_summary.md", rmsd_text)


def _values(
    rows: Sequence[Mapping[str, Any]], target: str, optimizer: str, metric: str
) -> np.ndarray:
    return np.sort(
        np.asarray(
            [
                row[metric]
                for row in rows
                if row["target"] == target and row["optimizer"] == optimizer
            ],
            dtype=np.float64,
        )
    )


def _plot_pool_maxima(
    rows: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    output_dir: Path,
    axis: Any | None = None,
) -> str | None:
    """Plot equal-target expected maxima of finite observed design pools."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def expected_maximum(values: np.ndarray, pool_size: int) -> float:
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        total = len(ordered)
        probability = pool_size / total
        result = probability * ordered[-1]
        for rank in range(total, pool_size, -1):
            probability *= (rank - pool_size) / (rank - 1)
            result += probability * ordered[rank - 2]
        return float(result)

    _publication_style(plt)
    targets = _target_order(str(row["target"]) for row in rows)
    maximum = min(
        sum(row["target"] == target and row["optimizer"] == optimizer for row in rows)
        for target in targets for optimizer in optimizers
    )
    pool_sizes = [size for size in (32, 64, 128, 256, 512, 1024) if size <= maximum]
    if not pool_sizes:
        pool_sizes = [maximum]
    standalone = axis is None
    if standalone:
        fig, axis = plt.subplots(figsize=(3.15, 2.12))
        fig.subplots_adjust(left=.19, right=.97, bottom=.20, top=.84)
    for optimizer in optimizers:
        curves = np.asarray([
            [expected_maximum(_values(rows, target, optimizer, "ranking_score"), size)
             for size in pool_sizes]
            for target in targets
        ])
        axis.plot(
            pool_sizes, curves.mean(axis=0), color=_plot_colors(optimizers)[optimizer],
            ls=(0, (3, 1.5)) if optimizer == "biohub" else "-",
            linewidth=1.4, marker="o", markersize=2.3, label=_plot_label(optimizer),
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(pool_sizes)
    axis.set_xticklabels([str(size) for size in pool_sizes])
    axis.set_xlabel("Designs evaluated")
    axis.set_ylabel("Mean pool-maximum ipSAE")
    axis.grid(axis="y", color=".9", linewidth=.4)
    if not standalone:
        return None
    axis.legend(loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=2,
                frameon=False, handlelength=1.8)
    path = output_dir / "max_ipsae_vs_pool_size.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _plot_survival(
    rows: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    output_dir: Path,
    n_bootstrap: int,
    analysis_seed: int,
    tail_max_fraction: float,
) -> list[str]:
    """Plot full and upper-tail primary ipSAE survival with exact bootstrap marginals."""
    return [
        _plot_survival_curve(
            rows, optimizers=optimizers, output_dir=output_dir, tail_max_fraction=fraction,
        )
        for fraction in (tail_max_fraction, None)
    ]


def _plot_survival_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    output_dir: Path,
    tail_max_fraction: float | None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, NullLocator, PercentFormatter
    from scipy.stats import binom

    _publication_style(plt)
    targets = _target_order(str(row["target"]) for row in rows)
    columns = min(5, len(targets))
    figure_rows = math.ceil(len(targets) / columns)
    height = 1.45 * figure_rows
    fig, axes = plt.subplots(
        figure_rows, columns, figsize=(6.5, height), squeeze=False, sharey=True
    )
    fig.subplots_adjust(left=.072, right=.992, bottom=.15, top=.86, hspace=.52, wspace=.19)
    colors = _plot_colors(optimizers)
    for axis, target in zip(axes.flat, targets, strict=False):
        arm_values = [_values(rows, target, arm, "ranking_score") for arm in optimizers]
        lower = (0.0 if tail_max_fraction is None else
                 float(np.quantile(np.concatenate(arm_values), 1 - tail_max_fraction)))
        upper = (1.0 if tail_max_fraction is None else
                 max(float(values.max()) for values in arm_values))
        grid = np.unique(np.r_[lower, *[v[v >= lower] for v in arm_values], upper])
        for arm, values in zip(optimizers, arm_values, strict=True):
            probability = (
                len(values) - np.searchsorted(values, grid, side="left")
            ) / len(values)
            low = binom.ppf(.025, len(values), probability) / len(values)
            high = binom.ppf(.975, len(values), probability) / len(values)
            axis.fill_between(
                grid, np.maximum(low, .0001), np.maximum(high, .0001),
                step="pre", color=colors[arm], alpha=.10, lw=0,
            )
            axis.step(
                grid, probability, where="pre", color=colors[arm],
                ls=(0, (3, 1.5)) if arm == "biohub" else "-", lw=.95,
            )
        axis.set_xlim(lower, upper)
        if tail_max_fraction is None:
            axis.set_ylim(0, 1)
            axis.set_yticks([0, .5, 1])
        else:
            axis.set_ylim(.005, .45)
            axis.set_yscale("log")
            axis.set_yticks([.01, .05, .2])
        axis.yaxis.set_major_formatter(PercentFormatter(1, 0))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_locator(MaxNLocator(3, min_n_ticks=2))
        axis.set_title(TARGET_LABELS.get(target, target), pad=2)
        axis.grid(axis="y", color=".91", lw=.35)
    for axis in axes.flat[len(targets):]:
        axis.set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend(
        [Line2D([], [], color=colors[arm],
                ls=(0, (3, 1.5)) if arm == "biohub" else "-", lw=1.2)
         for arm in optimizers],
        [_plot_label(arm) for arm in optimizers],
        loc="upper center", bbox_to_anchor=(.54, .995), ncol=len(optimizers),
        frameon=False, handlelength=2.1, columnspacing=1.2, borderaxespad=0,
    )
    fig.text(.54, .015, "ipSAE", ha="center")
    fig.text(.008, .5, "Fraction at or above score", va="center", rotation=90)
    suffix = "_tail" if tail_max_fraction is not None else ""
    path = output_dir / f"ranking_score_survival{suffix}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _plot_effects(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    contrast_pairs: Sequence[tuple[str, str]],
    top_fraction: float,
) -> str:
    """Plot paired effects for the two report endpoints."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _publication_style(plt)
    metrics = (("ranking_score", "ipSAE"), ("mean_winning_dockq", "scDockQ"))
    targets = _target_order(str(item["target"]) for item in comparisons)
    series = list(contrast_pairs)
    colors = {pair: _plot_colors((pair[1],))[pair[1]] for pair in series}
    markers = ("o", "s", "D", "^")
    offsets = (
        np.linspace(-0.15, 0.15, len(series))
        if len(series) > 1
        else np.zeros(1)
    )

    def oriented(
        item: Mapping[str, Any], reference: str, challenger: str
    ) -> tuple[float, float, float]:
        estimate = float(item["effect_sd_units"])
        low = float(item["familywise_effect_sd_ci_low"])
        high = float(item["familywise_effect_sd_ci_high"])
        if item["optimizer_a"] == reference and item["optimizer_b"] == challenger:
            return estimate, low, high
        return -estimate, -high, -low

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.25, 4.35), squeeze=False, sharey=True
    )
    for axis, (metric, title) in zip(axes[0], metrics, strict=True):
        for target_index in range(len(targets)):
            if target_index % 2 == 0:
                axis.axhspan(
                    target_index - 0.5,
                    target_index + 0.5,
                    color="0.96",
                    zorder=0,
                )
        for series_index, (reference, challenger) in enumerate(series):
            for target_index, target in enumerate(targets):
                item = next(
                    value
                    for value in comparisons
                    if value["metric"] == metric
                    and value["target"] == target
                    and {value["optimizer_a"], value["optimizer_b"]}
                    == {reference, challenger}
                )
                estimate, low, high = oriented(item, reference, challenger)
                significant = low > 0 or high < 0
                axis.errorbar(
                    estimate,
                    target_index + offsets[series_index],
                    xerr=[[estimate - low], [high - estimate]],
                    fmt=markers[series_index % len(markers)],
                    color=colors[(reference, challenger)],
                    markerfacecolor=(
                        colors[(reference, challenger)] if significant else "white"
                    ),
                    markersize=4.5,
                    capsize=2.1,
                    linewidth=0.95,
                    label=(
                        f"{_plot_label(challenger)} vs {_plot_label(reference)}"
                        if target_index == 0
                        else None
                    ),
                    zorder=2,
                )
        axis.axvline(0, color="0.35", linestyle="-", linewidth=0.9)
        axis.set_yticks(
            range(len(targets)),
            [TARGET_LABELS.get(target, target) for target in targets],
        )
        axis.set_ylim(len(targets) - 0.5, -0.5)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.18, linewidth=0.5)
    for axis in axes[0][1:]:
        axis.tick_params(labelleft=False)
    fig.supxlabel(
        f"Paired effect on top {100 * top_fraction:g}% mean "
        "(target SD units; PGD better →)",
        y=0.015,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.45, w_pad=0.8)
    path = output_dir / "paired_effects.pdf"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _plot_tail_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    output_dir: Path,
    axis: Any | None = None,
    tail_max_fraction: float,
) -> str | None:
    """Plot equal-target aggregate ipSAE means across selection fractions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _publication_style(plt)
    colors = _plot_colors(optimizers)
    targets = _target_order(str(row["target"]) for row in rows)
    fractions = np.linspace(0.01, tail_max_fraction, 24)
    standalone = axis is None
    if standalone:
        fig, axis = plt.subplots(figsize=(3.15, 2.12))
        fig.subplots_adjust(left=.19, right=.97, bottom=.20, top=.84)
    for optimizer in optimizers:
        curves = np.asarray([
            [_estimate(_values(rows, target, optimizer, "ranking_score"),
                       "top_fraction_mean", float(fraction))[0]
             for fraction in fractions]
            for target in targets
        ])
        axis.plot(
            100 * fractions, curves.mean(axis=0), color=colors[optimizer],
            ls=(0, (3, 1.5)) if optimizer == "biohub" else "-",
            linewidth=1.4, label=_plot_label(optimizer),
        )
    axis.axvline(10, color=".6", linestyle=":", linewidth=.6)
    axis.set_xlim(1, 100 * tail_max_fraction)
    axis.set_xticks([value for value in (1, 5, 10, 15, 20)
                     if value <= 100 * tail_max_fraction])
    axis.set_xlabel("Selected top fraction (%)")
    axis.set_ylabel("Mean selected ipSAE")
    axis.grid(axis="y", color=".9", linewidth=.4)
    if not standalone:
        return None
    axis.legend(loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=2,
                frameon=False, handlelength=1.8)
    path = output_dir / "tail_mean_vs_percentile.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path.name



def _plot_selection_sensitivity(
    rows: Sequence[Mapping[str, Any]], *, optimizers: Sequence[str],
    output_dir: Path, tail_max_fraction: float,
) -> str:
    """Stack selection-fraction and finite-pool maxima at wrapfigure size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    _publication_style(plt)
    fig, axes = plt.subplots(2, 1, figsize=(2.2, 2.45))
    fig.subplots_adjust(left=.28, right=.97, bottom=.15, top=.86, hspace=.85)
    _plot_tail_means(
        rows, optimizers=optimizers, output_dir=output_dir,
        tail_max_fraction=tail_max_fraction, axis=axes[0],
    )
    _plot_pool_maxima(rows, optimizers=optimizers, output_dir=output_dir, axis=axes[1])
    axes[0].set_xticks([value for value in (1, 10, 20) if value <= 100 * tail_max_fraction])
    sizes = axes[1].get_xticks()
    axes[1].set_xticks(np.unique([sizes[0], sizes[len(sizes) // 2], sizes[-1]]))
    for ax, panel, ylabel in zip(
        axes, ("(a)", "(b)"), ("Mean selected ipSAE", "Expected maximum ipSAE"), strict=True,
    ):
        ax.set_ylabel(ylabel)
        ax.set_title(panel, loc="left", pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.grid(False)
        for line in ax.lines:
            if line.get_label() in [_plot_label(arm) for arm in optimizers]:
                line.set_linewidth(1.1)
    colors = _plot_colors(optimizers)
    fig.legend(
        [Line2D([], [], color=colors[arm], ls=(0, (3, 1.5)) if arm == "biohub" else "-",
                lw=1.1) for arm in optimizers],
        [_plot_label(arm) for arm in optimizers],
        loc="upper center", bbox_to_anchor=(.60, .995), ncol=len(optimizers),
        frameon=False, handlelength=1.3, columnspacing=.9, borderaxespad=0,
    )
    path = output_dir / "selection_sensitivity.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path.name

def _rmsd_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    top_fraction: float,
    n_bootstrap: int,
    analysis_seed: int,
) -> list[dict[str, Any]]:
    if any(row.get("mean_winning_rmsd") is None for row in rows):
        return []

    def interval(values: np.ndarray, *, statistic: str, label: str) -> tuple[float, float, float]:
        indices = _rng(analysis_seed, label).integers(
            0, len(values), size=(n_bootstrap, len(values))
        )
        sampled = values[indices]
        if statistic == "mean":
            estimate = float(values.mean())
            boot = sampled.mean(axis=1)
        else:
            estimate = float(np.median(values))
            boot = np.median(sampled, axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        return estimate, float(low), float(high)

    output = []
    for target in sorted({str(row["target"]) for row in rows}):
        for optimizer in optimizers:
            selected = [
                row for row in rows if row["target"] == target and row["optimizer"] == optimizer
            ]
            selected.sort(key=lambda row: float(row["ranking_score"]), reverse=True)
            values = np.asarray([row["mean_winning_rmsd"] for row in selected], dtype=np.float64)
            top_n = max(1, int(np.ceil(top_fraction * len(values))))
            top_values = values[:top_n]
            mean, mean_low, mean_high = interval(
                values, statistic="mean", label=f"rmsd-mean:{target}:{optimizer}"
            )
            median, median_low, median_high = interval(
                values, statistic="median", label=f"rmsd-median:{target}:{optimizer}"
            )
            top_mean, top_mean_low, top_mean_high = interval(
                top_values, statistic="mean", label=f"rmsd-top-mean:{target}:{optimizer}"
            )
            top_median, top_median_low, top_median_high = interval(
                top_values, statistic="median", label=f"rmsd-top-median:{target}:{optimizer}"
            )
            output.append(
                {
                    "metric": "mean_winning_rmsd",
                    "target": target,
                    "optimizer": optimizer,
                    "n": len(values),
                    "mean": mean,
                    "mean_ci_low": mean_low,
                    "mean_ci_high": mean_high,
                    "median": median,
                    "median_ci_low": median_low,
                    "median_ci_high": median_high,
                    "p90": float(np.quantile(values, 0.9)),
                    "top_fraction": top_fraction,
                    "top_n": top_n,
                    "top_mean": top_mean,
                    "top_mean_ci_low": top_mean_low,
                    "top_mean_ci_high": top_mean_high,
                    "top_median": top_median,
                    "top_median_ci_low": top_median_low,
                    "top_median_ci_high": top_median_high,
                }
            )
    return output


def _resolve_contrasts(
    optimizer_names: Sequence[str],
    references: Sequence[str] | None,
    challengers: Sequence[str] | None,
) -> tuple[tuple[str, str], ...]:
    if references is None and challengers is None:
        return tuple(combinations(optimizer_names, 2))
    if not references or not challengers:
        raise ValueError("--references and --challengers must be supplied together")
    unknown = (set(references) | set(challengers)) - set(optimizer_names)
    if unknown:
        raise ValueError(f"unknown contrast optimizer(s): {sorted(unknown)}")
    if set(references) & set(challengers):
        raise ValueError("references and challengers must be disjoint")
    return tuple((reference, challenger) for reference in references for challenger in challengers)


def _write_report_endpoint_table(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    output_dir: Path,
) -> None:
    """Transpose the two-arm endpoint summary for the report."""
    if len(optimizers) != 2:
        return
    targets = _target_order(str(row["target"]) for row in rows)
    names = [TARGET_LABELS.get(target, target).replace("\u03b1", r"$\alpha$") for target in targets]
    estimates = {
        (str(row["metric"]), str(row["target"]), str(row["optimizer"])): float(row["estimate"])
        for row in summaries
    }
    effects = {
        (str(row["metric"]), str(row["target"])): row
        for row in comparisons
        if {str(row["optimizer_a"]), str(row["optimizer_b"])} == set(optimizers)
    }
    lines = [
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{1.5pt}\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ll*{"
        + str(len(targets)) + r"}{r}@{}}",
        r"\toprule",
        "Measure & Method & " + " & ".join(names) + r" \\",
        r"\midrule",
    ]
    for metric, label in (
        ("ranking_score", r"Top-10\% ipSAE"),
        ("mean_winning_dockq", r"Top-10\% scDockQ"),
    ):
        for index, arm in enumerate(optimizers):
            values = []
            for target in targets:
                comparison = effects[(metric, target)]
                low = float(comparison["familywise_ci_low"])
                high = float(comparison["familywise_ci_high"])
                winner = (
                    str(comparison["optimizer_b"]) if low > 0
                    else str(comparison["optimizer_a"]) if high < 0
                    else None
                )
                value = f"{estimates[(metric, target, arm)]:.3f}"
                values.append(r"\textbf{" + value + "}" if arm == winner else value)
            lines.append((label if index == 0 else "") + " & "
                         + _plot_label(arm) + " & " + " & ".join(values) + r" \\")
        lines.append(r"\midrule")
    for index, arm in enumerate(optimizers):
        counts = [
            sum(str(row["target"]) == target and str(row["optimizer"]) == arm
                and float(row["ranking_score"]) > 0.65 for row in rows)
            for target in targets
        ]
        lines.append((r"ipSAE $>0.65$" if index == 0 else "") + " & "
                     + _plot_label(arm) + " & " + " & ".join(map(str, counts)) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}\endgroup", ""])
    atomic_text(output_dir / "report_endpoint_table.tex", "\n".join(lines))


def _write_report_median_table(
    summaries: Sequence[Mapping[str, Any]],
    *,
    optimizers: Sequence[str],
    output_dir: Path,
) -> None:
    """Write descriptive whole-pool medians, without significance marking."""
    targets = _target_order(str(row["target"]) for row in summaries)
    names = [TARGET_LABELS.get(target, target).replace("\u03b1", r"$\alpha$") for target in targets]
    medians = {
        (str(row["metric"]), str(row["target"]), str(row["optimizer"])): float(row["median"])
        for row in summaries
    }
    lines = [
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{1.5pt}\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ll*{"
        + str(len(targets)) + r"}{r}@{}}",
        r"\toprule",
        "Measure & Method & " + " & ".join(names) + r" \\",
    ]
    for metric, label in (
        ("ranking_score", "Median ipSAE"),
        ("mean_winning_dockq", "Median scDockQ"),
    ):
        lines.append(r"\midrule")
        for index, arm in enumerate(optimizers):
            values = [f"{medians[(metric, target, arm)]:.3f}" for target in targets]
            lines.append((label if index == 0 else "") + " & "
                         + _plot_label(arm) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}\endgroup", ""])
    atomic_text(output_dir / "report_median_table.tex", "\n".join(lines))


def _plot_report_clusters(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    optimizers: Sequence[str],
) -> str | None:
    """Render the previously computed cluster counts at wrapfigure size."""
    if tuple(optimizers) != ("biohub", "pgd"):
        return None
    summary_path = output_dir / "structural_uniqueness" / "summary.csv"
    if not summary_path.exists():
        return None
    with summary_path.open(newline="") as stream:
        cluster_rows = list(csv.DictReader(stream))
    targets = _target_order(str(row["target"]) for row in comparisons)
    counts = {
        (str(row["target"]), str(row["optimizer"])): int(row["n_clusters"])
        for row in cluster_rows
    }
    if set(counts) != {(target, arm) for target in targets for arm in optimizers}:
        raise ValueError("Structural cluster summary does not match the paired cohort")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    _publication_style(plt)
    fig, ax = plt.subplots(figsize=(2.2, 3.15))
    fig.subplots_adjust(left=.27, right=.98, bottom=.16, top=.90)
    y = np.arange(len(targets))
    ax.set_yticks(y, [TARGET_LABELS.get(target, target) for target in targets])
    ax.set_ylim(len(targets) - .5, -.65)
    for offset, arm in zip((-.18, .18), optimizers, strict=True):
        bars = ax.barh(
            y + offset,
            [counts[(target, arm)] for target in targets],
            height=.32, color=PLOT_COLORS[arm], label=_plot_label(arm),
        )
        ax.bar_label(bars, padding=2, fontsize=6.4)
    ax.set_xlabel("Clusters among 103 designs")
    ax.set_xlim(0, max(counts.values()) + 9)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.grid(axis="x", color=".9", lw=.4)
    ax.set_axisbelow(True)
    fig.legend(
        [Line2D([], [], color=PLOT_COLORS[arm], lw=3) for arm in optimizers],
        [_plot_label(arm) for arm in optimizers],
        ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(.99, .99),
        handlelength=1.2,
    )
    path = output_dir / "fold_seek_clusters.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _reduce_pssm_dynamics(
    pssms: np.ndarray,
    indices: np.ndarray,
    steps: np.ndarray,
    *,
    n_designs: int,
    n_steps: int,
    cell: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one cell, preserving consecutive steps within each design."""
    if (indices.shape != (n_designs * n_steps,) or steps.shape != indices.shape
            or pssms.ndim != 3 or len(pssms) != len(indices)):
        raise ValueError(f"Incomplete PSSM trajectory: {cell}")
    order = np.lexsort((steps, indices))
    if (not np.array_equal(indices[order], np.repeat(np.arange(n_designs), n_steps))
            or not np.array_equal(steps[order], np.tile(np.arange(n_steps), n_designs))):
        raise ValueError(f"Duplicate or missing PSSM trajectory point: {cell}")
    if (not np.isfinite(pssms).all() or np.any(pssms < 0)
            or not np.allclose(pssms.sum(-1), 1, atol=1e-5)):
        raise ValueError(f"Invalid probability simplex in PSSM trajectory: {cell}")
    active = (pssms[order] > 1e-3).reshape(n_designs, n_steps, *pssms.shape[1:])
    support = active.sum(-1).mean(-1)
    # A re-entry is an inactive coefficient becoming active on the next step.
    # Step one has no preceding saved PSSM, so its re-entry count is undefined.
    reentry = np.full((n_designs, n_steps), np.nan)
    reentry[:, 1:] = (active[:, 1:] & ~active[:, :-1]).sum(-1).mean(-1)
    return support, reentry


def _summarize_dynamics(
    cells: Sequence[Mapping[str, Any]], *, output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Reduce saved trajectories cell by cell; average designs within each target."""
    paths = [Path(str(cell["input"])) / str(cell["cell"]) for cell in cells]
    if not any((path / "opt_trajectory_pssms.npz").exists() for path in paths):
        return None
    totals: dict[tuple[str, str], dict[str, Any]] = {}
    endpoints: list[dict[str, Any]] = []
    schedule = None
    for cell, path in zip(cells, paths, strict=True):
        result = json.loads((path / "result.json").read_text())
        designs = {int(item["design_index"]): item for item in result["designs"]}
        n_designs = len(designs)
        n_steps = int(result["config"]["design"]["n_steps"])
        tail_start = n_steps - int(result["config"]["design"]["tail_select_steps"])
        if schedule is not None and schedule != (n_steps, tail_start):
            raise ValueError("Inconsistent optimisation schedules across dynamics cells")
        schedule = n_steps, tail_start
        if set(designs) != set(range(n_designs)):
            raise ValueError(f"Non-contiguous design indices: {path}")
        objective = np.full((n_designs, n_steps), np.nan)
        points = json.loads((path / "opt_trajectory.json").read_text())["trajectory"]
        for point in points:
            index, step = int(point["design_index"]), int(point["step"])
            if not (0 <= index < n_designs and 0 <= step < n_steps):
                raise ValueError(f"Out-of-range objective trajectory point: {path}")
            if np.isfinite(objective[index, step]):
                raise ValueError(f"Duplicate objective trajectory point: {path}")
            objective[index, step] = _finite(point["objective"], field="objective", cell=path)
        if not np.isfinite(objective).all():
            raise ValueError(f"Incomplete objective trajectory: {path}")
        with np.load(path / "opt_trajectory_pssms.npz") as archive:
            pssms = archive["pssm"]
            indices, steps = archive["design_index"], archive["step"]
            support, reentry = _reduce_pssm_dynamics(
                pssms, indices, steps, n_designs=n_designs, n_steps=n_steps, cell=path,
            )
            with np.load(path / "pssms.npz") as selected:
                selected_pssms = dict(zip(selected["design_index"], selected["pssm"], strict=True))
            for index, design in designs.items():
                step = int(design["selected_step"])
                if not tail_start <= step < n_steps:
                    raise ValueError(f"Selected step outside tail window: {path}")
                row = np.flatnonzero((indices == index) & (steps == step))[0]
                if index not in selected_pssms or not np.allclose(
                    pssms[row], selected_pssms[index], atol=1e-6,
                ):
                    raise ValueError(f"Selected PSSM mismatch: {path}")
                soft_loss = _finite(design["design_objective"], field="soft loss", cell=path)
                if not np.isclose(soft_loss, objective[index, step]):
                    raise ValueError(f"Selected objective mismatch: {path}")
                hard_loss = _finite(design["argmax_loss"], field="argmax loss", cell=path)
                endpoints.append({
                    "target": cell["target"], "optimizer": cell["optimizer"],
                    "input": cell["input"], "cell": cell["cell"],
                    "base_seed": cell["base_seed"], "design_index": index,
                    "selected_step": step + 1, "soft_loss": soft_loss,
                    "argmax_loss": hard_loss, "argmax_minus_soft": hard_loss - soft_loss,
                    "selected_support": float(support[index, step]),
                })
            del pssms
        key = str(cell["target"]), str(cell["optimizer"])
        metrics = {
            "objective_change": (objective - objective[:, :1]).sum(0),
            "support": support.sum(0), "reentry": reentry.sum(0),
        }
        if key not in totals:
            totals[key] = {"n_designs": 0, **{metric: np.zeros(n_steps) for metric in metrics}}
        totals[key]["n_designs"] += n_designs
        for metric, values in metrics.items():
            totals[key][metric] += values
    records = [
        {"target": target, "optimizer": arm, "step": step + 1,
         "n_designs": total["n_designs"], "tail_start": schedule[1] + 1,
         **{metric: float(total[metric][step] / total["n_designs"])
            for metric in ("objective_change", "support", "reentry")}}
        for (target, arm), total in sorted(totals.items())
        for step in range(schedule[0])
    ]
    atomic_text(output_dir / "optimization_dynamics_curves.csv", _csv_text(records))
    atomic_text(output_dir / "optimization_dynamics_endpoints.csv", _csv_text(endpoints))
    atomic_json(output_dir / "optimization_dynamics.json", {
        "n_cells": len(cells), "n_designs": len(endpoints),
        "n_steps": schedule[0], "tail_start": schedule[1] + 1,
        "support_threshold": 1e-3,
        "reentry_definition": "choices moving from probability <=1e-3 to >1e-3 on the next step",
        "aggregation": "mean over designs within target, then equally weighted target means",
        "target_lines": "translucent lines show individual target means",
        "endpoint": "tail-selected iterate, not necessarily the final iterate",
        "loss_comparison": "soft and argmax losses use different stochastic model/dropout draws",
    })
    return records, endpoints


def _plot_report_dynamics(
    records: Sequence[Mapping[str, Any]], endpoints: Sequence[Mapping[str, Any]],
    *, output_dir: Path, optimizers: Sequence[str],
) -> str:
    """Four compact panels for objective, support, re-entry, and hardening."""
    targets = _target_order(str(row["target"]) for row in records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    _publication_style(plt)
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.50))
    fig.subplots_adjust(left=.075, right=.985, bottom=.28, top=.76, wspace=.60)
    colors = _plot_colors(optimizers)
    for ax, metric, title, ylabel in zip(
        axes[:3], ("objective_change", "support", "reentry"),
        ("(a) Loss", "(b) Active set", "(c) Re-entry"),
        ("Objective change", "Choices/position", "Revived choices/position"), strict=True,
    ):
        for arm in optimizers:
            curves = np.asarray([
                [float(row[metric]) for row in records
                 if row["target"] == target and row["optimizer"] == arm]
                for target in targets
            ])
            steps = np.arange(1, curves.shape[1] + 1)
            if metric == "reentry":
                curves, steps = curves[:, 1:], steps[1:]
            linestyle = (0, (3, 1.5)) if arm == "biohub" else "-"
            for curve in curves:
                ax.plot(steps, curve, color=colors[arm], ls=linestyle,
                        alpha=.20, lw=.55, zorder=1)
            ax.plot(steps, curves.mean(axis=0), color=colors[arm],
                    ls=linestyle, lw=1.1, zorder=3)
        n_steps = max(int(row["step"]) for row in records)
        ax.set_xlim(1, n_steps)
        ax.set_xticks([1, n_steps // 2, n_steps])
        ax.set_xlabel("Optimisation step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=4)
        ax.yaxis.set_major_locator(MaxNLocator(3))
    ax = axes[3]
    for arm in optimizers:
        values = np.asarray([
            [np.mean([float(row[metric]) for row in endpoints
                      if row["target"] == target and row["optimizer"] == arm])
             for metric in ("soft_loss", "argmax_loss")]
            for target in targets
        ])
        linestyle = (0, (3, 1.5)) if arm == "biohub" else "-"
        for target_values in values:
            ax.plot([0, 1], target_values, color=colors[arm], ls=linestyle,
                    alpha=.20, lw=.55, zorder=1)
        ax.plot([0, 1], values.mean(0), color=colors[arm], marker="o", markersize=2.5,
                ls=linestyle, lw=1.1, zorder=3)
    ax.set_xticks([0, 1], ["Soft", "Argmax"])
    ax.set_xlim(-.15, 1.15)
    ax.set_xlabel("Selected iterate")
    ax.set_ylabel("Objective")
    ax.set_title("(d) Discretisation", loc="left", pad=4)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    fig.legend(
        [Line2D([], [], color=colors[arm], ls=(0, (3, 1.5)) if arm == "biohub" else "-",
                lw=1.2) for arm in optimizers],
        [_plot_label(arm) for arm in optimizers],
        loc="upper center", ncol=len(optimizers), frameon=False, handlelength=1.7,
        bbox_to_anchor=(.5, 1.0), borderaxespad=0,
    )
    path = output_dir / "optimization_dynamics.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def run_analysis(
    experiment_dirs: Path | Sequence[Path],
    *,
    output_dir: Path,
    optimizers: Sequence[str] | None = None,
    exclude_targets: Sequence[str] = (),
    references: Sequence[str] | None = None,
    challengers: Sequence[str] | None = None,
    ipsae_weight: float,
    dockq_weight: float,
    top_fraction: float,
    n_bootstrap: int,
    analysis_seed: int,
    threshold: float = 0.85,
    tail_max_fraction: float = 0.20,
    make_plots: bool = True,
) -> dict[str, Any]:
    if not 0.01 <= tail_max_fraction <= 1:
        raise ValueError("tail_max_fraction must be in [0.01, 1]")
    rows, cells, composite, optimizer_names = load_optimizer_experiment(
        experiment_dirs,
        optimizers=optimizers,
        exclude_targets=exclude_targets,
        ipsae_weight=ipsae_weight,
        dockq_weight=dockq_weight,
    )
    contrast_pairs = _resolve_contrasts(optimizer_names, references, challengers)
    summaries, comparisons = summarize_endpoints(
        rows,
        optimizers=optimizer_names,
        top_fraction=top_fraction,
        n_bootstrap=n_bootstrap,
        analysis_seed=analysis_seed,
    )
    for summary in summaries:
        summary["significant_vs_reference"] = False
    for comparison in comparisons:
        selected_pair = next(
            (
                pair
                for pair in contrast_pairs
                if {comparison["optimizer_a"], comparison["optimizer_b"]} == set(pair)
            ),
            None,
        )
        comparison["selected_contrast"] = selected_pair is not None
        if selected_pair is None:
            continue
        reference, challenger = selected_pair
        if comparison["optimizer_a"] == reference:
            low = float(comparison["familywise_better_ci_low"])
        else:
            low = float(-comparison["familywise_better_ci_high"])
        if low > 0:
            selected_summary = next(
                item
                for item in summaries
                if item["metric"] == comparison["metric"]
                and item["target"] == comparison["target"]
                and item["optimizer"] == challenger
            )
            selected_summary["significant_vs_reference"] = True
    naturalness = summarize_naturalness(
        rows, optimizers=optimizer_names, n_bootstrap=n_bootstrap, analysis_seed=analysis_seed
    )
    rmsd_summaries = _rmsd_summaries(
        rows,
        optimizers=optimizer_names,
        top_fraction=top_fraction,
        n_bootstrap=n_bootstrap,
        analysis_seed=analysis_seed,
    )
    counts = threshold_counts(rows, optimizers=optimizer_names, threshold=threshold)
    compute = summarize_compute(cells)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "composite_ranking.json", composite)
    atomic_text(output_dir / "designs.csv", _csv_text(rows))
    atomic_text(output_dir / "cells.csv", _csv_text(cells))
    atomic_text(output_dir / "compute_summary.csv", _csv_text(compute))
    _write_compute_table(output_dir, compute)
    atomic_text(output_dir / "endpoint_summary.csv", _csv_text(summaries))
    atomic_text(output_dir / "paired_comparisons.csv", _csv_text(comparisons))
    _write_report_endpoint_table(
        rows, summaries, comparisons, optimizers=optimizer_names, output_dir=output_dir
    )
    _write_report_median_table(
        summaries, optimizers=optimizer_names, output_dir=output_dir
    )
    atomic_text(output_dir / "naturalness_summary.csv", _csv_text(naturalness))
    atomic_text(output_dir / "rmsd_summary.csv", _csv_text(rmsd_summaries))
    atomic_text(output_dir / "ranking_threshold_counts.csv", _csv_text(counts))
    _write_markdown_tables(
        output_dir,
        summaries=summaries,
        naturalness=naturalness,
        rmsd_summaries=rmsd_summaries,
        counts=counts,
        optimizers=optimizer_names,
    )

    plots: list[str] = []
    if make_plots:
        plots.extend(
            _plot_survival(
                rows,
                optimizers=optimizer_names,
                output_dir=output_dir,
                n_bootstrap=n_bootstrap,
                analysis_seed=analysis_seed,
                tail_max_fraction=tail_max_fraction,
            )
        )
        effects = _plot_effects(
            comparisons,
            output_dir=output_dir,
            contrast_pairs=contrast_pairs,
            top_fraction=top_fraction,
        )
        if effects:
            plots.append(effects)
        plots.append(
            _plot_selection_sensitivity(
                rows,
                optimizers=optimizer_names,
                output_dir=output_dir,
                tail_max_fraction=tail_max_fraction,
            )
        )
        clusters = _plot_report_clusters(
            comparisons, output_dir=output_dir, optimizers=optimizer_names
        )
        if clusters is not None:
            plots.append(clusters)
        dynamics = _summarize_dynamics(cells, output_dir=output_dir)
        if dynamics is not None:
            plots.append(_plot_report_dynamics(
                *dynamics, output_dir=output_dir, optimizers=optimizer_names,
            ))
        if (output_dir / "structural_uniqueness" / "structural_uniqueness.pdf").exists():
            plots.append("structural_uniqueness/structural_uniqueness.pdf")

    result = {
        "experiment_id": composite["experiment_id"],
        "experiment_ids": composite["experiment_ids"],
        "created_at_utc": utc_now(),
        "input": composite["inputs"][0] if len(composite["inputs"]) == 1 else None,
        "inputs": composite["inputs"],
        "optimizers": list(optimizer_names),
        "n_cells": len(cells),
        "n_designs": len(rows),
        "n_paired_initializations": len(rows) // len(optimizer_names),
        "total_gpu_hours": compute[-1]["total_gpu_hours"],
        "targets": sorted({str(row["target"]) for row in rows}),
        "excluded_targets": sorted(set(exclude_targets)),
        "composite_weights": {"ipsae": float(ipsae_weight), "dockq": float(dockq_weight)},
        "composite_formula": (
            "mean over judges of weighted target-cohort z(iPsae) and z(DockQ), "
            "using DockQ from each judge's iPsae-winning seed"
        ),
        "top_fraction": float(top_fraction),
        "tail_max_fraction": float(tail_max_fraction),
        "ranking_threshold": float(threshold),
        "n_bootstrap": int(n_bootstrap),
        "pool_max_n_bootstrap": 0,
        "analysis_seed": int(analysis_seed),
        "contrasts": [
            {"reference": reference, "challenger": challenger}
            for reference, challenger in contrast_pairs
        ],
        "effect_normalization": (
            "beneficial paired effect divided by target-wide population SD; "
            "positive means challenger is better"
        ),
        "significance": "Bonferroni-corrected paired bootstrap intervals within each target/metric",
        "mpnn_score_available": False,
        "rmsd_definition": (
            "align configured target self-consistency residues (full target by default) using "
            "backbone N/CA/C/O atoms, then calculate all-binder-backbone RMSD; average judge "
            "iPsae-winning predictions"
        ),
        "plots": plots,
    }
    atomic_json(output_dir / "analysis.json", result)
    return result


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="directories containing complete ranked cells"
    )
    parser.add_argument(
        "--references", nargs="+", help="reference optimizer arm(s) for selected effect contrasts"
    )
    parser.add_argument(
        "--challengers", nargs="+", help="challenger optimizer arm(s) for selected effect contrasts"
    )
    parser.add_argument(
        "--exclude-targets",
        nargs="+",
        default=(),
        help="target IDs to omit without moving their result cells",
    )
    parser.add_argument("--ipsae-weight", type=float, required=True)
    parser.add_argument("--dockq-weight", type=float, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--tail-max-fraction", type=float, default=0.20)
    parser.add_argument("--ranking-threshold", type=float, default=0.85)
    parser.add_argument("--n-bootstrap", type=int, default=5_000)
    parser.add_argument("--analysis-seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(arguments)
    if bool(args.references) != bool(args.challengers):
        parser.error("--references and --challengers must be supplied together")
    optimizers = (
        None
        if args.references is None
        else list(dict.fromkeys([*args.references, *args.challengers]))
    )
    if args.output is None and len(args.inputs) > 1:
        parser.error("--output is required when pooling multiple experiment directories")
    output = (
        args.output or Path(__file__).resolve().parent / "artifacts" / args.inputs[0].resolve().name
    )
    run_analysis(
        args.inputs,
        output_dir=output,
        optimizers=optimizers,
        exclude_targets=args.exclude_targets,
        references=args.references,
        challengers=args.challengers,
        ipsae_weight=args.ipsae_weight,
        dockq_weight=args.dockq_weight,
        top_fraction=args.top_fraction,
        tail_max_fraction=args.tail_max_fraction,
        threshold=args.ranking_threshold,
        n_bootstrap=args.n_bootstrap,
        analysis_seed=args.analysis_seed,
        make_plots=not args.no_plots,
    )
    print(output)


if __name__ == "__main__":
    main()
