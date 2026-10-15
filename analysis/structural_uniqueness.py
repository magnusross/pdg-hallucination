"""Cluster top-ranked binder structures after standard paired optimizer loading.

Run from the repository root with ``python -m analysis.structural_uniqueness``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analysis.optimizer_analysis import (
    PLOT_COLORS,
    TARGET_LABELS,
    _plot_label,
    _publication_style,
    _target_order,
    load_optimizer_experiment,
)

DEFAULT_TM_THRESHOLD = 0.6


def _design_id(row: Mapping[str, Any]) -> str:
    return (
        f"{row['target']}_{row['optimizer']}_seed{int(row['base_seed']):02d}"
        f"_design{int(row['design_index']):03d}"
    )


def select_top_designs(
    rows: Sequence[Mapping[str, Any]], *, top_fraction: float
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """Use the same per-target, per-arm top-fraction rule as optimizer_analysis."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    pools: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[(str(row["target"]), str(row["optimizer"]))].append(row)
    if not pools:
        raise ValueError("no ranked designs to cluster")
    return {
        pool: sorted(
            designs,
            key=lambda row: (-float(row["ranking_score"]), _design_id(row)),
        )[: max(1, math.ceil(top_fraction * len(designs)))]
        for pool, designs in sorted(pools.items())
    }


def write_binder_chain(source: Path, destination: Path, sequence: str) -> None:
    """Write chain A, the single binder chain in saved ranked complexes."""
    lines = []
    observed = set()
    seqres = []
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith("SEQRES") and line[11:12] == "A":
            seqres.extend(line[19:70].split())
            lines.append(line)
        elif line.startswith("ATOM  ") and line[21:22] == "A":
            if line[12:16].strip() == "CA":
                observed.add((line[22:27], line[17:20]))
            lines.append(line)
        elif line.startswith("TER   ") and line[21:22] == "A":
            lines.append(line)
    if not observed or len(observed) > len(sequence):
        raise ValueError(f"invalid binder chain A in {source}")
    if seqres and len(seqres) != len(sequence):
        raise ValueError(f"binder SEQRES length differs from final sequence: {source}")
    destination.write_text("".join(lines) + "END\n")


def _structure_for(row: Mapping[str, Any], cell_cache: dict[Path, dict]) -> Path:
    cell = Path(str(row["input"])) / str(row["cell"])
    if cell not in cell_cache:
        result = json.loads((cell / "result.json").read_text())
        cell_cache[cell] = {int(item["design_index"]): item for item in result["designs"]}
    design = cell_cache[cell][int(row["design_index"])]
    if tuple(design["design_key"]) != tuple(row["design_key"]):
        raise ValueError(f"design key mismatch in {cell}")
    if str(design["sequence"]) != str(row["sequence"]):
        raise ValueError(f"final sequence mismatch in {cell}")
    source = cell / str(design["structure"])
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def cluster_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    foldseek: Path,
    tm_threshold: float,
    cell_cache: dict[Path, dict],
) -> int:
    """Return the number of Foldseek representative-based clusters; keep no intermediates."""
    with tempfile.TemporaryDirectory(prefix="binderopt_foldseek_") as temporary:
        work = Path(temporary)
        binders = work / "binders"
        binders.mkdir()
        expected = set()
        for row in sorted(rows, key=_design_id):
            name = _design_id(row)
            if name in expected:
                raise ValueError(f"duplicate stable design ID: {name}")
            expected.add(name)
            write_binder_chain(
                _structure_for(row, cell_cache), binders / f"{name}.pdb", str(row["sequence"])
            )
        output = work / "cluster"
        command = [
            str(foldseek),
            "easy-cluster",
            str(binders),
            str(output),
            str(work / "tmp"),
            "--min-seq-id",
            "0",
            "-c",
            "0",
            "--cov-mode",
            "0",
            "--alignment-type",
            "1",
            "--tmscore-threshold-mode",
            "0",
            "--tmscore-threshold",
            str(tm_threshold),
            "--threads",
            "1",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"Foldseek failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
            )
        clusters = defaultdict(list)
        for line in (work / "cluster_cluster.tsv").read_text().splitlines():
            representative, member = line.split("\t")
            clusters[representative].append(member)
        actual = [member for members in clusters.values() for member in members]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Foldseek output does not contain every selected binder exactly once")
        return len(clusters)


def plot_summary(
    summaries: Sequence[Mapping[str, Any]], output: Path, optimizers: Sequence[str]
) -> None:
    """Compact paired horizontal bars for the selected structural clusters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_target: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in summaries:
        by_target[str(item["target"])][str(item["optimizer"])] = item
    if len(optimizers) != 2 or set(optimizers) != {
        str(item["optimizer"]) for item in summaries
    }:
        raise ValueError("the structural uniqueness plot requires two optimizers")

    _publication_style(plt)
    targets = _target_order(by_target)
    y = np.arange(len(targets), dtype=float)
    height = 0.34
    fig, axis = plt.subplots(figsize=(3.0, 4.35))
    for offset, optimizer in zip((-0.5, 0.5), optimizers, strict=True):
        values = [
            int(by_target[target][optimizer]["n_clusters"]) for target in targets
        ]
        bars = axis.barh(
            y + offset * height,
            values,
            height,
            color=PLOT_COLORS.get(optimizer, "0.5"),
            label=_plot_label(optimizer),
            zorder=2,
        )
        axis.bar_label(
            bars,
            labels=[str(value) for value in values],
            padding=2,
            fontsize=6.2,
            color="0.25",
        )
    maximum = max(int(item["n_clusters"]) for item in summaries)
    axis.set_xlim(0, maximum + 9)
    axis.set_yticks(
        y,
        [TARGET_LABELS.get(target, target) for target in targets],
    )
    axis.set_ylim(len(targets) - 0.5, -0.5)
    axis.set_xlabel("Foldseek clusters (103 selected designs)")
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        handlelength=1.4,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.45)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_analysis(
    inputs: Sequence[Path],
    *,
    output: Path,
    foldseek: Path,
    optimizers: Sequence[str] | None,
    exclude_targets: Sequence[str],
    ipsae_weight: float,
    dockq_weight: float,
    top_fraction: float,
    tm_threshold: float,
) -> list[dict[str, Any]]:
    if not 0 < tm_threshold <= 1:
        raise ValueError("tm_threshold must be in (0, 1]")
    rows, _cells, _composite, optimizer_names = load_optimizer_experiment(
        inputs,
        optimizers=optimizers,
        exclude_targets=exclude_targets,
        ipsae_weight=ipsae_weight,
        dockq_weight=dockq_weight,
    )
    if len(optimizer_names) != 2:
        raise ValueError("select exactly two optimizer arms for this paired plot")
    pools = select_top_designs(rows, top_fraction=top_fraction)
    foldseek_version = subprocess.check_output([str(foldseek), "version"], text=True).strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    summaries = []
    cell_cache: dict[Path, dict] = {}
    for (target, optimizer), selected in pools.items():
        n_clusters = cluster_pool(
            selected, foldseek=foldseek, tm_threshold=tm_threshold, cell_cache=cell_cache
        )
        n_designs = sum(row["target"] == target and row["optimizer"] == optimizer for row in rows)
        summaries.append(
            {
                "target": target,
                "optimizer": optimizer,
                "n_designs": n_designs,
                "n_selected": len(selected),
                "n_clusters": n_clusters,
                "cluster_fraction": n_clusters / len(selected),
            }
        )
        print(f"{target}/{optimizer}: {n_clusters}/{len(selected)} clusters", flush=True)
    with tempfile.TemporaryDirectory(prefix="structural_uniqueness_", dir=output.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        with (stage / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        plot_summary(summaries, stage / "structural_uniqueness.pdf", optimizer_names)
        method = (
            "# Structural diversity of top-ranked binders\n\n"
            "Ranked seed shards were pooled for each target and optimizer after validation "
            "of design-key pairing and judge scores. The highest primary `ranking_score` "
            f"designs were selected within each pool (top {100 * top_fraction:g}%, rounded up). "
            "The binder chain A was extracted from each saved final `design.structure` "
            "(ESMFold2 Full for these results).\n\n"
            f"Foldseek `easy-cluster` build `{foldseek_version}` used TM-score threshold "
            f"{tm_threshold:g}, alignment-type 1, alignment-length TM normalization, "
            "zero sequence-identity and coverage cutoffs, coverage mode 0, and one thread. "
            "No sequence preclustering or additional qualification filter was applied. "
            "Cluster counts are threshold-dependent similarities to representatives, "
            "not all-pairs distinctness or biological fold classes.\n"
        )
        (stage / "methods.md").write_text(method)
        stage.rename(output)
    return summaries


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="directories containing complete ranked cells"
    )
    parser.add_argument("--references", nargs="+", help="reference optimizer arm")
    parser.add_argument("--challengers", nargs="+", help="challenger optimizer arm")
    parser.add_argument(
        "--exclude-targets",
        nargs="+",
        default=(),
        help="target IDs to omit without moving result cells",
    )
    parser.add_argument("--ipsae-weight", type=float, required=True)
    parser.add_argument("--dockq-weight", type=float, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--tm-threshold", type=float, default=DEFAULT_TM_THRESHOLD)
    parser.add_argument("--foldseek", type=Path, default=Path("foldseek"))
    parser.add_argument("--output", type=Path)
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
        args.output
        or Path(__file__).resolve().parent
        / "artifacts"
        / args.inputs[0].name
        / "structural_uniqueness"
    )
    run_analysis(
        args.inputs,
        output=output,
        foldseek=args.foldseek,
        optimizers=optimizers,
        exclude_targets=args.exclude_targets,
        ipsae_weight=args.ipsae_weight,
        dockq_weight=args.dockq_weight,
        top_fraction=args.top_fraction,
        tm_threshold=args.tm_threshold,
    )
    print(output)


if __name__ == "__main__":
    main()
