"""Atomic final cell outputs with generation-stable identity."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_state(repo: Path) -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_design_references(
    out_dir: Path,
    designs: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Atomically write reference PDBs and attach their provenance to designs."""
    if len(designs) != len(references):
        raise ValueError("one reference set is required for every completed design")
    published_designs: list[dict[str, Any]] = []
    for design, reference_set in zip(designs, references, strict=True):
        design_index = int(design["design_index"])
        model_records: dict[str, dict[str, Any]] = {}
        for reference in reference_set["models"]:
            record = dict(reference)
            structure_path = (
                out_dir
                / "design_references"
                / f"design_{design_index:03d}"
                / f"{record['model_name']}.pdb"
            )
            atomic_text(structure_path, str(record.pop("_structure_text")))
            record["structure"] = str(structure_path.relative_to(out_dir))
            model_records[str(record["model_name"])] = record
        selected_name = str(reference_set["selected_model"])
        selected = model_records[selected_name]
        published_designs.append(
            dict(design)
            | {
                "design_references": model_records,
                "selected_reference": {
                    "model_name": selected_name,
                    "checkpoint": selected["checkpoint"],
                    "protein_iptm": selected["protein_iptm"],
                    "structure": selected["structure"],
                },
            }
        )
    return published_designs


def write_design_outputs(
    out_dir: Path,
    *,
    identity: dict[str, Any],
    config: dict[str, Any],
    git_sha: str,
    hydra_overrides: list[str],
    designs: list[dict[str, Any]],
    pssms: list[np.ndarray],
    trajectory: list[dict[str, Any]],
    references: list[dict[str, Any]],
    process_started: float,
    started_at_utc: str,
    setup_wall_time_s: float,
    optimization_wall_time_s: float,
    reference_prediction_wall_time_s: float,
    gpu_count: int,
) -> dict[str, Any]:
    """Write one complete design cell after all batches succeed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"design output already exists: {result_path}")
    if len(designs) != len(pssms) or len(designs) != len(references):
        raise ValueError("one PSSM and reference set are required for every completed design")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")

    indices = [int(item["design_index"]) for item in designs]
    if indices != list(range(len(designs))):
        raise ValueError("design indices must be contiguous and generation ordered")
    binder_length = int(identity["binder_length"])
    arrays = [np.asarray(pssm, dtype=np.float32) for pssm in pssms]
    if any(array.shape != (binder_length, 20) for array in arrays):
        raise ValueError(f"PSSMs must have shape ({binder_length}, 20)")

    if arrays:
        atomic_npz(
            out_dir / "pssms.npz",
            pssm=np.stack(arrays),
            design_index=np.asarray(indices, dtype=np.int64),
        )

    trajectory_pssms = [np.asarray(point["pssm"], dtype=np.float32) for point in trajectory]
    if any(array.shape != (binder_length, 20) for array in trajectory_pssms):
        raise ValueError(f"trajectory PSSMs must have shape ({binder_length}, 20)")
    if trajectory_pssms:
        atomic_npz(
            out_dir / "opt_trajectory_pssms.npz",
            pssm=np.stack(trajectory_pssms),
            design_index=np.asarray(
                [int(point["design_index"]) for point in trajectory], dtype=np.int64
            ),
            step=np.asarray([int(point["step"]) for point in trajectory], dtype=np.int64),
        )
    atomic_json(
        out_dir / "opt_trajectory.json",
        {"trajectory": [{k: v for k, v in point.items() if k != "pssm"} for point in trajectory]},
    )
    published_designs = write_design_references(out_dir, designs, references)
    atomic_text(
        out_dir / "optimizer_designs.fasta",
        fasta_text(published_designs, sequence_field="optimizer_sequence"),
    )
    completed = time.perf_counter()
    result = {
        **identity,
        "num_designs": len(designs),
        "design_timing": {
            "started_at_utc": started_at_utc,
            "completed_at_utc": utc_now(),
            "process_wall_time_s": round(completed - process_started, 3),
            "setup_wall_time_s": round(setup_wall_time_s, 3),
            "optimization_wall_time_s": round(optimization_wall_time_s, 3),
            "reference_prediction_wall_time_s": round(reference_prediction_wall_time_s, 3),
            "gpu_count": gpu_count,
        },
        "git_sha": git_sha,
        "hydra_overrides": hydra_overrides,
        "config": config,
        "designs": published_designs,
    }
    atomic_json(result_path, result)
    return result


def fasta_text(designs: list[dict[str, Any]], *, sequence_field: str = "sequence") -> str:
    records = []
    for design in designs:
        rank_loss = design.get("rank_loss")
        rank = "pending" if rank_loss is None else f"{rank_loss:.6f}"
        records.append(
            f">design_{design['design_index']:03d} rank_loss={rank}\n{design[sequence_field]}\n"
        )
    return "".join(records)
