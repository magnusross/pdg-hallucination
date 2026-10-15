"""Standard DockQ self-consistency scoring with positional residue matching."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

_DOCKQ_INPUT_IDS = itertools.count()


def _parse_pdb(source: str | Path) -> tuple[Any, ...]:
    """Load protein chains with DockQ's parser, excluding HETATM records."""
    from DockQ.parsers import PDBParser

    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    model = PDBParser(QUIET=True).get_structure(
        "self_consistency", StringIO(text), chains=[], parse_hetatms=False
    )
    chains = tuple(chain for chain in model if len(chain))
    if len(chains) < 2:
        raise ValueError("self-consistency scoring requires a binder and at least one target chain")
    return chains


def _target_permutations(target_sequences: Sequence[str]) -> list[tuple[int, ...]]:
    groups: dict[str, list[int]] = {}
    for index, sequence in enumerate(target_sequences):
        groups.setdefault(sequence, []).append(index)
    choices = [itertools.permutations(indices) for indices in groups.values()]
    mappings = []
    for selected in itertools.product(*choices):
        mapping = list(range(len(target_sequences)))
        for reference_indices, prediction_indices in zip(groups.values(), selected, strict=True):
            for reference_index, prediction_index in zip(
                reference_indices, prediction_indices, strict=True
            ):
                mapping[reference_index] = prediction_index
        mappings.append(tuple(mapping))
    return mappings


def _validate_layout(
    chains: Sequence[Any], binder_length: int, target_sequences: Sequence[str]
) -> None:
    expected = (binder_length, *(len(sequence) for sequence in target_sequences))
    observed = tuple(len(chain) for chain in chains)
    if observed != expected:
        raise ValueError(f"protein chain lengths differ: expected {expected}, observed {observed}")


def _normalize_target_residue_ranges(
    target_sequences: Sequence[str],
    target_residue_ranges: Sequence[Sequence[Sequence[int]]] | None,
) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    if target_residue_ranges is None:
        return None
    if len(target_residue_ranges) != len(target_sequences):
        raise ValueError("target residue ranges must have one entry per target chain")
    normalized: list[tuple[tuple[int, int], ...]] = []
    for chain_index, (sequence, values) in enumerate(
        zip(target_sequences, target_residue_ranges, strict=True)
    ):
        try:
            ranges = tuple((int(value[0]), int(value[1])) for value in values)
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(f"invalid target residue ranges for chain {chain_index}") from error
        if not ranges:
            raise ValueError(f"target residue ranges are empty for chain {chain_index}")
        previous_end = 0
        for start, end in ranges:
            if start <= previous_end or end < start or end > len(sequence):
                raise ValueError(
                    "target residue ranges must be ordered, non-overlapping, one-based, "
                    "inclusive, and within each target sequence"
                )
            previous_end = end
        normalized.append(ranges)
    return tuple(normalized)


def _select_residues(chain: Any, ranges: Sequence[tuple[int, int]]) -> Any:
    """Select one-based inclusive sequence positions while retaining chain order."""
    from Bio.PDB.Chain import Chain

    selected: Any = Chain(chain.id)  # type: ignore[no-untyped-call]
    residues = list(chain)
    sequence = []
    for start, end in ranges:
        sequence.append(chain.sequence[start - 1 : end])
        for residue in residues[start - 1 : end]:
            selected.add(residue.copy())
    selected.sequence = "".join(sequence)
    selected.is_het = None
    return selected


def _merge_chains(chains: Sequence[Any], chain_id: str) -> Any:
    """Copy one or more chains into one continuously numbered DockQ partner."""
    from Bio.PDB.Chain import Chain

    merged: Any = Chain(chain_id)  # type: ignore[no-untyped-call]
    sequence = []
    for chain in chains:
        sequence.append(chain.sequence)
        for residue in chain:
            copied = residue.copy()
            copied.id = (" ", len(merged) + 1, " ")
            merged.add(copied)
    merged.sequence = "".join(sequence)
    merged.is_het = None
    return merged


def _backbone_coordinates(chain: Any, atom_names: Sequence[str]) -> np.ndarray:
    coordinates: list[Any] = []
    for residue in chain:
        missing = [name for name in atom_names if name not in residue]
        if missing:
            raise ValueError(f"protein residue {chain.id}:{residue.id[1]} lacks atoms {missing}")
        coordinates.extend(residue[name].coord for name in atom_names)
    return np.asarray(coordinates, dtype=np.float64)


def _target_aligned_binder_rmsd(
    model_binder: Any,
    model_target: Any,
    native_binder: Any,
    native_target: Any,
    atom_names: Sequence[str],
) -> float:
    from Bio.SVDSuperimposer import SVDSuperimposer

    native_target_atoms = _backbone_coordinates(native_target, atom_names)
    model_target_atoms = _backbone_coordinates(model_target, atom_names)
    superimposer: Any = SVDSuperimposer()  # type: ignore[no-untyped-call]
    superimposer.set(native_target_atoms, model_target_atoms)
    superimposer.run()
    rotation, translation = superimposer.get_rotran()
    native_binder_atoms = _backbone_coordinates(native_binder, atom_names)
    model_binder_atoms = _backbone_coordinates(model_binder, atom_names)
    difference = model_binder_atoms @ rotation + translation - native_binder_atoms
    return float(np.sqrt(np.mean(np.sum(np.square(difference), axis=1))))


def _score_mapping(
    native_chains: Sequence[Any],
    model_chains: Sequence[Any],
    mapping: Sequence[int],
    atom_names: Sequence[str],
    target_residue_ranges: tuple[tuple[tuple[int, int], ...], ...] | None = None,
) -> dict[str, Any]:
    from DockQ.DockQ import run_on_chains

    # DockQ caches Biopython entities by full_id, so synthetic chains need unique IDs.
    input_id = next(_DOCKQ_INPUT_IDS)
    native_binder = _merge_chains(native_chains[:1], f"native_binder_{input_id}")
    model_binder = _merge_chains(model_chains[:1], f"model_binder_{input_id}")
    native_targets = list(native_chains[1:])
    model_targets = [model_chains[index + 1] for index in mapping]
    if target_residue_ranges is not None:
        native_targets = [
            _select_residues(chain, ranges)
            for chain, ranges in zip(native_targets, target_residue_ranges, strict=True)
        ]
        model_targets = [
            _select_residues(chain, ranges)
            for chain, ranges in zip(model_targets, target_residue_ranges, strict=True)
        ]
    native_target = _merge_chains(native_targets, f"native_target_{input_id}")
    model_target = _merge_chains(model_targets, f"model_target_{input_id}")
    result = run_on_chains(
        (model_binder, model_target),
        (native_binder, native_target),
        no_align=True,
        small_molecule=False,
    )
    target_aligned_rmsd = _target_aligned_binder_rmsd(
        model_binder, model_target, native_binder, native_target, atom_names
    )
    target_chain_mapping = [
        {
            "reference": native_chains[index + 1].id,
            "prediction": model_chains[prediction_index + 1].id,
        }
        for index, prediction_index in enumerate(mapping)
    ]
    if result is None:
        # DockQ is undefined when the selected design-model reference has no
        # native interface contacts. Treat that failed binding pose as score
        # zero while retaining the independently defined target-aligned RMSD.
        return {
            "dockq": 0.0,
            "dockq_status": "no_reference_contacts",
            "fnat": 0.0,
            "fnonnat": None,
            "contact_f1": 0.0,
            "interface_rmsd": None,
            "dockq_ligand_rmsd": None,
            "dockq_receptor": None,
            "dockq_ligand": None,
            "target_aligned_binder_backbone_rmsd": target_aligned_rmsd,
            "interface_contact_counts": {
                "reference": 0,
                "prediction": None,
                "recovered": 0,
                "non_native": None,
            },
            "clashes": None,
            "target_chain_mapping": target_chain_mapping,
        }

    binder_role = str(result["class1"])
    return {
        "dockq": float(result["DockQ"]),
        "dockq_status": "scored",
        "fnat": float(result["fnat"]),
        "fnonnat": float(result["fnonnat"]),
        "contact_f1": float(result["F1"]),
        "interface_rmsd": float(result["iRMSD"]),
        "dockq_ligand_rmsd": float(result["LRMSD"]),
        "dockq_receptor": "binder" if binder_role == "receptor" else "target",
        "dockq_ligand": "binder" if binder_role == "ligand" else "target",
        "target_aligned_binder_backbone_rmsd": target_aligned_rmsd,
        "interface_contact_counts": {
            "reference": int(result["nat_total"]),
            "prediction": int(result["model_total"]),
            "recovered": int(result["nat_correct"]),
            "non_native": int(result["nonnat_count"]),
        },
        "clashes": int(result["clashes"]),
        "target_chain_mapping": target_chain_mapping,
    }


def score_self_consistency(
    reference_pdb: str | Path,
    prediction_pdb: str | Path,
    *,
    binder_length: int,
    target_sequences: Sequence[str],
    atom_names: Sequence[str] = ("N", "CA", "C", "O"),
    target_residue_ranges: Sequence[Sequence[Sequence[int]]] | None = None,
) -> dict[str, Any]:
    """Report primary DockQ/RMSD, optionally over fixed target sequence ranges."""
    native_chains = _parse_pdb(reference_pdb)
    model_chains = _parse_pdb(prediction_pdb)
    _validate_layout(native_chains, binder_length, target_sequences)
    _validate_layout(model_chains, binder_length, target_sequences)
    ranges = _normalize_target_residue_ranges(target_sequences, target_residue_ranges)
    candidates = [
        _score_mapping(native_chains, model_chains, mapping, atom_names, ranges)
        for mapping in _target_permutations(target_sequences)
    ]
    primary = max(candidates, key=lambda candidate: candidate["dockq"])
    if ranges is None:
        return primary

    unmasked_candidates = [
        _score_mapping(native_chains, model_chains, mapping, atom_names)
        for mapping in _target_permutations(target_sequences)
    ]
    unmasked = max(unmasked_candidates, key=lambda candidate: candidate["dockq"])
    return primary | {
        "self_consistency_target_residue_ranges": [
            [[start, end] for start, end in chain_ranges] for chain_ranges in ranges
        ],
        "self_consistency_range_indexing": "one_based_inclusive",
        "unmasked_self_consistency": unmasked,
    }
