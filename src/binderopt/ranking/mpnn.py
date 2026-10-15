"""SolubleMPNN redesign stage and final-sequence naturalness scoring."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from binderopt.outputs import atomic_json, utc_now

SequenceScorer = Callable[[str, Any], float]


def mpnn_structure_input(
    reference_pdb: str | Path,
    *,
    binder_length: int,
    target_sequences: Sequence[str],
) -> Any:
    """Convert a reference PDB to the protein-only fields consumed by Mosaic MPNN."""
    import gemmi
    import jax
    import jax.numpy as jnp
    from mosaic.common import TOKENS

    from binderopt.ranking.protein_mpnn import MPNNInput

    text = (
        Path(reference_pdb).read_text(encoding="utf-8")
        if isinstance(reference_pdb, Path)
        else reference_pdb
    )
    structure = gemmi.read_pdb_string(text)
    if len(structure) != 1:
        raise ValueError("MPNN reference must contain exactly one model")
    chains = []
    for structure_chain in structure[0]:
        residues = [
            residue
            for residue in structure_chain
            if gemmi.find_tabulated_residue(residue.name).is_amino_acid()
        ]
        if residues:
            chains.append(residues)

    expected_lengths = (binder_length, *(len(sequence) for sequence in target_sequences))
    observed_lengths = tuple(len(chain) for chain in chains)
    if observed_lengths != expected_lengths:
        raise ValueError(
            f"MPNN protein chain lengths differ: expected {expected_lengths}, "
            f"observed {observed_lengths}"
        )
    sequences = [
        "".join(gemmi.find_tabulated_residue(residue.name).one_letter_code for residue in chain)
        for chain in chains
    ]
    if tuple(sequences[1:]) != tuple(target_sequences):
        raise ValueError("MPNN reference target sequences differ from the configured target")
    full_sequence = "".join(sequences)
    if any(letter not in TOKENS for letter in full_sequence):
        raise ValueError("MPNN reference contains a non-canonical protein residue")

    atom_names = ("N", "CA", "C", "O")
    coordinates = []
    for chain_residues in chains:
        chain_coordinates = []
        for residue in chain_residues:
            atoms = {atom.name.strip(): atom for atom in residue if atom.altloc in ("\x00", "A")}
            missing = [name for name in atom_names if name not in atoms]
            if missing:
                raise ValueError(f"MPNN reference residue {residue.seqid} lacks atoms {missing}")
            chain_coordinates.append(
                [[atoms[name].pos.x, atoms[name].pos.y, atoms[name].pos.z] for name in atom_names]
            )
        coordinates.extend(chain_coordinates)
    asym_id = np.concatenate(
        [np.full(len(chain), index, dtype=np.int32) for index, chain in enumerate(chains)]
    )
    residue_idx = np.concatenate([np.arange(len(chain), dtype=np.int32) for chain in chains])
    return MPNNInput(
        backbone_coordinates=jnp.asarray(coordinates),
        full_sequence=jax.nn.one_hot(
            jnp.asarray([TOKENS.index(letter) for letter in full_sequence]), len(TOKENS)
        ),
        asym_id=jnp.asarray(asym_id),
        residue_idx=jnp.asarray(residue_idx),
    )


def redesign_sequence(
    mpnn: Any,
    reference_pdb: str | Path,
    *,
    binder_length: int,
    target_sequences: Sequence[str],
    temperature: float,
    excluded_amino_acids: Sequence[str],
    key: Any,
) -> str:
    """Redesign only the first chain while retaining every target protein chain."""
    import jax.numpy as jnp
    from mosaic.common import TOKENS

    from binderopt.ranking.protein_mpnn import autoregressive_inverse_fold

    if temperature < 0:
        raise ValueError("MPNN temperature must be non-negative")
    excluded = set(excluded_amino_acids)
    unknown = excluded.difference(TOKENS)
    if unknown:
        raise ValueError(f"unknown MPNN amino-acid exclusions: {sorted(unknown)}")
    output = mpnn_structure_input(
        reference_pdb,
        binder_length=binder_length,
        target_sequences=target_sequences,
    )
    bias = jnp.zeros((binder_length, len(TOKENS)))
    for letter in excluded:
        bias = bias.at[:, TOKENS.index(letter)].set(-jnp.inf)
    indices = autoregressive_inverse_fold(
        mpnn,
        binder_length,
        output,
        temp=temperature,
        key=key,
        bias=bias,
    )
    sequence = "".join(TOKENS[int(index)] for index in np.asarray(indices))
    if len(sequence) != binder_length or excluded.intersection(sequence):
        raise ValueError("SolubleMPNN returned an invalid or excluded amino acid")
    return sequence


def make_esmc_scorer(
    language_model: Mapping[str, Any], objective: Mapping[str, Any]
) -> SequenceScorer:
    """Load the design recipe's ESMC once and score final one-hot sequences."""
    import jax
    import jax.numpy as jnp
    from hydra.utils import get_method
    from mosaic.common import TOKENS
    from mosaic.losses.esmc import ESMCPseudoPerplexity

    esmc = get_method(str(language_model["factory"]))(str(language_model["checkpoint"]))
    pll = objective["esmc_pll"]
    loss = ESMCPseudoPerplexity(
        esmc,
        masking_fraction=float(pll["masking_fraction"]),
        num_samples=int(pll["num_samples"]),
    )
    batches = int(objective["final_esmc_perplexity_batches"])
    if batches <= 0:
        raise ValueError("final_esmc_perplexity_batches must be positive")

    def score(sequence: str, key: Any) -> float:
        one_hot = jax.nn.one_hot(
            jnp.asarray([TOKENS.index(letter) for letter in sequence]), len(TOKENS)
        )
        total = 0.0
        for batch in range(batches):
            value, _ = loss(one_hot, key=jax.random.fold_in(key, batch))
            total += float(np.asarray(value))
        return total / batches

    return score


def run_soluble_mpnn_worker(
    cell_dir: Path,
    *,
    recipe: Mapping[str, Any],
    ranking_seed: int,
    cache_root: Path,
    offline: bool,
    design_indices: Sequence[int] | None = None,
) -> None:
    """Load SolubleMPNN and ESMC once in an isolated single-cell worker."""
    from hydra.utils import get_method

    from binderopt.assets.cache import configure_process_caches

    configure_process_caches(cache_root, offline=offline)
    mpnn_spec = recipe["self_consistency"]["mpnn"]
    mpnn = get_method(str(mpnn_spec["factory"]))()
    result = json.loads((cell_dir / "result.json").read_text(encoding="utf-8"))
    design_config = result["config"]["design"]
    language_model = result["config"]["profile"]["design"]["language_model"]
    score_sequence = make_esmc_scorer(
        language_model,
        design_config["objective"]
        | {"final_esmc_perplexity_batches": design_config["final_esmc_perplexity_batches"]},
    )
    run_soluble_mpnn_cell(
        cell_dir,
        recipe_name=str(recipe["name"]),
        mpnn_spec=mpnn_spec,
        ranking_seed=ranking_seed,
        mpnn=mpnn,
        score_sequence=score_sequence,
        design_indices=design_indices,
    )


def run_soluble_mpnn_cell(
    cell_dir: Path,
    *,
    recipe_name: str,
    mpnn_spec: Mapping[str, Any],
    ranking_seed: int,
    mpnn: Any,
    score_sequence: SequenceScorer,
    design_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Redesign selected optimizer sequences and atomically publish one cell shard."""
    import jax

    started = time.perf_counter()
    result = json.loads((cell_dir / "result.json").read_text(encoding="utf-8"))
    candidates = list(result.get("design_candidates", result["designs"]))
    indices = [int(design["design_index"]) for design in candidates]
    if indices != list(range(len(candidates))):
        raise ValueError("MPNN requires contiguous generation-ordered design indices")
    requested = indices if design_indices is None else [int(index) for index in design_indices]
    selected = set(requested)
    if len(selected) != len(requested):
        raise ValueError("MPNN design indices must be unique")
    if not selected.issubset(indices):
        raise ValueError("MPNN design indices are not present in the design cell")
    designs = [design for design in candidates if int(design["design_index"]) in selected]
    target_config = result.get("config", {}).get("target")
    if not isinstance(target_config, Mapping):
        raise ValueError("design result has no resolved target config")
    target_sequences = [str(chain["sequence"]) for chain in target_config["chains"]]
    keys = jax.random.split(jax.random.key(ranking_seed), (len(designs), 2))
    records: list[dict[str, Any]] = []
    for design, (mpnn_key, naturalness_key) in zip(designs, keys, strict=True):
        reference = design.get("selected_reference")
        if not isinstance(reference, Mapping):
            raise ValueError("design has no selected optimizer reference; rerun the design cell")
        source = cell_dir / str(reference["structure"])
        sequence = redesign_sequence(
            mpnn,
            source,
            binder_length=int(result["binder_length"]),
            target_sequences=target_sequences,
            temperature=float(mpnn_spec["temperature"]),
            excluded_amino_acids=list(mpnn_spec["excluded_amino_acids"]),
            key=mpnn_key,
        )
        log_perplexity = score_sequence(sequence, naturalness_key)
        records.append(
            {
                "design_index": int(design["design_index"]),
                "optimizer_sequence": str(design["optimizer_sequence"]),
                "sequence": sequence,
                "source_structure": str(reference["structure"]),
                "source_model": str(reference["model_name"]),
                "esmc_log_perplexity": log_perplexity,
                "esmc_perplexity": float(np.exp(log_perplexity)),
            }
        )
    shard = {
        "model_name": "soluble_mpnn",
        "provenance": {
            "recipe": recipe_name,
            "model": dict(mpnn_spec),
            "ranking_seed": ranking_seed,
        },
        "timing": {
            "completed_at_utc": utc_now(),
            "wall_time_s": round(time.perf_counter() - started, 3),
        },
        "designs": records,
    }
    atomic_json(cell_dir / "ranking" / "soluble_mpnn.json", shard)
    return shard
