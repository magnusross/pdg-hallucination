"""Typed target data used after Hydra has composed a target YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class ProteinChain:
    entity_id: str
    sequence: str
    use_msa: bool


@dataclass(frozen=True, slots=True)
class Ligand:
    entity_id: str
    ccd: str
    count: int


@dataclass(frozen=True, slots=True)
class Target:
    target_id: str
    name: str
    default_binder_length: int
    chains: tuple[ProteinChain, ...]
    ligands: tuple[Ligand, ...]
    self_consistency_residue_ranges: tuple[tuple[tuple[int, int], ...], ...] | None


def _self_consistency_residue_ranges(
    config: Mapping[str, Any], chains: tuple[ProteinChain, ...]
) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    specification = config.get("self_consistency")
    if specification is None:
        return None
    if not isinstance(specification, Mapping):
        raise ValueError("target self_consistency must be a mapping")
    raw_ranges = specification.get("primary_target_residue_ranges")
    if raw_ranges is None:
        return None
    if not isinstance(raw_ranges, Mapping):
        raise ValueError("primary_target_residue_ranges must map target entity IDs to ranges")
    chain_ids = {chain.entity_id for chain in chains}
    unknown = set(raw_ranges) - chain_ids
    if unknown:
        raise ValueError(f"self-consistency ranges name unknown target chains: {sorted(unknown)}")

    selected: list[tuple[tuple[int, int], ...]] = []
    for chain in chains:
        values = raw_ranges.get(chain.entity_id, ((1, len(chain.sequence)),))
        try:
            ranges = tuple((int(value[0]), int(value[1])) for value in values)
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(f"invalid self-consistency ranges for {chain.entity_id}") from error
        if not ranges:
            raise ValueError(f"self-consistency ranges are empty for {chain.entity_id}")
        previous_end = 0
        for start, end in ranges:
            if start <= previous_end or end < start or end > len(chain.sequence):
                raise ValueError(
                    f"self-consistency ranges for {chain.entity_id} must be ordered, "
                    "non-overlapping, one-based, inclusive, and within the sequence"
                )
            previous_end = end
        selected.append(ranges)
    return tuple(selected)


def target_from_config(config: Mapping[str, Any]) -> Target:
    """Convert the already validated target config into immutable data."""
    chains = tuple(ProteinChain(**chain) for chain in config["chains"])
    return Target(
        target_id=config["target_id"],
        name=config["name"],
        default_binder_length=config["default_binder_length"],
        chains=chains,
        ligands=tuple(Ligand(**ligand) for ligand in config["ligands"]),
        self_consistency_residue_ranges=_self_consistency_residue_ranges(config, chains),
    )


def apply_msa_policy(
    chains: tuple[ProteinChain, ...], *, supports_msa: bool
) -> tuple[ProteinChain, ...]:
    """Disable requested MSAs for a model which cannot consume them."""
    if supports_msa:
        return chains
    return tuple(replace(chain, use_msa=False) for chain in chains)
