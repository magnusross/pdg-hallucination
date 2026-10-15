"""ESM feature construction and explicit entity masks."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from binderopt.structure.targets import Ligand, Target, apply_msa_policy


@dataclass(frozen=True, slots=True)
class EntityMasks:
    binder: NDArray[np.bool_]
    target: NDArray[np.bool_]
    ligand: NDArray[np.bool_]

    @property
    def protein(self) -> NDArray[np.bool_]:
        return self.binder | self.target


@dataclass(frozen=True, slots=True)
class PreparedComplex:
    features: Any
    writer: Any
    masks: EntityMasks


def entity_masks(
    entity_ids: NDArray[Any],
    protein_mask: NDArray[np.bool_],
    *,
    binder_entity: int,
    target_entities: tuple[int, ...],
) -> EntityMasks:
    """Separate one binder, the combined protein target, and non-protein tokens."""
    entities = np.asarray(entity_ids)
    protein = np.asarray(protein_mask, dtype=bool)
    if entities.ndim != 1 or protein.shape != entities.shape:
        raise ValueError("entity_ids and protein_mask must be matching one-dimensional arrays")

    binder = (entities == binder_entity) & protein
    target = np.isin(entities, target_entities) & protein
    ligand = ~protein
    if not binder.any():
        raise ValueError("binder entity has no protein tokens")
    if not target.any():
        raise ValueError("combined target group has no protein tokens")
    if np.any(binder & target):
        raise ValueError("binder and target entities must be distinct")
    if np.any(protein & ~(binder | target)):
        raise ValueError("every protein token must belong to the binder or target group")
    return EntityMasks(binder=binder, target=target, ligand=ligand)


def protein_pair_mask(masks: EntityMasks) -> NDArray[np.bool_]:
    """Symmetric binder-versus-combined-target pair mask; ligands are absent."""
    return (masks.binder[:, None] & masks.target[None, :]) | (
        masks.target[:, None] & masks.binder[None, :]
    )


def build_esm_binder_features(
    model: Any,
    binder_length: int,
    target: Target,
    *,
    supports_msa: bool,
) -> PreparedComplex:
    """Build one designed binder followed by ordered target chains and ligands."""
    from mosaic.structure_prediction import TargetChain

    chains = apply_msa_policy(target.chains, supports_msa=supports_msa)
    mosaic_chains = [
        TargetChain(sequence=chain.sequence, use_msa=chain.use_msa) for chain in chains
    ]
    if target.ligands:
        features, writer = _esm_ligand_features(
            model, binder_length, mosaic_chains, target.ligands
        )
    else:
        features, writer = model.binder_features(
            binder_length=binder_length, chains=mosaic_chains
        )
    masks = _masks_from_writer(writer, protein_entities=1 + len(chains))
    return PreparedComplex(features, writer, masks)



def build_esm_ranking_features(
    model: Any, sequence: str, target: Target, *, supports_msa: bool
) -> PreparedComplex:
    """Build native final-sequence ESM features, with the binder as chain zero."""
    from mosaic.structure_prediction import TargetChain

    target_chains = apply_msa_policy(target.chains, supports_msa=supports_msa)
    chains = [
        TargetChain(sequence=sequence, use_msa=False),
        *(
            TargetChain(sequence=chain.sequence, use_msa=chain.use_msa)
            for chain in target_chains
        ),
    ]
    if target.ligands:
        features, writer = _esm_ligand_features(model, None, chains, target.ligands)
    else:
        features, writer = model.target_only_features(chains=chains)
    masks = _masks_from_writer(writer, protein_entities=len(chains))
    return PreparedComplex(features, writer, masks)


def _masks_from_writer(writer: Any, *, protein_entities: int) -> EntityMasks:
    tokens = [token for chain in writer for token in chain.tokens]
    if not tokens:
        raise ValueError("ESMFold2 feature writer contains no tokens")
    token_count = max(int(token.token_index) for token in tokens) + 1
    entities = np.full(token_count, -1, dtype=np.int32)
    protein = np.zeros(token_count, dtype=bool)
    for entity, chain in enumerate(writer):
        for token in chain.tokens:
            index = int(token.token_index)
            entities[index] = entity
            protein[index] = entity < protein_entities
    if np.any(entities < 0):
        raise ValueError("ESMFold2 feature writer left token indices unassigned")
    return entity_masks(
        entities,
        protein,
        binder_entity=0,
        target_entities=tuple(range(1, protein_entities)),
    )


def _esm_ligand_features(
    model: Any, binder_length: int | None, chains: list[Any], ligands: tuple[Ligand, ...]
) -> tuple[Any, Any]:
    """Minimal adapter around ESMFold2's native input builder, proteins first."""
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )
    from mosaic.models.esmfold2 import _build_pack, _fetch_msa

    chain_ids = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    all_chains = (
        chains
        if binder_length is None
        else [type(chains[0])(sequence="W" * binder_length, use_msa=False), *chains]
    )
    model._check_msa_compat(all_chains)
    sequences = [
        ProteinInput(
            id=next(chain_ids),
            sequence=chain.sequence,
            msa=_fetch_msa(chain.sequence) if chain.use_msa else None,
        )
        for chain in all_chains
    ]
    for ligand in ligands:
        ids = [next(chain_ids) for _ in range(ligand.count)]
        sequences.append(LigandInput(id=ids, ccd=[ligand.ccd]))
    raw, writer = ESMFold2InputBuilder().prepare_input(
        StructurePredictionInput(sequences=sequences)
    )
    pack = _build_pack(
        model.esmc,
        raw,
        design_positions=np.arange(0 if binder_length is None else binder_length),
        unk_input_id=model.unk_input_id,
    )
    return dataclasses.replace(pack, refresh=binder_length is not None), writer
