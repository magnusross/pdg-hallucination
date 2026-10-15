"""The single supported design objective and protein-only iPTM monitor."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from mosaic.common import LossTerm
from mosaic.losses.structure_prediction import (
    ESMFoldGlobularity,
    ESMFoldInterContact,
    WithinBinderContact,
    predicted_tm_score,
)
from mosaic.optimizers import batched_eval

from binderopt.optimization.optimizers import TailSelection
from binderopt.optimization.seeding import fold_in_step
from binderopt.structure.features import EntityMasks


def build_structure_objective(config: Mapping[str, Any]) -> Any:
    """Build the fixed three-term Mosaic loss from explicit YAML values."""
    expected = {"within_binder_contact", "inter_contact", "globularity", "esmc_pll"}
    if set(config) != expected:
        raise ValueError(f"design.objective fields must be exactly {sorted(expected)}")
    within = config["within_binder_contact"]
    inter = config["inter_contact"]
    globularity = config["globularity"]
    return (
        float(within["weight"])
        * WithinBinderContact(num_contacts_per_residue=int(within["num_contacts_per_residue"]))
        + float(inter["weight"])
        * ESMFoldInterContact(contact_distance=float(inter["contact_distance"]))
        + float(globularity["weight"]) * ESMFoldGlobularity()
    )


def validate_contact_distance(distogram_bins: Any, contact_distance: float) -> None:
    """Fail before optimization when a requested cutoff lies in the dead tail."""
    max_bin = float(np.asarray(distogram_bins)[-1])
    useful_limit = max_bin - 5.0
    if contact_distance >= useful_limit:
        raise ValueError(
            f"contact_distance={contact_distance:g} is outside the useful distogram range; "
            f"checkpoint maximum is {max_bin:g} and cutoff must be below {useful_limit:g}"
        )


class ProteinOnlyStructureLoss(LossTerm):  # type: ignore[misc]
    """Give the fixed distogram losses only binder and target protein tokens."""

    inner: Any
    protein_indices: jax.Array

    def __call__(self, sequence: jax.Array, output: Any, *, key: jax.Array) -> Any:
        indices = self.protein_indices
        distogram = output.distogram_logits[indices[:, None], indices[None, :]]
        return self.inner(
            sequence,
            dataclasses.replace(output, distogram_logits=distogram),
            key=key,
        )


class IPTMMonitor(LossTerm):  # type: ignore[misc]
    """Expose protein-only binder/combined-target iPTM without optimizing it."""

    protein_indices: jax.Array
    pair_mask: jax.Array

    def __init__(self, masks: EntityMasks) -> None:
        protein_indices = np.flatnonzero(masks.protein)
        binder = masks.binder[protein_indices]
        target = masks.target[protein_indices]
        self.protein_indices = jnp.asarray(protein_indices)
        self.pair_mask = jnp.asarray(
            (binder[:, None] & target[None, :]) | (target[:, None] & binder[None, :])
        )

    def __call__(
        self, sequence: jax.Array, output: Any, *, key: jax.Array
    ) -> tuple[Any, dict[str, Any]]:
        del sequence, key
        indices = self.protein_indices
        logits = output.pae_logits[indices[:, None], indices[None, :]]
        iptm = predicted_tm_score(logits, output.pae_bins, self.pair_mask).max()
        return jnp.zeros_like(iptm), {"ranking_loss": -iptm, "iptm": iptm}


def protein_only(loss: Any, masks: EntityMasks) -> ProteinOnlyStructureLoss:
    return ProteinOnlyStructureLoss(loss, jnp.asarray(np.flatnonzero(masks.protein)))


def harden_and_evaluate(
    loss: Any,
    selections: Sequence[TailSelection],
    *,
    alphabet_size: int,
    keys: jax.Array,
) -> tuple[jax.Array, np.ndarray]:
    """Argmax selected PSSMs and evaluate the written one-hot sequences once."""
    pssms = jnp.asarray(np.stack([selection.pssm for selection in selections]))
    one_hot = jax.nn.one_hot(pssms.argmax(-1), alphabet_size)
    if keys.shape[:1] != (len(selections),):
        raise ValueError("one hardening key per selection is required")
    values, _, _ = batched_eval(loss, one_hot, keys)
    return one_hot, np.asarray(values)


@eqx.filter_jit
def _batched_log_perplexity(loss: Any, sequences: jax.Array, keys: jax.Array) -> jax.Array:
    def single(sequence: jax.Array, key: jax.Array) -> jax.Array:
        _, auxiliary = loss.pll_term(sequence, key=key)
        return auxiliary["esmc_log_perplexity"]

    return jax.vmap(single)(sequences, keys)


def hardened_esmc_log_perplexity(
    loss: Any,
    one_hot: jax.Array,
    *,
    keys: jax.Array,
    batches: int,
) -> np.ndarray:
    """Mean ESMC pseudo-log-perplexity of each written one-hot sequence.

    Uses the design loss's own `esmc_pll` term, so the recorded value is exactly the
    quantity the objective weights, measured on the sequence that is written and ranked
    rather than on the soft PSSM. Independent batches are averaged instead of widening
    that term's internal vmap, so peak memory matches one design step.
    """
    if batches <= 0:
        raise ValueError("final_esmc_perplexity_batches must be positive")
    if keys.shape[:1] != one_hot.shape[:1]:
        raise ValueError("one perplexity key per hardened sequence is required")
    total = jnp.zeros(one_hot.shape[0], dtype=jnp.float32)
    for batch in range(batches):
        total = total + _batched_log_perplexity(loss, one_hot, fold_in_step(keys, batch))
    return np.asarray(total / batches)
