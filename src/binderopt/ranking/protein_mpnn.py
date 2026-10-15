"""ProteinMPNN sampling."""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from mosaic.losses.protein_mpnn import _per_chain_residue_idx, boltz_to_mpnn_matrix


class MPNNInput(NamedTuple):
    backbone_coordinates: Any
    full_sequence: Any
    asym_id: Any
    residue_idx: Any


@eqx.filter_jit
def autoregressive_inverse_fold(mpnn, binder_length, output, temp, key, bias=None):
    """Sample the binder one residue at a time in a random decoding order."""
    total_length = output.full_sequence.shape[0]
    mask = jnp.ones(total_length, dtype=jnp.int32)
    residue_idx = _per_chain_residue_idx(output.asym_id, output.residue_idx)
    encode_key, order_key, sample_key = jax.random.split(key, 3)
    h_V, h_E, E_idx = mpnn.encode(
        X=output.backbone_coordinates,
        mask=mask,
        residue_idx=residue_idx,
        chain_encoding_all=output.asym_id,
        key=encode_key,
    )
    decoding_order = jax.random.uniform(order_key, (total_length,))
    decoding_order = decoding_order.at[:binder_length].add(2.0)
    binder_order = jnp.argsort(decoding_order)[-binder_length:]
    token_matrix = jnp.asarray(boltz_to_mpnn_matrix())
    sequence = output.full_sequence.at[:binder_length].set(0)

    def sample_residue(carry, position):
        sequence, sample_key = carry
        sample_key, residue_key = jax.random.split(sample_key)
        logits = mpnn.decode(
            S=sequence @ token_matrix,
            h_V=h_V,
            h_E=h_E,
            E_idx=E_idx,
            mask=mask,
            decoding_order=decoding_order,
        )[0, position] @ token_matrix.T
        if bias is not None:
            logits += bias[position]
        residue = (
            logits.argmax()
            if temp == 0
            else jax.random.categorical(residue_key, logits / temp)
        )
        sequence = sequence.at[position].set(
            jax.nn.one_hot(residue, output.full_sequence.shape[-1])
        )
        return (sequence, sample_key), None

    (sequence, _), _ = jax.lax.scan(
        sample_residue,
        (sequence, sample_key),
        binder_order,
    )
    return sequence[:binder_length].argmax(-1)
