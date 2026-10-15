"""Key derivation from the configured seed, and with it the arm pairing contract."""

from __future__ import annotations

import jax
import numpy as np
from numpy.typing import NDArray


def design_keys(base_seed: int, num_designs: int) -> jax.Array:
    """One independent key per design index, derived from the configured seed.

    `split` is prefix stable, so design `d`'s key depends on the base seed alone, not
    on `num_designs`, `batch_size`, or the optimizer arm. That is what pairs the arms.
    """
    if base_seed < 0 or num_designs <= 0:
        raise ValueError("base_seed must be non-negative and num_designs positive")
    return jax.random.split(jax.random.key(base_seed), num_designs)


def lane_keys(keys: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split each design's key into its initialization, optimizer, and hardening uses."""
    split = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
    return split[:, 0], split[:, 1], split[:, 2]


def fold_in_step(keys: jax.Array, step: int) -> jax.Array:
    """Advance every lane's key by the step index, identically in both arms."""
    return jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, step)


def key_words(keys: jax.Array) -> NDArray[np.uint32]:
    """The words behind each key, recorded per design as the `design_key` join key."""
    return np.asarray(jax.random.key_data(keys))
