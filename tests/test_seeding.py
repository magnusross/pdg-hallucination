from __future__ import annotations

import numpy as np
import pytest

from binderopt.optimization.optimizers import initialize_logits
from binderopt.optimization.seeding import design_keys, key_words, lane_keys

BINDER_LENGTH = 80
LIVE = np.asarray([letter != "C" for letter in "ARNDCQEGHILKMFPSTWYV"])
MASK = np.broadcast_to(LIVE, (BINDER_LENGTH, 20))


def _initial_logits(base_seed: int, num_designs: int, *, init_scale: float = 0.01):
    init_keys, _, _ = lane_keys(design_keys(base_seed, num_designs))
    return np.asarray(
        initialize_logits(
            init_keys,
            binder_length=BINDER_LENGTH,
            alphabet_size=20,
            init_scale=init_scale,
            mask=MASK,
        )
    )


def test_both_optimizer_arms_start_from_the_same_logits() -> None:
    """Pairing rests on nothing but the base seed: the arms share this derivation."""
    np.testing.assert_array_equal(_initial_logits(1, 16), _initial_logits(1, 16))


def test_a_design_key_does_not_depend_on_num_designs_or_batching() -> None:
    """`split` is prefix stable, so a cell still pairs after a re-batch or a resize."""
    np.testing.assert_array_equal(
        key_words(design_keys(1, 64)), key_words(design_keys(1, 128))[:64]
    )
    np.testing.assert_array_equal(_initial_logits(1, 16), _initial_logits(1, 64)[:16])


def test_separate_base_seeds_give_disjoint_independent_designs() -> None:
    """Seed-sharded cells must not silently repeat each other's designs."""
    words = np.concatenate([key_words(design_keys(seed, 128)) for seed in range(8)])
    assert len(np.unique(words, axis=0)) == len(words)

    # Masked tokens hold a large sentinel, so compare the drawn entries only.
    first, second = (_initial_logits(seed, 64)[:, :, LIVE].reshape(64, -1) for seed in (1, 2))
    unit = [array / np.linalg.norm(array, axis=1, keepdims=True) for array in (first, second)]
    # 1600-dimensional draws: independence shows up as near-orthogonality throughout.
    assert np.abs(unit[0] @ unit[1].T).max() < 0.2


def test_a_design_cell_needs_a_positive_seed_and_design_count() -> None:
    with pytest.raises(ValueError, match="num_designs"):
        design_keys(1, 0)
    with pytest.raises(ValueError, match="base_seed"):
        design_keys(-1, 4)
