from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from binderopt.optimization.optimizers import optimize_pgd


class _ToyLoss:
    def __init__(self, target: np.ndarray, *, ranking: bool = False) -> None:
        self.target = jnp.asarray(target)
        self.ranking = ranking

    def __call__(self, sequence, *, key):
        del key
        value = jnp.square(sequence - self.target).sum()
        return value, {"ranking_loss": value} if self.ranking else {}


def test_pgd_matches_accepted_legacy_fixture_except_stable_lane_order() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "pgd_legacy_toy.json").read_text(encoding="utf-8")
    )
    initial = jnp.asarray(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
            [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]],
        ],
        dtype=jnp.float32,
    )
    target = np.array([[0.8, 0.2, 0.0], [0.1, 0.2, 0.7]], np.float32)
    result = optimize_pgd(
        loss=_ToyLoss(target),
        tail_loss=_ToyLoss(target, ranking=True),
        initial_probabilities=initial,
        n_steps=4,
        tail_select_steps=2,
        binder_length=2,
        stepsize_scale=0.2 / np.sqrt(2),
        momentum=0.0,
        max_gradient_norm=1.0,
        mask=np.array([[1, 1, 0], [1, 1, 0]], bool),
        keys=jax.random.split(jax.random.key(9), 2),
    )
    ordered = sorted(result.selections, key=lambda item: item.tail_ranking_loss)
    np.testing.assert_allclose(
        np.stack([item.pssm for item in ordered]), fixture["pssms"], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        [item.tail_ranking_loss for item in ordered], fixture["losses"], rtol=1e-6, atol=1e-7
    )
    assert [item.lane for item in result.selections] == [0, 1]
