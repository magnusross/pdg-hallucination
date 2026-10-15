from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from binderopt.ranking.models import protein_ipsae_min
from binderopt.structure.features import EntityMasks


@dataclass(frozen=True)
class Output:
    full_sequence: jax.Array
    pae_logits: jax.Array
    pae_bins: jax.Array


def _local_direction(pae: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> float:
    scores = []
    for row in rows:
        values = pae[row, columns]
        values = values[values < 10.0]
        if not len(values):
            scores.append(0.0)
            continue
        d0 = 1.24 * (max(len(values), 27) - 15) ** (1 / 3) - 1.8
        scores.append(float(np.mean(1 / (1 + np.square(values / d0)))))
    return max(scores)


def test_mosaic_ipsae_min_matches_independent_synthetic_calculation() -> None:
    bins = np.array([2.0, 6.0, 12.0, 20.0], np.float32)
    chosen = np.array(
        [
            [3, 3, 0, 1, 2],
            [3, 3, 1, 0, 1],
            [1, 0, 3, 3, 3],
            [0, 1, 3, 3, 3],
            [2, 0, 3, 3, 3],
        ]
    )
    logits = np.full((5, 5, 4), -30.0, np.float32)
    np.put_along_axis(logits, chosen[..., None], 30.0, axis=-1)
    pae = bins[chosen]
    binder = np.array([0, 1])
    target = np.array([2, 3, 4])
    expected = min(
        _local_direction(pae, binder, target),
        _local_direction(pae, target, binder),
    )
    masks = EntityMasks(
        binder=np.array([1, 1, 0, 0, 0], dtype=bool),
        target=np.array([0, 0, 1, 1, 1], dtype=bool),
        ligand=np.zeros(5, dtype=bool),
    )
    actual = protein_ipsae_min(
        jnp.zeros((2, 20)),
        Output(jnp.arange(5), jnp.asarray(logits), jnp.asarray(bins)),
        masks,
        key=jax.random.key(0),
    )
    np.testing.assert_allclose(actual, expected, atol=1e-4)
