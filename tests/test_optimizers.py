from __future__ import annotations

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mosaic.optimizers import simplex_APGM

from binderopt.optimization.optimizers import (
    clip_gradients,
    optimize_biohub,
    optimize_pgd,
    project_simplex,
)


class ToyLoss:
    def __init__(self, target: np.ndarray, *, report_ranking: bool = False) -> None:
        self.target = jnp.asarray(target)
        self.report_ranking = report_ranking

    def __call__(self, sequence, *, key):
        del key
        value = jnp.square(sequence - self.target).sum()
        auxiliary = {"toy": value}
        if self.report_ranking:
            auxiliary |= {"ranking_loss": value, "iptm": -value}
        return value, auxiliary


class WrappedRankingLoss:
    """Mosaic's `reduce_samples` wraps scalar aux leaves in a one-element list."""

    def __call__(self, sequence, *, key):
        del key
        value = jnp.square(sequence).sum()
        return value, {"ranking_loss": [value], "iptm": [value]}


class SilentLoss:
    def __call__(self, sequence, *, key):
        del key
        return jnp.square(sequence).sum(), {}


def _mask(*rows: list[int]) -> np.ndarray:
    return np.array(rows, dtype=bool)


def test_simplex_projection_and_per_lane_clipping() -> None:
    values = np.array(
        [
            [[2.0, -1.0, 0.5], [0.2, 0.2, 0.2]],
            [[-3.0, 4.0, 1.0], [0.8, -0.1, 0.4]],
        ]
    )
    projected = project_simplex(values, _mask([1, 0, 1], [1, 0, 1])[None])
    np.testing.assert_allclose(projected.sum(axis=-1), 1.0, atol=1e-7)
    assert np.all(projected >= 0)
    assert np.all(projected[..., 1] == 0)

    gradients = jnp.stack([jnp.ones((2, 3)), 100 * jnp.ones((2, 3))])
    clipped = clip_gradients(gradients, 1.0)
    np.testing.assert_allclose(np.linalg.norm(clipped, axis=(1, 2)), [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(clipped[1] / clipped[0], 1.0, rtol=1e-6)


def test_pgd_keeps_lane_identity_and_selects_only_from_tail() -> None:
    initial = jnp.asarray(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
            [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]],
        ],
        dtype=jnp.float32,
    )
    target = np.array([[0.8, 0.2, 0.0], [0.1, 0.2, 0.7]], dtype=np.float32)
    result = optimize_pgd(
        loss=ToyLoss(target),
        tail_loss=ToyLoss(target, report_ranking=True),
        initial_probabilities=initial,
        n_steps=4,
        tail_select_steps=2,
        binder_length=2,
        stepsize_scale=0.2 / np.sqrt(2),
        momentum=0.0,
        max_gradient_norm=1.0,
        mask=_mask([1, 1, 0], [1, 1, 0]),
        keys=jax.random.split(jax.random.key(9), 2),
    )
    assert [selection.lane for selection in result.selections] == [0, 1]
    assert all(selection.step in (2, 3) for selection in result.selections)
    assert all(np.all(selection.pssm[..., 2] == 0) for selection in result.selections)
    # Every lane is recorded at every step, in lane order, and never reordered.
    assert [(point.lane, point.step) for point in result.trajectory] == [
        (lane, step) for step in range(4) for lane in range(2)
    ]
    # The monitor runs only in the tail window, so earlier steps report no ranking loss.
    assert [point.ranking_loss is None for point in result.trajectory] == [
        step < 2 for step in range(4) for _ in range(2)
    ]


def test_tail_ranking_unwraps_mosaic_single_sample_auxiliary() -> None:
    result = optimize_pgd(
        loss=WrappedRankingLoss(),
        tail_loss=WrappedRankingLoss(),
        initial_probabilities=jnp.full((2, 2, 3), 1 / 3),
        n_steps=1,
        tail_select_steps=1,
        binder_length=2,
        stepsize_scale=0.1 / np.sqrt(2),
        momentum=0.0,
        max_gradient_norm=1.0,
        mask=_mask([1, 1, 1], [1, 1, 1]),
        keys=jax.random.split(jax.random.key(1), 2),
    )
    assert len(result.selections) == 2
    assert all(np.isfinite(item.tail_ranking_loss) for item in result.selections)


def test_a_tail_loss_without_ranking_loss_is_rejected() -> None:
    with pytest.raises(ValueError, match="ranking_loss"):
        optimize_pgd(
            loss=SilentLoss(),
            tail_loss=SilentLoss(),
            initial_probabilities=jnp.full((1, 2, 3), 1 / 3),
            n_steps=2,
            tail_select_steps=1,
            binder_length=2,
            stepsize_scale=0.1 / np.sqrt(2),
            momentum=0.0,
            max_gradient_norm=1.0,
            mask=_mask([1, 1, 1], [1, 1, 1]),
            keys=jax.random.split(jax.random.key(1), 1),
        )


class _MarkedLoss:
    """Marks its value so a trajectory shows which loss ran at each step."""

    def __init__(self, offset: float, *, ranking: bool) -> None:
        self.offset = offset
        self.ranking = ranking
        self.pll_weight = 0.15

    def structure_term(self, sequence, *, key):
        value, auxiliary = self(sequence, key=key)
        return value, {
            "model_index": jnp.asarray(0),
            "structure": auxiliary,
        }

    def pll_term(self, sequence, *, key):
        del key
        value = jnp.zeros_like(sequence).sum()
        return value, {"toy_pll": value}

    def __call__(self, sequence, *, key):
        del key
        value = jnp.square(sequence).sum() + self.offset
        return value, {"ranking_loss": value} if self.ranking else {}


@pytest.mark.parametrize("optimizer", ["pgd", "biohub"])
def test_the_monitor_runs_only_inside_the_tail_window(optimizer) -> None:
    """The iPTM monitor must only run inside the selection window."""
    common = {
        "loss": _MarkedLoss(0.0, ranking=False),
        "tail_loss": _MarkedLoss(1000.0, ranking=True),
        "n_steps": 10,
        "tail_select_steps": 3,
        "mask": np.ones((2, 3), dtype=bool),
        "keys": jax.random.split(jax.random.key(0), 1),
    }
    if optimizer == "pgd":
        result = optimize_pgd(
            initial_probabilities=jnp.full((1, 2, 3), 1 / 3),
            binder_length=2,
            stepsize_scale=0.1,
            momentum=0.0,
            max_gradient_norm=1.0,
            **common,
        )
    else:
        result = optimize_biohub(
            logits=jnp.zeros((1, 2, 3)),
            learning_rate=0.1,
            temperature_min=0.01,
            beta=0.5,
            **common,
        )
    in_window = [point.step >= 7 for point in result.trajectory]
    assert [point.objective > 500 for point in result.trajectory] == in_window
    assert [point.ranking_loss is not None for point in result.trajectory] == in_window
    assert sum(in_window) == 3


class _BranchedLinearLoss:
    pll_weight = 0.15

    def __init__(
        self, structure_gradient: np.ndarray, pll_gradient: np.ndarray, *, ranking: bool
    ) -> None:
        self.structure_gradient = jnp.asarray(structure_gradient)
        self.pll_gradient = jnp.asarray(pll_gradient)
        self.ranking = ranking

    def structure_term(self, sequence, *, key):
        del key
        value = (sequence * self.structure_gradient).sum()
        auxiliary = {"ranking_loss": value} if self.ranking else {}
        return value, {
            "model_index": jnp.asarray(0),
            "structure": auxiliary,
        }

    def pll_term(self, sequence, *, key):
        del key
        value = (sequence * self.pll_gradient).sum()
        return value, {"toy_pll": value}


def _reference_normalize(gradient: jax.Array) -> jax.Array:
    effective_length = (jnp.square(gradient).sum(-1) > 0).sum(-1)
    norm = jnp.linalg.norm(gradient, axis=(-1, -2))
    return gradient / (norm[:, None, None] + 1e-7) * jnp.sqrt(effective_length)[:, None, None]


@pytest.mark.parametrize("excluded", [None, 2])
def test_biohub_normalizes_each_branch_after_the_softmax_pullback(excluded) -> None:
    logits = jnp.asarray([[[0.20, -0.10, 0.05], [-0.15, 0.25, 0.10]]], dtype=jnp.float32)
    structure_gradient = np.asarray([[3.0, -1.0, 0.5], [0.2, 2.0, -0.7]], dtype=np.float32)
    pll_gradient = np.asarray([[0.1, 4.0, -2.0], [-3.0, 0.5, 1.0]], dtype=np.float32)
    loss = _BranchedLinearLoss(structure_gradient, pll_gradient, ranking=False)
    tail_loss = _BranchedLinearLoss(structure_gradient, pll_gradient, ranking=True)
    mask = np.ones((2, 3), dtype=bool)
    if excluded is not None:
        mask[:, excluded] = False
    learning_rate = 0.1
    temperature_min = 0.01

    actual = optimize_biohub(
        loss=loss,
        tail_loss=tail_loss,
        logits=logits,
        mask=mask,
        n_steps=2,
        tail_select_steps=1,
        learning_rate=learning_rate,
        temperature_min=temperature_min,
        beta=0.5,
        keys=jax.random.split(jax.random.key(12), 1),
    )

    token_mask = jnp.asarray(mask[None])
    z0 = jnp.where(token_mask, logits, -1e9)
    temperature0 = temperature_min + (1 - temperature_min) * 0.5
    design0 = jax.nn.softmax(z0 / temperature0, axis=-1)

    def pullback(gradient):
        centered = (design0 * gradient).sum(-1, keepdims=True)
        return design0 * (gradient - centered) / temperature0 * token_mask

    structure_logit_gradient = pullback(jnp.asarray(structure_gradient)[None])
    pll_logit_gradient = pullback(jnp.asarray(pll_gradient)[None])
    reference_gradient = _reference_normalize(
        structure_logit_gradient
    ) + loss.pll_weight * _reference_normalize(pll_logit_gradient)
    z1 = jnp.where(
        token_mask,
        z0 - learning_rate * temperature0 * reference_gradient,
        -1e9,
    )
    expected = jax.nn.softmax(z1 / temperature_min, axis=-1)
    np.testing.assert_allclose(actual.selections[0].pssm, expected[0], rtol=2e-6, atol=1e-7)

    # Combining gradients before normalization changes the update direction.
    def mosaic_probability_normalize(gradient, scale):
        centered = gradient - gradient.mean(axis=-1, keepdims=True)
        return scale * centered / jnp.sqrt(jnp.square(centered).sum() + 1e-8)

    legacy_design_gradient = mosaic_probability_normalize(
        jnp.asarray(structure_gradient), 1.0
    ) + mosaic_probability_normalize(jnp.asarray(pll_gradient), loss.pll_weight)
    legacy_gradient = _reference_normalize(pullback(legacy_design_gradient[None]))
    assert not np.allclose(reference_gradient, legacy_gradient, rtol=1e-3, atol=1e-4)

    assert actual.selections[0].lane == 0
    assert actual.selections[0].step == 1
    if excluded is not None:
        assert np.all(actual.selections[0].pssm[..., excluded] == 0)


@pytest.mark.parametrize("momentum", [0.0, 0.25])
def test_pgd_matches_mosaic_updates_with_parameter_scan_momentum(momentum: float) -> None:
    initial = jnp.asarray([[[0.60, 0.25, 0.15], [0.20, 0.35, 0.45]]], dtype=jnp.float32)
    target = np.asarray([[0.10, 0.70, 0.20], [0.65, 0.20, 0.15]], dtype=np.float32)
    key = jax.random.key(12)
    with contextlib.redirect_stdout(io.StringIO()):
        expected, _ = simplex_APGM(
            loss_function=ToyLoss(target),
            x=initial[0],
            n_steps=3,
            stepsize=0.2,
            momentum=momentum,
            key=key,
            max_gradient_norm=1.0,
            scale=1.0,
            logspace=False,
        )
        previous, _ = simplex_APGM(
            loss_function=ToyLoss(target),
            x=initial[0],
            n_steps=2,
            stepsize=0.2,
            momentum=momentum,
            key=key,
            max_gradient_norm=1.0,
            scale=1.0,
            logspace=False,
        )
    result = optimize_pgd(
        loss=ToyLoss(target),
        tail_loss=ToyLoss(target, report_ranking=True),
        initial_probabilities=initial,
        n_steps=4,
        tail_select_steps=1,
        binder_length=2,
        stepsize_scale=0.2 / np.sqrt(2),
        momentum=momentum,
        max_gradient_norm=1.0,
        mask=np.ones((2, 3), dtype=bool),
        keys=key[None],
    )
    # Compare the same pre-update extrapolate; Mosaic returns post-update x.
    np.testing.assert_allclose(
        result.selections[0].pssm,
        expected + momentum * (expected - previous),
        rtol=2e-6, atol=2e-7,
    )
