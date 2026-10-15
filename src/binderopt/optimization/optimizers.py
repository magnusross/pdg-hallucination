"""The supported batched design optimizers with explicit lane-stable output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mosaic.optimizers import batched_eval, projection_simplex
from numpy.typing import NDArray

from binderopt.optimization.biohub_gradients import batched_biohub_eval
from binderopt.optimization.seeding import fold_in_step

# Sentinel for an excluded token, matching Mosaic's `biohub_optimizer`. It softmaxes
# to exactly zero and always projects to zero, and unlike -inf it cannot produce a nan.
MASKED_LOGIT = -1e9


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    lane: int
    step: int
    objective: float
    # Only the tail window evaluates the monitor, so this is None outside it.
    ranking_loss: float | None
    components: dict[str, float]
    pssm: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TailSelection:
    lane: int
    step: int
    tail_ranking_loss: float
    design_objective: float
    pssm: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    selections: tuple[TailSelection, ...]
    trajectory: tuple[TrajectoryPoint, ...]


def project_simplex(values: NDArray[Any], mask: NDArray[Any]) -> NDArray[np.float32]:
    """Project the final axis onto the probability simplex, enforcing a token mask."""
    array = np.where(mask, np.asarray(values, dtype=np.float64), MASKED_LOGIT)
    projected = projection_simplex(array.reshape(-1, array.shape[-1])).reshape(array.shape)
    return np.where(mask, projected, 0.0).astype(np.float32)


def initialize_logits(
    keys: jax.Array,
    *,
    binder_length: int,
    alphabet_size: int,
    init_scale: float,
    mask: NDArray[Any],
) -> jax.Array:
    """Draw each lane's initial logits from that lane's own key.

    Per lane rather than as one batched block, so a design's initialization does not
    depend on the batch it was placed in.
    """
    draw = jax.vmap(
        lambda key: jax.random.normal(key, (binder_length, alphabet_size), dtype=jnp.float32)
    )
    return jnp.where(jnp.asarray(mask)[None], init_scale * draw(keys), MASKED_LOGIT)


def clip_gradients(gradient: jax.Array, max_norm: float) -> jax.Array:
    """Clip each batch lane by its own Frobenius norm."""
    norms = jnp.linalg.norm(gradient, axis=(-1, -2))
    return gradient * jnp.minimum(1.0, max_norm / (norms + 1e-7))[:, None, None]


def optimize_pgd(
    *,
    loss: Any,
    tail_loss: Any,
    initial_probabilities: jax.Array,
    n_steps: int,
    tail_select_steps: int,
    binder_length: int,
    stepsize_scale: float,
    momentum: float,
    max_gradient_norm: float,
    mask: NDArray[Any],
    keys: jax.Array,
) -> OptimizationResult:
    """Lane-stable simplex PGD with optional momentum."""
    stepsize = stepsize_scale * np.sqrt(binder_length)
    if binder_length <= 0 or stepsize <= 0 or max_gradient_norm <= 0:
        raise ValueError("binder_length, stepsize_scale, and max_gradient_norm must be positive")

    token_mask = np.asarray(mask, dtype=bool)[None]
    x = jnp.asarray(project_simplex(np.asarray(initial_probabilities), token_mask))
    _check_lane_keys(keys, x.shape[0])
    recorder = _Recorder(x.shape[0], n_steps=n_steps, tail_select_steps=tail_select_steps)
    x_prev = x

    for step in range(n_steps):
        # Preserve Mosaic's simplex_APGM update order, including the pre-update
        # tail selection and per-step random keys used by the paired Biohub arm.
        design = jnp.asarray(x + momentum * (x - x_prev), dtype=jnp.float32)
        step_loss = tail_loss if recorder.in_tail(step) else loss
        values, aux, gradient = batched_eval(step_loss, design, keys)
        recorder.record(step, design, values, aux)
        gradient = clip_gradients(gradient, max_gradient_norm)
        x_new = jnp.asarray(
            project_simplex(np.asarray(design - stepsize * gradient), token_mask)
        )
        x_prev, x = x, x_new
        keys = fold_in_step(keys, step)

    return recorder.result()


def optimize_biohub(
    *,
    loss: Any,
    tail_loss: Any,
    logits: jax.Array,
    n_steps: int,
    tail_select_steps: int,
    learning_rate: float,
    temperature_min: float,
    beta: float,
    mask: NDArray[Any],
    keys: jax.Array,
) -> OptimizationResult:
    """Minimal lane-stable implementation of Mosaic/Biohub Algorithm 11."""
    token_mask = jnp.asarray(np.asarray(mask, dtype=bool)[None])
    z = jnp.where(token_mask, jnp.asarray(logits), MASKED_LOGIT)
    _check_lane_keys(keys, z.shape[0])
    velocity = jnp.zeros_like(z)
    recorder = _Recorder(z.shape[0], n_steps=n_steps, tail_select_steps=tail_select_steps)

    for step in range(n_steps):
        temperature = float(
            temperature_min
            + (1.0 - temperature_min) * 0.5 * (1.0 + np.cos(np.pi * (step + 1) / n_steps))
        )
        design = jax.nn.softmax(z / temperature, axis=-1)
        step_loss = tail_loss if recorder.in_tail(step) else loss
        if not all(
            hasattr(step_loss, attribute)
            for attribute in ("structure_term", "pll_term", "pll_weight")
        ):
            raise TypeError("Biohub losses must expose separate structure and PLL terms")
        values, aux, logit_gradient = batched_biohub_eval(
            step_loss,
            z,
            jnp.asarray(temperature, dtype=z.dtype),
            jnp.broadcast_to(token_mask, z.shape),
            keys,
        )
        recorder.record(step, design, values, aux)

        velocity = beta * velocity + (learning_rate * temperature) * logit_gradient
        z = jnp.where(token_mask, z - velocity, MASKED_LOGIT)
        keys = fold_in_step(keys, step)

    return recorder.result()


def _check_lane_keys(keys: jax.Array, batch_size: int) -> None:
    if keys.shape[:1] != (batch_size,):
        raise ValueError("one optimizer key per batch lane is required")


class _Recorder:
    """Per-lane trajectory and tail-window selection.

    Owns every rule that must be identical across both optimizer arms: which steps
    are in the tail window, that lanes are recorded in place and never reordered,
    that an iterate is only ever selected inside the window, and that every lane
    finishes with a selection.
    """

    def __init__(self, batch_size: int, *, n_steps: int, tail_select_steps: int) -> None:
        if not 0 < tail_select_steps <= n_steps:
            raise ValueError("tail_select_steps must be in (0, n_steps]")
        self.tail_start = n_steps - tail_select_steps
        self.trajectory: list[TrajectoryPoint] = []
        self.best: list[TailSelection | None] = [None] * batch_size

    def in_tail(self, step: int) -> bool:
        """Whether this step evaluates the tail loss and its iPTM monitor."""
        return step >= self.tail_start

    def record(self, step: int, design: jax.Array, values: jax.Array, aux: Any) -> None:
        batch_size = len(self.best)
        in_tail = self.in_tail(step)
        ranking = _ranking_loss(aux) if in_tail else None
        if in_tail and ranking is None:
            raise ValueError("tail loss must expose a ranking_loss auxiliary value")

        for lane in range(batch_size):
            objective = float(values[lane])
            rank_value = None if ranking is None else float(ranking[lane])
            self.trajectory.append(
                TrajectoryPoint(
                    lane=lane,
                    step=step,
                    objective=objective,
                    ranking_loss=rank_value,
                    components=_scalar_aux(aux, lane, batch_size),
                    pssm=np.asarray(design[lane], dtype=np.float32).copy(),
                )
            )
            current = self.best[lane]
            if rank_value is not None and (
                current is None or rank_value < current.tail_ranking_loss
            ):
                self.best[lane] = TailSelection(
                    lane=lane,
                    step=step,
                    tail_ranking_loss=rank_value,
                    design_objective=objective,
                    pssm=np.asarray(design[lane], dtype=np.float32).copy(),
                )

    def result(self) -> OptimizationResult:
        selections = [item for item in self.best if item is not None]
        if len(selections) != len(self.best):  # protected by the tail-window validation
            raise RuntimeError("optimizer completed without selecting every lane")
        return OptimizationResult(tuple(selections), tuple(self.trajectory))


def _ranking_loss(aux: Any) -> Any | None:
    """Find the iPTM monitor's batched `ranking_loss` leaf anywhere in the aux tree."""
    for path, leaf in jax.tree_util.tree_leaves_with_path(aux):
        if any(getattr(entry, "key", None) == "ranking_loss" for entry in path):
            return leaf
    return None


def _scalar_aux(aux: Any, lane: int, batch_size: int) -> dict[str, float]:
    """Flatten one lane's per-batch scalar aux leaves into dotted trajectory keys."""
    out: dict[str, float] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(aux):
        array = np.asarray(leaf)
        if array.shape == (batch_size,):
            out[jax.tree_util.keystr(path, simple=True, separator=".")] = float(array[lane])
    return out
