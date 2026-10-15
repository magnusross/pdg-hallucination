from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from mosaic.common import LossTerm

from binderopt.optimization.objective import (
    IPTMMonitor,
    ProteinOnlyStructureLoss,
    build_structure_objective,
    harden_and_evaluate,
    hardened_esmc_log_perplexity,
    validate_contact_distance,
)
from binderopt.optimization.optimizers import TailSelection
from binderopt.structure.features import entity_masks


def _objective_config() -> dict:
    return yaml.safe_load(open("configs/design.yaml", encoding="utf-8"))["design"]["objective"]


def test_fixed_loss_builder_uses_all_yaml_values() -> None:
    loss = build_structure_objective(_objective_config())
    np.testing.assert_allclose(loss.weights, [0.5, 0.5, 0.2])
    assert loss.l[0].num_contacts_per_residue == 2
    assert loss.l[1].contact_distance == 22.0

    bad = _objective_config() | {"configurable_extra_term": {"weight": 1.0}}
    with pytest.raises(ValueError, match="fields must be exactly"):
        build_structure_objective(bad)


def test_contact_distance_must_leave_a_learned_margin() -> None:
    validate_contact_distance(np.linspace(2, 52, 64), 22.0)
    with pytest.raises(ValueError, match="outside the useful distogram range"):
        validate_contact_distance(np.linspace(2, 22, 64), 22.0)


def test_iptm_combines_target_chains_and_ignores_ligands() -> None:
    masks = entity_masks(
        np.array([0, 0, 1, 1, 2, 3]),
        np.array([True, True, True, True, True, False]),
        binder_entity=0,
        target_entities=(1, 2),
    )
    monitor = IPTMMonitor(masks)
    assert int(monitor.pair_mask.sum()) == 12

    logits = jax.random.normal(jax.random.key(1), (6, 6, 8))
    output = SimpleNamespace(pae_logits=logits, pae_bins=jnp.linspace(0.5, 31.5, 8))
    _, first = monitor(jnp.zeros((2, 20)), output, key=jax.random.key(2))
    changed = logits.at[5, :, :].set(1e4).at[:, 5, :].set(-1e4)
    _, second = monitor(
        jnp.zeros((2, 20)),
        SimpleNamespace(pae_logits=changed, pae_bins=output.pae_bins),
        key=jax.random.key(3),
    )
    np.testing.assert_allclose(first["iptm"], second["iptm"], rtol=0, atol=0)
    assert float(first["ranking_loss"]) == -float(first["iptm"])


@dataclasses.dataclass(frozen=True)
class _DistogramOutput:
    distogram_logits: jax.Array


class _SumDistogram:
    def __call__(self, sequence, output, *, key):
        del sequence, key
        value = output.distogram_logits.sum()
        return value, {"sum": value}


def test_structure_loss_sees_only_protein_tokens() -> None:
    loss = ProteinOnlyStructureLoss(_SumDistogram(), jnp.asarray([0, 1, 2]))
    base = jnp.ones((4, 4, 2))
    changed = base.at[3, :, :].set(100).at[:, 3, :].set(100)
    first, _ = loss(jnp.zeros((1, 20)), _DistogramOutput(base), key=jax.random.key(0))
    second, _ = loss(jnp.zeros((1, 20)), _DistogramOutput(changed), key=jax.random.key(0))
    assert float(first) == float(second)


class _OneHotLoss:
    def __call__(self, sequence, *, key):
        del key
        value = (sequence * jnp.arange(sequence.shape[-1])).sum()
        return value, {}


def test_hardened_loss_is_evaluated_on_written_argmax_sequence() -> None:
    selections = (
        TailSelection(0, 3, -0.7, 1.2, np.array([[0.1, 0.8, 0.1]], np.float32)),
        TailSelection(1, 2, -0.6, 1.3, np.array([[0.7, 0.2, 0.1]], np.float32)),
    )
    one_hot, values = harden_and_evaluate(
        _OneHotLoss(), selections, alphabet_size=3, keys=jax.random.split(jax.random.key(0), 2)
    )
    np.testing.assert_array_equal(one_hot.argmax(-1), [[1], [0]])
    np.testing.assert_allclose(values, [1.0, 0.0])


class _StructureStandIn(LossTerm):
    """A key-sensitive structure term, so an added monitor cannot hide a key shift."""

    def __call__(self, sequence, output, *, key):
        value = (sequence * jax.random.normal(key, ())).sum() + output.pae_logits.sum()
        return value, {"structure": value}


def test_iptm_monitor_does_not_perturb_the_design_objective() -> None:
    """The tail loss must optimize exactly the same objective as the other steps.

    `design.py` builds the tail loss as `design objective + IPTMMonitor`, so the
    monitor has to contribute exactly zero to the value, exactly zero to the
    gradient, and consume no key a preceding term would otherwise have seen.
    Otherwise the last `tail_select_steps` steps would silently optimize something
    different from the first ones.
    """
    masks = entity_masks(
        np.array([0, 0, 1, 1]),
        np.array([True, True, True, True]),
        binder_entity=0,
        target_entities=(1,),
    )
    output = SimpleNamespace(
        pae_logits=jax.random.normal(jax.random.key(4), (4, 4, 8)),
        pae_bins=jnp.linspace(0.5, 31.5, 8),
    )
    sequence = jax.random.uniform(jax.random.key(5), (2, 20))
    key = jax.random.key(6)

    plain = 1.0 * _StructureStandIn()
    monitored = plain + IPTMMonitor(masks)

    def value_of(loss, x):
        return loss(x, output, key=key)[0]

    bare_value, bare_aux = plain(sequence, output, key=key)
    full_value, full_aux = monitored(sequence, output, key=key)
    assert float(bare_value) == float(full_value)
    assert float(bare_aux[0]["structure"]) == float(full_aux[0]["structure"])
    np.testing.assert_array_equal(
        np.asarray(jax.grad(lambda x: value_of(plain, x))(sequence)),
        np.asarray(jax.grad(lambda x: value_of(monitored, x))(sequence)),
    )
    assert float(full_aux[1]["ranking_loss"]) == -float(full_aux[1]["iptm"])


class _SumPLL:
    """A `pll_term` standing in for the objective's ESMC branch."""

    def __init__(self, *, key_sensitive: bool) -> None:
        self.key_sensitive = key_sensitive

    def pll_term(self, sequence, *, key):
        value = sequence.sum()
        if self.key_sensitive:
            value = value + jax.random.uniform(key, ())
        return value, {"esmc_log_perplexity": value}


def test_harden_returns_host_side_values() -> None:
    """`design.py` reads timing off `one_hot` alone and must not block on the values.

    Regression for `0250687`, which added `hardened.block_until_ready()` and crashed
    every design run: these values have been a numpy array since the initial commit.
    """
    selections = (TailSelection(0, 1, -0.5, 1.0, np.array([[0.2, 0.7, 0.1]], np.float32)),)
    one_hot, values = harden_and_evaluate(
        _OneHotLoss(), selections, alphabet_size=3, keys=jax.random.split(jax.random.key(0), 1)
    )
    assert isinstance(values, np.ndarray) and not hasattr(values, "block_until_ready")
    assert hasattr(one_hot, "block_until_ready")


def test_hardened_perplexity_scores_each_written_sequence() -> None:
    """The recorded value must come from the objective's own PLL branch, per lane."""
    one_hot = jnp.asarray(np.stack([np.eye(4, dtype=np.float32)[[0, 1]] * scale
                                    for scale in (1.0, 2.0)]))
    values = hardened_esmc_log_perplexity(
        _SumPLL(key_sensitive=False),
        one_hot,
        keys=jax.random.split(jax.random.key(0), 2),
        batches=4,
    )
    # Deterministic branch: averaging batches must not perturb the per-lane value.
    np.testing.assert_allclose(values, [2.0, 4.0], rtol=1e-6)


def test_hardened_perplexity_averages_independent_draws() -> None:
    """Batches must use distinct keys, so more of them cut the sampling noise."""
    one_hot = jnp.zeros((64, 3, 4))
    keys = jax.random.split(jax.random.key(0), 64)
    single = hardened_esmc_log_perplexity(
        _SumPLL(key_sensitive=True), one_hot, keys=keys, batches=1
    )
    many = hardened_esmc_log_perplexity(
        _SumPLL(key_sensitive=True), one_hot, keys=keys, batches=32
    )
    assert not np.allclose(single, many)
    # Averaging distinct draws shrinks the spread toward the mean of the draw.
    assert many.std() < single.std() / 2


def test_hardened_perplexity_rejects_bad_arguments() -> None:
    one_hot = jnp.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="must be positive"):
        hardened_esmc_log_perplexity(
            _SumPLL(key_sensitive=False),
            one_hot,
            keys=jax.random.split(jax.random.key(0), 2),
            batches=0,
        )
    with pytest.raises(ValueError, match="one perplexity key"):
        hardened_esmc_log_perplexity(
            _SumPLL(key_sensitive=False),
            one_hot,
            keys=jax.random.split(jax.random.key(0), 3),
            batches=2,
        )
