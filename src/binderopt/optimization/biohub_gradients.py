"""Branch-aware JAX differentiation for the Biohub optimizer."""

import equinox as eqx
import jax
import jax.numpy as jnp


@jax.custom_vjp
def _branch_probabilities(logits, temperature, token_mask, pll_weight):
    """Share one softmax while keeping the two branch pullbacks separate."""
    del token_mask, pll_weight
    design = jax.nn.softmax(logits / temperature, axis=-1)
    return design, design


def _branch_probabilities_fwd(logits, temperature, token_mask, pll_weight):
    design = jax.nn.softmax(logits / temperature, axis=-1)
    return (design, design), (design, temperature, token_mask, pll_weight)


def _branch_probabilities_bwd(residuals, gradients):
    design, temperature, token_mask, pll_weight = residuals
    structure_gradient, weighted_pll_gradient = gradients

    def pullback_and_normalize(probability_gradient):
        # Apply the softmax Jacobian before Biohub's per-branch normalization.
        centered = (design * probability_gradient).sum(-1, keepdims=True)
        gradient = design * (probability_gradient - centered) / temperature * token_mask
        effective_length = (jnp.square(gradient).sum(-1) > 0).sum()
        return gradient / (jnp.linalg.norm(gradient) + 1e-7) * jnp.sqrt(effective_length)

    # The PLL cotangent already includes its outer loss weight.
    gradient = pullback_and_normalize(structure_gradient)
    gradient += pll_weight * pullback_and_normalize(weighted_pll_gradient / pll_weight)
    return gradient, None, None, None


_branch_probabilities.defvjp(_branch_probabilities_fwd, _branch_probabilities_bwd)


@eqx.filter_jit
def batched_biohub_eval(loss, logits, temperature, token_masks, keys):
    """Evaluate both branches once and return their normalized logit gradient."""

    def single(z, token_mask, key):
        def value(logits):
            structure_design, pll_design = _branch_probabilities(
                logits, temperature, token_mask, jnp.asarray(loss.pll_weight)
            )
            structure_value, structure_aux = loss.structure_term(structure_design, key=key)
            pll_value, pll_aux = loss.pll_term(pll_design, key=key)
            auxiliary = structure_aux | {"pll": pll_aux}
            return structure_value + loss.pll_weight * pll_value, auxiliary

        (loss_value, auxiliary), gradient = eqx.filter_value_and_grad(value, has_aux=True)(z)
        return jnp.nan_to_num(loss_value, nan=1e6), auxiliary, jnp.nan_to_num(gradient)

    return jax.vmap(single)(logits, token_masks, keys)
