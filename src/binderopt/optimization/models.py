"""Assembly of the one supported shared-ESMC design ensemble."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
from hydra.utils import get_method
from mosaic.losses.esmc import ESMCPseudoPerplexity
from mosaic.losses.esmfold2 import StackedEsmFold2PLL


@dataclass(frozen=True, slots=True)
class DesignEnsemble:
    esmc: Any
    replicates: tuple[Any, ...]
    features_model: Any


class BinderDesignLoss(StackedEsmFold2PLL):
    """Mosaic's shared-ESMC loss with raw branch access for Biohub updates.

    Mosaic owns stacking, shared-buffer wiring, and the ordinary callable used
    by PGD. These accessors only defer branch combination long enough for
    Biohub to normalize each branch after its softmax pullback.
    """

    def structure_term(self, sequence, *, key):
        """Evaluate the unnormalized structural branch."""
        pick_key, structure_key, _ = jax.random.split(key, 3)
        model_index = jax.random.randint(pick_key, (), 0, self.n)
        model = eqx.combine(
            jax.tree_util.tree_map(lambda value: value[model_index], self.stacked),
            self.static,
        )
        model = eqx.tree_at(
            lambda item: item.esmc,
            model,
            self.esmc.esmc,
            is_leaf=lambda value: value is None,
        )
        value, auxiliary = model(sequence, key=structure_key)
        return value, {"model_index": model_index, "structure": auxiliary}

    def pll_term(self, sequence, *, key):
        """Evaluate the unnormalized language-model branch."""
        _, _, pll_key = jax.random.split(key, 3)
        pll = eqx.tree_at(
            lambda item: item.esm,
            self.pll,
            self.esmc,
            is_leaf=lambda value: value is None,
        )
        return pll(sequence, key=pll_key)


def load_design_ensemble(config: Mapping[str, Any]) -> DesignEnsemble:
    """Load two reviewed YAML model specs while keeping one shared ESMC."""
    language_model = config["language_model"]
    load_esmc = get_method(language_model["factory"])
    esmc = load_esmc(language_model["checkpoint"])

    replicates = []
    for spec in config["structure_models"]:
        factory = get_method(spec["factory"])
        metadata = {"factory", "checkpoint", "name"}
        arguments = {key: value for key, value in spec.items() if key not in metadata}
        model = factory(**arguments)
        if not hasattr(model, "esmc") or not hasattr(model, "build_loss"):
            raise TypeError(f"{spec['factory']} does not provide the ESMFold2 design contract")
        replicates.append(eqx.tree_at(lambda item: item.esmc, model, None))
    if len(replicates) != 2:
        raise ValueError("the design recipe requires exactly two structural replicates")
    features_model = eqx.tree_at(
        lambda item: item.esmc,
        replicates[0],
        esmc,
        is_leaf=lambda value: value is None,
    )
    return DesignEnsemble(esmc, tuple(replicates), features_model)


def build_design_loss(
    ensemble: DesignEnsemble,
    structure_loss: Any,
    features: Any,
    inference: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> Any:
    """Fuse the fixed structure block and explicitly configured ESMC PLL."""
    if int(inference["num_diffusion_samples"]) != 1:
        raise ValueError("design num_diffusion_samples must be exactly 1")
    build_arguments = {
        "recycling_steps": int(inference["recycling_steps"]),
        "sampling_steps": int(inference["sampling_steps"]),
        "msa_max_depth": int(inference["msa_max_depth"]),
        "lm_dropout": float(inference["lm_dropout"]),
    }
    structure_losses = [
        model.build_loss(loss=structure_loss, features=features, **build_arguments)
        for model in ensemble.replicates
    ]
    pll_config = objective["esmc_pll"]
    pll = ESMCPseudoPerplexity(
        None,
        masking_fraction=float(pll_config["masking_fraction"]),
        num_samples=int(pll_config["num_samples"]),
    )
    return BinderDesignLoss(
        structure_losses,
        ensemble.esmc,
        pll,
        pll_weight=float(pll_config["weight"]),
    )
