from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import yaml

import binderopt.optimization.models as models


class _FakeModel(eqx.Module):
    esmc: object
    calls: list = eqx.field(static=True)

    def build_loss(self, **kwargs):
        self.calls.append(kwargs)
        return {"built": kwargs}


@dataclass
class _FakePLL:
    esm: object
    masking_fraction: float
    num_samples: int


def test_ensemble_and_loss_assembly_take_values_from_yaml(monkeypatch) -> None:
    profile = yaml.safe_load(Path("configs/profile/l40s.yaml").read_text(encoding="utf-8"))[
        "design"
    ]
    objective = yaml.safe_load(Path("configs/design.yaml").read_text(encoding="utf-8"))["design"][
        "objective"
    ]
    shared_esmc = object()

    def resolve(name):
        if name.endswith("load_esmc"):
            return lambda checkpoint: shared_esmc
        return lambda **kwargs: _FakeModel(esmc=object(), calls=[])

    captured = {}

    def stacked(structure_losses, esmc, pll, pll_weight):
        captured.update(
            structure_losses=structure_losses,
            esmc=esmc,
            pll=pll,
            pll_weight=pll_weight,
        )
        return captured

    monkeypatch.setattr(models, "get_method", resolve)
    monkeypatch.setattr(models, "ESMCPseudoPerplexity", _FakePLL)
    monkeypatch.setattr(models, "BinderDesignLoss", stacked)

    ensemble = models.load_design_ensemble(profile)
    assert len(ensemble.replicates) == 2
    assert ensemble.features_model.esmc is shared_esmc
    built = models.build_design_loss(
        ensemble,
        structure_loss="fixed-three-terms",
        features="features",
        inference=profile["inference"],
        objective=objective,
    )
    assert built["pll"] == _FakePLL(None, 0.15, 4)
    assert built["pll_weight"] == 0.15
    for item in built["structure_losses"]:
        assert item["built"] == {
            "loss": "fixed-three-terms",
            "features": "features",
            "recycling_steps": 1,
            "sampling_steps": 14,
            "msa_max_depth": 1024,
            "lm_dropout": 0.5,
        }


def test_design_recipe_rejects_multiple_diffusion_samples(monkeypatch) -> None:
    profile = yaml.safe_load(Path("configs/profile/l40s.yaml").read_text(encoding="utf-8"))[
        "design"
    ]
    objective = yaml.safe_load(Path("configs/design.yaml").read_text(encoding="utf-8"))["design"][
        "objective"
    ]
    inference = profile["inference"] | {"num_diffusion_samples": 2}
    with __import__("pytest").raises(ValueError, match="exactly 1"):
        models.build_design_loss(
            models.DesignEnsemble(None, (), None), None, None, inference, objective
        )
