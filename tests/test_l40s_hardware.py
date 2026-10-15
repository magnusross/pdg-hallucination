from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

if os.environ.get("BINDEROPT_RUN_L40S_HARDWARE") != "1":
    pytest.skip("set BINDEROPT_RUN_L40S_HARDWARE=1 on an L40S", allow_module_level=True)


@pytest.mark.hardware
def test_l40s_design_checkpoints_report_dead_confidence_heads() -> None:
    import equinox as eqx
    import jax
    import numpy as np
    import yaml
    from mosaic.common import TOKENS

    from binderopt.optimization.models import load_design_ensemble
    from binderopt.optimization.objective import IPTMMonitor
    from binderopt.structure.features import build_esm_binder_features
    from binderopt.structure.targets import target_from_config

    root = Path(__file__).parents[1]
    profile = yaml.safe_load((root / "configs/profile/l40s.yaml").read_text())
    target = target_from_config(
        yaml.safe_load((root / "configs/target/4ZQK.yaml").read_text())
    )
    binder_length = int(target.default_binder_length)
    sequences = (
        "A" * binder_length,
        ("WYDEKR" * (binder_length // 6 + 1))[:binder_length],
    )

    ensemble = load_design_ensemble(profile["design"])
    prepared = build_esm_binder_features(
        ensemble.features_model,
        binder_length,
        target,
        supports_msa=False,
    )
    monitor = IPTMMonitor(prepared.masks)
    token_index = {letter: index for index, letter in enumerate(TOKENS)}
    report: dict[str, object] = {"sequences": sequences, "checkpoints": {}}

    for spec, replicate in zip(
        profile["design"]["structure_models"], ensemble.replicates, strict=True
    ):
        model = eqx.tree_at(
            lambda item: item.esmc,
            replicate,
            ensemble.esmc,
            is_leaf=lambda value: value is None,
        )
        scores = []
        for sequence in sequences:
            pssm = jax.nn.one_hot(
                jax.numpy.asarray([token_index[letter] for letter in sequence]),
                len(TOKENS),
            )
            key = jax.random.key(20250824)
            output = model.model_output(
                PSSM=pssm,
                features=prepared.features,
                recycling_steps=int(profile["design"]["inference"]["recycling_steps"]),
                sampling_steps=int(profile["design"]["inference"]["sampling_steps"]),
                msa_max_depth=int(profile["design"]["inference"]["msa_max_depth"]),
                key=key,
            )
            _, metrics = monitor(pssm, output, key=key)
            scores.append(float(np.asarray(metrics["iptm"])))
        variance = float(np.var(scores))
        report["checkpoints"][spec["checkpoint"]] = {
            "scores": scores,
            "variance": variance,
        }
        assert variance <= 1e-12

    assert profile["scientific_validity"] == "development_only"
    print(json.dumps(report, sort_keys=True))
