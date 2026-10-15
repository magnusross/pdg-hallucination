from __future__ import annotations

from pathlib import Path

import pytest

from binderopt.structure.protenix import configure_protenix_assets


def test_protenix_asset_preflight_lists_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        configure_protenix_assets(tmp_path)
    message = str(error.value)
    assert "protenix-v2.eqx" in message
    assert "components.v20240608.cif" in message


@pytest.mark.integration
def test_staged_protenix_final_sequence_and_rbx1_ligand_features() -> None:
    import dataclasses
    import os

    import yaml

    from binderopt.structure.protenix import build_protenix_ranking_features
    from binderopt.structure.targets import target_from_config

    cache = Path(os.environ.get("MOSAIC_CACHE_DIR", "~/.cache/mosaic")).expanduser()
    configure_protenix_assets(cache)
    target_path = Path(__file__).parents[1] / "configs/target/RBX1.yaml"
    target = target_from_config(yaml.safe_load(target_path.read_text()))
    target = dataclasses.replace(
        target,
        chains=tuple(dataclasses.replace(chain, use_msa=False) for chain in target.chains),
    )
    sequence = "A" * 80
    prepared = build_protenix_ranking_features(sequence, target)
    assert prepared.features["restype"].shape[0] == len(prepared.masks.binder)
    assert prepared.masks.binder.sum() == 80
    assert prepared.masks.target.sum() == len(target.chains[0].sequence)
    assert prepared.masks.ligand.sum() == 3
    assert prepared.features["restype"][:80].argmax(-1).tolist() == [0] * 80
