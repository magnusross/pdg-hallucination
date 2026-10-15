from __future__ import annotations

from pathlib import Path

import yaml

from binderopt.structure.targets import apply_msa_policy, target_from_config

TARGET_DIR = Path(__file__).parents[1] / "configs" / "target"


def test_target_configs_are_valid() -> None:
    for path in sorted(TARGET_DIR.glob("*.yaml")):
        target = target_from_config(yaml.safe_load(path.read_text()))
        assert target.target_id == path.stem
        assert target.chains
        assert all(
            chain.sequence and set(chain.sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
            for chain in target.chains
        )
        entities = [chain.entity_id for chain in target.chains] + [
            ligand.entity_id for ligand in target.ligands
        ]
        assert len(entities) == len(set(entities))
        assert all(ligand.count > 0 for ligand in target.ligands)


def test_paper_multichain_ligand_and_residue_mask_constructs() -> None:
    targets = {
        path.stem: target_from_config(yaml.safe_load(path.read_text()))
        for path in TARGET_DIR.glob("*.yaml")
    }
    for name, chains in (("VEGFA", 2), ("TNFa", 3)):
        assert len(targets[name].chains) == chains
        assert len({chain.sequence for chain in targets[name].chains}) == 1
    assert [(item.ccd, item.count) for item in targets["RBX1"].ligands] == [("ZN", 3)]
    assert targets["TREM2"].self_consistency_residue_ranges == (((1, 112),),)
    assert targets["RBX1"].self_consistency_residue_ranges is None


def test_msa_policy_does_not_mutate_target() -> None:
    target = target_from_config(yaml.safe_load((TARGET_DIR / "VEGFA.yaml").read_text()))
    downgraded = apply_msa_policy(target.chains, supports_msa=False)
    assert all(chain.use_msa for chain in target.chains)
    assert not any(chain.use_msa for chain in downgraded)
    assert apply_msa_policy(target.chains, supports_msa=True) is target.chains
