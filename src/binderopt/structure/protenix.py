"""Offline final-sequence Protenix features with persistent target MSAs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from binderopt.assets.msa import protenix_msa_dir
from binderopt.structure.features import PreparedComplex, entity_masks
from binderopt.structure.targets import Target, apply_msa_policy


def protenix_asset_paths(cache_root: Path) -> dict[str, Path]:
    root = cache_root / "protenix"
    return {
        "components": root / "components.v20240608.cif",
        "rdkit_components": root / "components.v20240608.cif.rdkit_mol.pkl",
        "clusters": root / "clusters-by-entity-40.txt",
        "model": root / "protenix-v2.eqx",
        "skeleton": root / "protenix-v2.skeleton.pkl",
    }


def configure_protenix_assets(cache_root: Path) -> dict[str, Path]:
    """Point Protenix chemistry/model globals at Mosaic's reviewed asset root."""
    paths = protenix_asset_paths(cache_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Protenix assets: " + ", ".join(missing))

    import protenix.configs.configs_data as configs_data
    import protenix.data.ccd as ccd

    ccd.COMPONENTS_FILE = str(paths["components"])
    ccd.RKDIT_MOL_PKL = paths["rdkit_components"]
    configs_data.data_configs["ccd_components_file"] = str(paths["components"])
    configs_data.data_configs["ccd_components_rdkit_mol_file"] = str(
        paths["rdkit_components"]
    )
    configs_data.data_configs["pdb_cluster_file"] = str(paths["clusters"])
    ccd.biotite_load_ccd_cif.cache_clear()
    ccd.get_component_atom_array.cache_clear()
    return paths


def build_protenix_ranking_features(sequence: str, target: Target) -> PreparedComplex:
    """Rebuild every sequence-dependent field for a finished binder complex."""
    from protenix.data.json_to_feature import SampleDictToFeatures
    from protenix.data.msa_featurizer import InferenceMSAFeaturizer
    from protenix.data.template.template_featurizer import _make_dummy_msa
    from protenix.data.utils import data_type_transform

    chains = [
        (sequence, False),
        *(
            (chain.sequence, chain.use_msa)
            for chain in apply_msa_policy(target.chains, supports_msa=True)
        ),
    ]
    with TemporaryDirectory(prefix="binderopt_protenix_") as temporary:
        entities: list[dict[str, Any]] = []
        for index, (chain_sequence, use_msa) in enumerate(chains):
            if use_msa:
                msa_dir = protenix_msa_dir(chain_sequence)
            else:
                msa_dir = Path(temporary) / str(index)
                _make_dummy_msa(chain_sequence, str(msa_dir))
            entities.append(
                {
                    "proteinChain": {
                        "sequence": chain_sequence,
                        "count": 1,
                        "msa": {
                            "precomputed_msa_dir": str(msa_dir),
                            "pairing_db": "uniref100",
                        },
                    }
                }
            )
        entities.extend(
            {
                "ligand": {
                    "ligand": f"CCD_{ligand.ccd.removeprefix('CCD_')}",
                    "count": ligand.count,
                }
            }
            for ligand in target.ligands
        )

        features, atom_array, token_array = SampleDictToFeatures(
            {"name": target.target_id, "sequences": entities}
        ).get_feature_dict()
        features.pop("constraint_feature", None)
        features["distogram_rep_atom_mask"] = np.asarray(
            atom_array.distogram_rep_atom_mask, dtype=np.int64
        )
        entity_to_asym: dict[str, set[int]] = defaultdict(set)
        for entity_id, asym_id in zip(
            atom_array.label_entity_id, atom_array.asym_id_int, strict=True
        ):
            entity_to_asym[str(entity_id)].add(int(asym_id))
        msa_features = InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=entities,
            entity_to_asym_id=dict(entity_to_asym),
            token_array=token_array,
            atom_array=atom_array,
        )
        if msa_features:
            features.update(msa_features)
        features = data_type_transform(features)
        features["atom_rep_atom_idx"] = np.flatnonzero(
            features["distogram_rep_atom_mask"]
        )

    asym_id = np.asarray(features["asym_id"])
    protein_entities = len(chains)
    masks = entity_masks(
        asym_id,
        np.isin(asym_id, np.arange(protein_entities)),
        binder_entity=0,
        target_entities=tuple(range(1, protein_entities)),
    )
    return PreparedComplex(features, atom_array, masks)
