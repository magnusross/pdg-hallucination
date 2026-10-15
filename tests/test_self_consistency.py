from __future__ import annotations

import gemmi
import numpy as np
import pytest

from binderopt.ranking.self_consistency import score_self_consistency


def _pdb(
    chains: list[tuple[str, list[tuple[float, float, float]], str]],
    *,
    ligand: bool = False,
    sidechains: bool = False,
) -> str:
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    atom_offsets = {
        "N": (-0.5, 0.0, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (0.5, 0.0, 0.0),
        "O": (0.8, 0.2, 0.0),
    }
    for chain_id, positions, residue_name in chains:
        chain = gemmi.Chain(chain_id)
        for index, center in enumerate(positions, start=1):
            residue = gemmi.Residue()
            residue.name = residue_name
            residue.seqid = gemmi.SeqId(index, " ")
            offsets = dict(atom_offsets)
            if sidechains and residue_name != "GLY":
                offsets["CB"] = (0.0, 6.0, 0.0)
            for atom_name, offset in offsets.items():
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(*(np.asarray(center) + np.asarray(offset)))
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    if ligand:
        chain = gemmi.Chain("Z")
        residue = gemmi.Residue()
        residue.name = "ATP"
        residue.seqid = gemmi.SeqId(1, " ")
        atom = gemmi.Atom()
        atom.name = "P"
        atom.element = gemmi.Element("P")
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    return structure.make_pdb_string()


def test_standard_dockq_matches_mutated_residues_by_position_and_excludes_ligands() -> None:
    reference = _pdb(
        [
            ("A", [(0, 0, 0), (2, 0, 0)], "ALA"),
            ("B", [(0, 4, 0), (2, 4, 0), (4, 4, 0)], "ALA"),
        ],
        ligand=True,
    )
    prediction = _pdb(
        [
            ("X", [(0, 0, 0), (2, 0, 0)], "GLY"),
            ("Y", [(0, 4, 0), (2, 4, 0), (4, 4, 0)], "ALA"),
        ]
    )

    metric = score_self_consistency(
        reference, prediction, binder_length=2, target_sequences=["AAA"]
    )

    assert metric["dockq"] == pytest.approx(1.0)
    assert metric["fnat"] == 1.0
    assert metric["interface_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert metric["target_aligned_binder_backbone_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert metric["dockq_receptor"] == "target"
    assert metric["dockq_ligand"] == "binder"


def test_standard_fnat_reflects_sidechain_contact_changes() -> None:
    reference = _pdb(
        [
            ("A", [(0, 0, 0)], "ALA"),
            ("B", [(0, 10, 0), (20, 10, 0)], "ALA"),
        ],
        sidechains=True,
    )
    prediction = _pdb(
        [
            ("X", [(0, 0, 0)], "GLY"),
            ("Y", [(0, 10, 0), (20, 10, 0)], "ALA"),
        ],
        sidechains=True,
    )

    metric = score_self_consistency(reference, prediction, binder_length=1, target_sequences=["AA"])

    assert metric["fnat"] == 0.0
    assert metric["dockq"] == pytest.approx(2 / 3)
    assert metric["target_aligned_binder_backbone_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_identical_target_chains_choose_best_standard_dockq_permutation() -> None:
    binder = [(0, 0, 0), (2, 0, 0)]
    near = [(0, 4, 0), (2, 4, 0)]
    far = [(40, 4, 0), (42, 4, 0)]
    reference = _pdb([("A", binder, "ALA"), ("B", near, "ALA"), ("C", far, "ALA")])
    prediction = _pdb([("X", binder, "GLY"), ("Y", far, "ALA"), ("Z", near, "ALA")])

    metric = score_self_consistency(
        reference, prediction, binder_length=2, target_sequences=["AA", "AA"]
    )

    assert metric["dockq"] == pytest.approx(1.0)
    assert metric["target_chain_mapping"] == [
        {"reference": "B", "prediction": "Z"},
        {"reference": "C", "prediction": "Y"},
    ]


def test_standard_dockq_records_when_short_target_is_ligand() -> None:
    structure = _pdb(
        [
            ("A", [(0, 0, 0), (2, 0, 0), (4, 0, 0)], "ALA"),
            ("B", [(2, 4, 0)], "ALA"),
        ]
    )

    metric = score_self_consistency(structure, structure, binder_length=3, target_sequences=["A"])

    assert metric["dockq"] == pytest.approx(1.0)
    assert metric["dockq_receptor"] == "binder"
    assert metric["dockq_ligand"] == "target"
    assert metric["target_aligned_binder_backbone_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_reference_without_interface_contacts_scores_zero() -> None:
    structure = _pdb([("A", [(0, 0, 0)], "ALA"), ("B", [(100, 0, 0)], "ALA")])
    metric = score_self_consistency(structure, structure, binder_length=1, target_sequences=["A"])

    assert metric["dockq"] == 0.0
    assert metric["dockq_status"] == "no_reference_contacts"
    assert metric["fnat"] == 0.0
    assert metric["interface_contact_counts"]["reference"] == 0
    assert metric["interface_rmsd"] is None
    assert metric["dockq_ligand_rmsd"] is None
    assert metric["target_aligned_binder_backbone_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_layout_validation_fails_loudly() -> None:
    structure = _pdb([("A", [(0, 0, 0)], "ALA"), ("B", [(100, 0, 0)], "ALA")])
    with pytest.raises(ValueError, match="chain lengths differ"):
        score_self_consistency(structure, structure, binder_length=2, target_sequences=["A"])


def test_fixed_target_ranges_are_primary_and_retain_unmasked_metrics() -> None:
    binder = [(0, 0, 0), (2, 0, 0)]
    core = [(0, 4, 0), (2, 4, 0), (4, 4, 0)]
    reference = _pdb([("A", binder, "ALA"), ("B", [*core, (0, 20, 0), (2, 20, 0)], "ALA")])
    prediction = _pdb([("X", binder, "GLY"), ("Y", [*core, (30, 20, 0), (32, 20, 0)], "ALA")])

    metric = score_self_consistency(
        reference,
        prediction,
        binder_length=2,
        target_sequences=["AAAAA"],
        target_residue_ranges=[[[1, 3]]],
    )

    assert metric["dockq"] == pytest.approx(1.0)
    assert metric["target_aligned_binder_backbone_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert metric["self_consistency_target_residue_ranges"] == [[[1, 3]]]
    assert metric["self_consistency_range_indexing"] == "one_based_inclusive"
    assert metric["unmasked_self_consistency"]["dockq"] < metric["dockq"]
    assert (
        metric["unmasked_self_consistency"]["target_aligned_binder_backbone_rmsd"]
        > metric["target_aligned_binder_backbone_rmsd"]
    )


def test_target_ranges_are_validated_against_sequence_positions() -> None:
    structure = _pdb([("A", [(0, 0, 0)], "ALA"), ("B", [(0, 4, 0)], "ALA")])
    with pytest.raises(ValueError, match="one-based"):
        score_self_consistency(
            structure,
            structure,
            binder_length=1,
            target_sequences=["A"],
            target_residue_ranges=[[[0, 1]]],
        )
