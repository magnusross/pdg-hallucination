from __future__ import annotations

import numpy as np
import pytest

from binderopt.structure.features import entity_masks, protein_pair_mask


def test_multichain_target_is_one_group_and_ligands_are_excluded() -> None:
    entity_ids = np.array([0, 0, 1, 1, 2, 2, 3, 4, 4])
    protein = np.array([True, True, True, True, True, True, False, False, False])
    masks = entity_masks(
        entity_ids,
        protein,
        binder_entity=0,
        target_entities=(1, 2),
    )
    np.testing.assert_array_equal(masks.binder, [1, 1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(masks.target, [0, 0, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(masks.ligand, [0, 0, 0, 0, 0, 0, 1, 1, 1])

    pairs = protein_pair_mask(masks)
    assert pairs[:2, 2:6].all()
    assert pairs[2:6, :2].all()
    assert not pairs[:, 6:].any()
    assert not pairs[6:, :].any()


def test_unassigned_protein_entity_fails() -> None:
    with pytest.raises(ValueError, match="every protein token"):
        entity_masks(
            np.array([0, 1, 2]),
            np.ones(3, dtype=bool),
            binder_entity=0,
            target_entities=(1,),
        )
