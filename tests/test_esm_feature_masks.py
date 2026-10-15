from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from binderopt.structure.features import _masks_from_writer


def _chain(*indices):
    return SimpleNamespace(tokens=[SimpleNamespace(token_index=index) for index in indices])


def test_writer_masks_preserve_multichain_target_and_exclude_ligand_tokens() -> None:
    masks = _masks_from_writer(
        [_chain(0, 1), _chain(2, 3), _chain(4), _chain(5, 6, 7)],
        protein_entities=3,
    )
    np.testing.assert_array_equal(masks.binder, [1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(masks.target, [0, 0, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(masks.ligand, [0, 0, 0, 0, 0, 1, 1, 1])
