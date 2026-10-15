from __future__ import annotations

import json

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binderopt.ranking.mpnn as rank_mpnn
from binderopt.ranking.mpnn import mpnn_structure_input, redesign_sequence, run_soluble_mpnn_cell
from binderopt.ranking.protein_mpnn import MPNNInput, autoregressive_inverse_fold


def _reference_pdb(*, ligand: bool = True) -> str:
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for chain_id, length, residue_name, y in (("A", 2, "ALA", 0.0), ("B", 2, "GLY", 5.0)):
        chain = gemmi.Chain(chain_id)
        for index in range(length):
            residue = gemmi.Residue()
            residue.name = residue_name
            residue.seqid = gemmi.SeqId(index + 1, " ")
            for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(index * 2 + atom_index * 0.2, y, 0)
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


def test_mpnn_input_strips_ligands_and_preserves_all_target_chains() -> None:
    output = mpnn_structure_input(_reference_pdb(), binder_length=2, target_sequences=["GG"])
    assert output.backbone_coordinates.shape == (4, 4, 3)
    np.testing.assert_array_equal(output.asym_id, [0, 0, 1, 1])
    assert output.full_sequence.shape == (4, 20)


def test_redesign_uses_one_key_and_no_cys_bias(monkeypatch) -> None:
    calls = []

    def fake_inverse_fold(mpnn, binder_length, output, temp, key, bias):
        calls.append((mpnn, binder_length, output, temp, key, np.asarray(bias)))
        return jnp.asarray([0, 0])

    monkeypatch.setattr(
        "binderopt.ranking.protein_mpnn.autoregressive_inverse_fold", fake_inverse_fold
    )
    key = jax.random.key(7)
    sequence = redesign_sequence(
        object(),
        _reference_pdb(),
        binder_length=2,
        target_sequences=["GG"],
        temperature=0.2,
        excluded_amino_acids=["C"],
        key=key,
    )
    assert sequence == "AA"
    assert len(calls) == 1
    from mosaic.common import TOKENS

    assert np.isneginf(calls[0][-1][:, TOKENS.index("C")]).all()
    np.testing.assert_array_equal(jax.random.key_data(calls[0][4]), jax.random.key_data(key))


class _ToyMPNN(eqx.Module):
    def encode(self, **kwargs):
        return (), (), ()

    def decode(self, *, S, decoding_order, **kwargs):
        from mosaic.losses.protein_mpnn import boltz_to_mpnn_matrix

        token_matrix = jnp.asarray(boltz_to_mpnn_matrix())
        tokens = (S @ token_matrix.T).argmax(-1)
        order = jnp.argsort(decoding_order)
        ranks = jnp.zeros_like(order).at[order].set(jnp.arange(order.size))
        previous = ranks[None, :] < ranks[:, None]
        centers = (previous * (tokens + 1)).sum(-1) % token_matrix.shape[0]
        token_ids = jnp.arange(token_matrix.shape[0])
        logits = -0.2 * jnp.abs(token_ids[None, :] - centers[:, None])
        return (logits @ token_matrix)[None]


def _sequential_sample(mpnn, binder_length, output, temp, key, bias):
    from mosaic.losses.protein_mpnn import _per_chain_residue_idx, boltz_to_mpnn_matrix

    total_length = output.full_sequence.shape[0]
    mask = jnp.ones(total_length, dtype=jnp.int32)
    encode_key, order_key, sample_key = jax.random.split(key, 3)
    h_V, h_E, E_idx = mpnn.encode(
        X=output.backbone_coordinates,
        mask=mask,
        residue_idx=_per_chain_residue_idx(output.asym_id, output.residue_idx),
        chain_encoding_all=output.asym_id,
        key=encode_key,
    )
    decoding_order = jax.random.uniform(order_key, (total_length,))
    decoding_order = decoding_order.at[:binder_length].add(2.0)
    binder_order = jnp.argsort(decoding_order)[-binder_length:]
    token_matrix = jnp.asarray(boltz_to_mpnn_matrix())
    sequence = output.full_sequence.at[:binder_length].set(0)
    for position in np.asarray(binder_order):
        sample_key, residue_key = jax.random.split(sample_key)
        logits = mpnn.decode(
            S=sequence @ token_matrix,
            h_V=h_V,
            h_E=h_E,
            E_idx=E_idx,
            mask=mask,
            decoding_order=decoding_order,
        )[0, position] @ token_matrix.T
        logits += bias[position]
        residue = jax.random.categorical(residue_key, logits / temp)
        sequence = sequence.at[position].set(jax.nn.one_hot(residue, 20))
    return sequence[:binder_length].argmax(-1)


def test_autoregressive_sampler_matches_sequential_reference() -> None:
    binder_length = 3
    key = jax.random.key(11)
    output = MPNNInput(
        backbone_coordinates=jnp.zeros((4, 4, 3)),
        full_sequence=jnp.vstack((jnp.zeros((3, 20)), jax.nn.one_hot(7, 20))),
        asym_id=jnp.asarray([0, 0, 0, 1]),
        residue_idx=jnp.asarray([0, 1, 2, 0]),
    )
    bias = jnp.zeros((binder_length, 20)).at[:, 1].set(-jnp.inf)
    expected = _sequential_sample(_ToyMPNN(), binder_length, output, 0.4, key, bias)
    actual = autoregressive_inverse_fold(
        _ToyMPNN(), binder_length, output, temp=0.4, key=key, bias=bias
    )
    np.testing.assert_array_equal(actual, expected)


def test_mpnn_shard_is_deterministic(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "design_references/design_000/design_a.pdb"
    reference.parent.mkdir(parents=True)
    reference.write_text(_reference_pdb())
    result = {
        "binder_length": 2,
        "config": {
            "target": {"chains": [{"sequence": "GG"}]},
        },
        "designs": [
            {
                "design_index": 0,
                "optimizer_sequence": "AR",
                "selected_reference": {
                    "model_name": "design_a",
                    "structure": "design_references/design_000/design_a.pdb",
                },
            }
        ],
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    mpnn_keys = []
    esmc_keys = []

    def fake_redesign(*args, **kwargs):
        mpnn_keys.append(kwargs["key"])
        return "DD"

    def fake_score(sequence, key):
        esmc_keys.append(key)
        return 1.5

    monkeypatch.setattr(rank_mpnn, "redesign_sequence", fake_redesign)
    spec = {
        "factory": "fixed",
        "checkpoint": "soluble",
        "temperature": 0.1,
        "excluded_amino_acids": ["C"],
    }
    first = run_soluble_mpnn_cell(
        tmp_path,
        recipe_name="full",
        mpnn_spec=spec,
        ranking_seed=9,
        mpnn=object(),
        score_sequence=fake_score,
    )
    second = run_soluble_mpnn_cell(
        tmp_path,
        recipe_name="full",
        mpnn_spec=spec,
        ranking_seed=9,
        mpnn=object(),
        score_sequence=fake_score,
    )
    record = first["designs"][0]
    assert record["optimizer_sequence"] == "AR"
    assert record["sequence"] == "DD"
    assert record["source_model"] == "design_a"
    assert json.loads((tmp_path / "ranking/soluble_mpnn.json").read_text()) == second
    np.testing.assert_array_equal(
        jax.random.key_data(mpnn_keys[0]), jax.random.key_data(mpnn_keys[1])
    )
    np.testing.assert_array_equal(
        jax.random.key_data(esmc_keys[0]), jax.random.key_data(esmc_keys[1])
    )
    assert not np.array_equal(
        jax.random.key_data(mpnn_keys[0]), jax.random.key_data(esmc_keys[0])
    )


def test_mpnn_reference_target_sequence_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="target sequences differ"):
        mpnn_structure_input(_reference_pdb(), binder_length=2, target_sequences=["AA"])


def test_mpnn_ranks_only_requested_design_indices(tmp_path, monkeypatch) -> None:
    designs = []
    for design_index in range(3):
        reference = tmp_path / f"design_references/design_{design_index:03d}/design_a.pdb"
        reference.parent.mkdir(parents=True)
        reference.write_text(_reference_pdb())
        designs.append(
            {
                "design_index": design_index,
                "optimizer_sequence": "AR",
                "selected_reference": {
                    "model_name": "design_a",
                    "structure": str(reference.relative_to(tmp_path)),
                },
            }
        )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "binder_length": 2,
                "config": {"target": {"chains": [{"sequence": "GG"}]}},
                "designs": designs,
            }
        )
    )
    monkeypatch.setattr(rank_mpnn, "redesign_sequence", lambda *args, **kwargs: "DD")
    result = run_soluble_mpnn_cell(
        tmp_path,
        recipe_name="full",
        mpnn_spec={
            "factory": "fixed",
            "checkpoint": "soluble",
            "temperature": 0.1,
            "excluded_amino_acids": ["C"],
        },
        ranking_seed=9,
        mpnn=object(),
        score_sequence=lambda sequence, key: 1.5,
        design_indices=[2, 0],
    )
    assert [item["design_index"] for item in result["designs"]] == [0, 2]
