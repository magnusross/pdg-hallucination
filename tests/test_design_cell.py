from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from binderopt.optimization.cell import GeneratedDesign, predict_design_references, run_design_cell
from binderopt.optimization.seeding import design_keys, key_words


def _generated(indices: tuple[int, ...], keys):
    del keys
    alphabet = "ARND"
    return tuple(
        GeneratedDesign(
            sequence=alphabet[index] * 3,
            pssm=np.eye(20, dtype=np.float32)[[index] * 3],
            selected_step=3,
            tail_ranking_loss=-0.5 - index,
            design_objective=4.0 - index,
            argmax_loss=1.0 + index,
            esmc_log_perplexity=2.0 + index,
            design_time_s=0.1,
            trajectory=tuple(
                {"design_index": index, "step": step, "lane": lane, "objective": 2.0 - step}
                for step in range(4)
            ),
        )
        for lane, index in enumerate(indices)
    )


def test_design_cell_collects_batches_in_stable_generation_order() -> None:
    calls = []

    def runner(indices, keys):
        calls.append((indices, keys))
        return _generated(indices, keys)

    keys = design_keys(5, 4)
    completed = run_design_cell(keys=keys, batch_size=2, run_batch=runner)
    assert [index for index, _, _, _ in completed] == [0, 1, 2, 3]
    assert len({item.sequence for _, _, _, item in completed}) == 4
    assert [indices for indices, _ in calls] == [(0, 1), (2, 3)]
    # Every batch receives exactly its own designs' keys, in design-index order.
    batched = np.concatenate([key_words(item) for _, item in calls])
    np.testing.assert_array_equal(batched, key_words(keys))


def test_design_cell_does_not_checkpoint_or_resume() -> None:
    calls = []

    def runner(indices, keys):
        calls.append((indices, keys))
        return _generated(indices, keys)

    kwargs = {"keys": design_keys(5, 2), "batch_size": 2, "run_batch": runner}
    run_design_cell(**kwargs)
    run_design_cell(**kwargs)
    assert [indices for indices, _ in calls] == [(0, 1), (0, 1)]


def test_reference_selection_uses_protein_iptm_and_yaml_order_for_ties(monkeypatch) -> None:
    predictions = []

    class FakeModel:
        def __init__(self, score: float, name: str) -> None:
            self.score = score
            self.name = name
            self.esmc = None

        def predict(self, **kwargs):
            predictions.append((self.name, kwargs))
            structure = SimpleNamespace(make_pdb_string=lambda: f"MODEL {self.name}\n")
            return SimpleNamespace(st=structure, model_output=SimpleNamespace(score=self.score))

    class FakeMonitor:
        def __init__(self, masks) -> None:
            self.masks = masks

        def __call__(self, sequence, output, *, key):
            return 0.0, {"iptm": output.score}

    monkeypatch.setattr("equinox.tree_at", lambda *args, **kwargs: args[1])
    monkeypatch.setattr("binderopt.optimization.objective.IPTMMonitor", FakeMonitor)
    ensemble = SimpleNamespace(
        replicates=(FakeModel(0.8, "first"), FakeModel(0.8, "second")),
        esmc=object(),
    )
    prepared = SimpleNamespace(features=object(), writer=object(), masks=object())
    specs = [{"name": name, "checkpoint": name} for name in ("first", "second")]
    inference = {
        "recycling_steps": 1,
        "sampling_steps": 2,
        "msa_max_depth": 3,
    }
    pssm = np.zeros((3, 20), dtype=np.float32)
    pssm[:, 4] = 2.0
    keys = design_keys(4, 1)
    first = predict_design_references(
        ensemble=ensemble,
        prepared=prepared,
        pssms=[pssm],
        design_keys_array=keys,
        model_specs=specs,
        inference=inference,
    )
    predict_design_references(
        ensemble=ensemble,
        prepared=prepared,
        pssms=[pssm],
        design_keys_array=keys,
        model_specs=specs,
        inference=inference,
    )
    assert first[0]["selected_model"] == "first"
    np.testing.assert_array_equal(
        key_words(predictions[0][1]["key"]), key_words(predictions[2][1]["key"])
    )
    assert not np.array_equal(
        key_words(predictions[0][1]["key"]), key_words(predictions[1][1]["key"])
    )
    first_kwargs = predictions[0][1]
    assert np.all(np.asarray(first_kwargs["PSSM"]).sum(axis=-1) == 1)
    assert np.all(np.asarray(first_kwargs["PSSM"]).argmax(axis=-1) == 4)
    assert {name: first_kwargs[name] for name in inference} == inference
    assert first[0]["models"][0]["settings"] == inference
