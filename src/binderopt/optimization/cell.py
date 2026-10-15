"""Batch optimization and final design-model reference predictions."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GeneratedDesign:
    """One finished design, returned in batch-lane order. The caller names it."""

    sequence: str
    pssm: NDArray[np.float32]
    selected_step: int
    tail_ranking_loss: float
    design_objective: float
    argmax_loss: float
    esmc_log_perplexity: float
    design_time_s: float
    metrics: dict[str, Any] = field(default_factory=dict)
    trajectory: tuple[dict[str, Any], ...] = ()


BatchRunner = Callable[[tuple[int, ...], Any], Sequence[GeneratedDesign]]


def run_design_cell(
    *,
    keys: Any,
    batch_size: int,
    run_batch: BatchRunner,
) -> tuple[tuple[int, int, int, GeneratedDesign], ...]:
    """Run every batch and return designs in stable generation order."""
    num_designs = len(keys)
    if num_designs <= 0 or batch_size <= 0 or num_designs % batch_size:
        raise ValueError(
            "num_designs and batch_size must be positive and batch_size must divide num_designs"
        )

    completed: list[tuple[int, int, int, GeneratedDesign]] = []
    for batch, start in enumerate(range(0, num_designs, batch_size)):
        indices = tuple(range(start, start + batch_size))
        generated = tuple(run_batch(indices, keys[start : start + batch_size]))
        if len(generated) != len(indices):
            raise ValueError(f"batch {batch} returned {len(generated)} designs for lanes {indices}")

        completed.extend((indices[lane], batch, lane, item) for lane, item in enumerate(generated))
    return tuple(completed)


def make_batch_runner(
    config: Mapping[str, Any], ensemble: Any, complex_features: Any
) -> BatchRunner:
    import jax

    from binderopt.optimization.models import build_design_loss
    from binderopt.optimization.objective import (
        IPTMMonitor,
        build_structure_objective,
        harden_and_evaluate,
        hardened_esmc_log_perplexity,
        protein_only,
        validate_contact_distance,
    )
    from binderopt.optimization.optimizers import (
        initialize_logits,
        optimize_biohub,
        optimize_pgd,
    )
    from binderopt.optimization.seeding import lane_keys

    design_config = config["design"]
    objective_config = design_config["objective"]
    fixed_structure = protein_only(
        build_structure_objective(objective_config), complex_features.masks
    )
    validate_contact_distance(
        ensemble.features_model.distogram_bins,
        float(objective_config["inter_contact"]["contact_distance"]),
    )
    design_loss = build_design_loss(
        ensemble,
        fixed_structure,
        complex_features.features,
        config["profile"]["design"]["inference"],
        objective_config,
    )
    # Keep confidence and diffusion evaluation confined to the tail window.
    tail_loss = build_design_loss(
        ensemble,
        fixed_structure + IPTMMonitor(complex_features.masks),
        complex_features.features,
        config["profile"]["design"]["inference"],
        objective_config,
    )

    alphabet = design_config["alphabet"]
    excluded = set(design_config["excluded_amino_acids"])
    token_mask = np.asarray([letter not in excluded for letter in alphabet], dtype=bool)
    mask = np.broadcast_to(token_mask, (int(config["binder_length"]), len(alphabet)))
    optimizer = config["optimizer"]

    def run(indices: tuple[int, ...], keys: Any) -> tuple[GeneratedDesign, ...]:
        started = time.perf_counter()
        init_keys, optimizer_keys, hard_keys = lane_keys(keys)
        logits = initialize_logits(
            init_keys,
            binder_length=int(config["binder_length"]),
            alphabet_size=len(alphabet),
            init_scale=float(optimizer["init_scale"]),
            mask=mask,
        )
        common = {
            "loss": design_loss,
            "tail_loss": tail_loss,
            "n_steps": int(design_config["n_steps"]),
            "tail_select_steps": int(design_config["tail_select_steps"]),
            "mask": mask,
            "keys": optimizer_keys,
        }
        if optimizer["kind"] == "biohub":
            result = optimize_biohub(
                logits=logits,
                learning_rate=float(optimizer["learning_rate"]),
                temperature_min=float(optimizer["temperature_min"]),
                beta=float(optimizer["beta"]),
                **common,
            )
        elif optimizer["kind"] == "pgd":
            result = optimize_pgd(
                initial_probabilities=jax.nn.softmax(logits, axis=-1),
                binder_length=int(config["binder_length"]),
                stepsize_scale=float(optimizer["stepsize_scale"]),
                momentum=float(optimizer["momentum"]),
                max_gradient_norm=float(optimizer["max_gradient_norm"]),
                **common,
            )
        else:
            raise ValueError("optimizer.kind must be biohub or pgd")

        one_hot, hardened = harden_and_evaluate(
            design_loss, result.selections, alphabet_size=len(alphabet), keys=hard_keys
        )
        one_hot.block_until_ready()
        # Scored on the written sequence, not the PSSM
        log_perplexity = hardened_esmc_log_perplexity(
            design_loss,
            one_hot,
            keys=hard_keys,
            batches=int(design_config["final_esmc_perplexity_batches"]),
        )
        elapsed = time.perf_counter() - started
        points: dict[int, list[dict[str, Any]]] = {lane: [] for lane in range(len(indices))}
        for point in result.trajectory:
            points[point.lane].append(asdict(point) | {"design_index": indices[point.lane]})
        generated: list[GeneratedDesign] = []
        for lane, selection in enumerate(result.selections):
            sequence = "".join(alphabet[index] for index in np.asarray(one_hot[lane]).argmax(-1))
            generated.append(
                GeneratedDesign(
                    sequence=sequence,
                    pssm=selection.pssm,
                    selected_step=selection.step,
                    tail_ranking_loss=selection.tail_ranking_loss,
                    design_objective=selection.design_objective,
                    argmax_loss=float(hardened[lane]),
                    esmc_log_perplexity=float(log_perplexity[lane]),
                    design_time_s=elapsed / len(indices),
                    trajectory=tuple(points[lane]),
                )
            )
        return tuple(generated)

    return run


def predict_design_references(
    *,
    ensemble: Any,
    prepared: Any,
    pssms: Sequence[NDArray[np.float32]],
    design_keys_array: Any,
    model_specs: Sequence[Mapping[str, Any]],
    inference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Fold every hardened optimizer PSSM through both ordered design models."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from binderopt.optimization.objective import IPTMMonitor

    specs = list(model_specs)
    if len(specs) != len(ensemble.replicates):
        raise ValueError("one named reference spec is required for every design model")
    names = [str(spec["name"]) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("design model names must be unique")
    monitor = IPTMMonitor(prepared.masks)
    settings = {
        "recycling_steps": int(inference["recycling_steps"]),
        "sampling_steps": int(inference["sampling_steps"]),
        "msa_max_depth": int(inference["msa_max_depth"]),
    }
    models = [
        eqx.tree_at(
            lambda item: item.esmc,
            replicate,
            ensemble.esmc,
            is_leaf=lambda value: value is None,
        )
        for replicate in ensemble.replicates
    ]
    references: list[dict[str, Any]] = []
    for design_index, pssm in enumerate(pssms):
        hardened = jax.nn.one_hot(jnp.asarray(pssm).argmax(-1), 20)
        records: list[dict[str, Any]] = []
        for model_index, (model, spec) in enumerate(zip(models, specs, strict=True)):
            key = jax.random.fold_in(design_keys_array[design_index], model_index)
            prediction = model.predict(
                PSSM=hardened,
                features=prepared.features,
                writer=prepared.writer,
                recycling_steps=int(settings["recycling_steps"]),
                sampling_steps=int(settings["sampling_steps"]),
                msa_max_depth=int(settings["msa_max_depth"]),
                key=key,
            )
            _, metrics = monitor(hardened, prediction.model_output, key=key)
            records.append(
                {
                    "model_name": str(spec["name"]),
                    "checkpoint": str(spec["checkpoint"]),
                    "protein_iptm": float(np.asarray(metrics["iptm"])),
                    "settings": dict(settings),
                    "_structure_text": prediction.st.make_pdb_string(),
                }
            )
        selected = max(range(len(records)), key=lambda index: records[index]["protein_iptm"])
        references.append({"models": records, "selected_model": records[selected]["model_name"]})
    return references
