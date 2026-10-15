"""One isolated ranking-model worker and its final output."""

from __future__ import annotations

import dataclasses
import faulthandler
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax

    from binderopt.structure.features import EntityMasks, PreparedComplex

    FeatureBuilder = Callable[[str], PreparedComplex]


@dataclasses.dataclass(frozen=True, slots=True)
class Prediction:
    ipsae_min: float
    structure: str
    self_consistency: Mapping[str, Any] | None = None


PredictSeed = Callable[[int], Prediction]
MakePredictor = Callable[[Mapping[str, Any]], PredictSeed]


def protein_ipsae_min(
    sequence: jax.Array, output: Any, masks: EntityMasks, *, key: jax.Array
) -> float:
    """Calculate Mosaic IPSAE_min after removing ligand tokens from the score."""
    import numpy as np
    from mosaic.losses.structure_prediction import IPSAE_min

    indices = np.flatnonzero(masks.protein)
    protein_output = dataclasses.replace(
        output,
        full_sequence=output.full_sequence[indices],
        pae_logits=output.pae_logits[indices[:, None], indices[None, :]],
    )
    _, auxiliary = IPSAE_min()(sequence, protein_output, key=key)
    return float(auxiliary["ipsae_min"])


def run_model(
    cell_dir: Path,
    *,
    model_name: str,
    designs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    provenance: Mapping[str, Any],
    make_predictor: MakePredictor,
    worker_started: float | None = None,
    worker_started_at_utc: str | None = None,
    model_load_wall_time_s: float = 0.0,
    gpu_count: int = 1,
) -> dict[str, Any]:
    """Prepare each design once, run its seeds, and write one complete model result."""
    import numpy as np

    from binderopt.outputs import atomic_json, atomic_text, utc_now

    requested = [int(seed) for seed in seeds]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("ranking seeds must be non-empty and unique")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if worker_started is None:
        worker_started = time.perf_counter()
    if worker_started_at_utc is None:
        worker_started_at_utc = utc_now()

    records: list[dict[str, Any]] = []
    feature_prep_wall_time_s = 0.0
    prediction_wall_time_s = 0.0
    for design in designs:
        seed_scores: list[dict[str, Any]] = []
        predictions: dict[int, Prediction] = {}
        feature_started = time.perf_counter()
        predict = make_predictor(design)
        feature_elapsed = time.perf_counter() - feature_started
        feature_prep_wall_time_s += feature_elapsed
        for seed in requested:
            print(
                "ranking_prediction_start "
                f"model={model_name} cell={cell_dir.name} "
                f"design_index={int(design['design_index'])} seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            prediction_started = time.perf_counter()
            prediction = predict(seed)
            score = float(prediction.ipsae_min)
            prediction_elapsed = time.perf_counter() - prediction_started
            prediction_wall_time_s += prediction_elapsed
            if not np.isfinite(score):
                raise ValueError(f"{model_name} seed {seed} returned a non-finite IPSAE_min")
            if prediction.self_consistency is None:
                raise ValueError(f"{model_name} seed {seed} has no self-consistency metrics")
            dockq = float(prediction.self_consistency["dockq"])
            if not np.isfinite(dockq):
                raise ValueError(f"{model_name} seed {seed} returned non-finite self-consistency")
            predictions[seed] = prediction
            seed_scores.append(
                {
                    "seed": seed,
                    "ipsae_min": score,
                    **dict(prediction.self_consistency),
                    "prediction_wall_time_s": round(prediction_elapsed, 3),
                }
            )

        winning_seed = min(
            requested,
            key=lambda seed: (-predictions[seed].ipsae_min, seed),
        )
        winning_prediction = predictions[winning_seed]
        winning_self_consistency = winning_prediction.self_consistency
        assert winning_self_consistency is not None
        records.append(
            {
                "design_index": int(design["design_index"]),
                "sequence": str(design["sequence"]),
                "seed_scores": seed_scores,
                "feature_prep_wall_time_s": round(feature_elapsed, 3),
                "best_ipsae_min": winning_prediction.ipsae_min,
                "winning_dockq": winning_self_consistency["dockq"],
                "winning_seed": winning_seed,
                "_structure_text": predictions[winning_seed].structure,
            }
        )

    structure_write_started = time.perf_counter()
    for record in records:
        structure_path = (
            cell_dir / "structures" / f"design_{record['design_index']:03d}" / f"{model_name}.pdb"
        )
        atomic_text(structure_path, record.pop("_structure_text"))
        record["structure"] = str(structure_path.relative_to(cell_dir))
    structure_write_wall_time_s = time.perf_counter() - structure_write_started

    completed = time.perf_counter()
    result = {
        "model_name": model_name,
        "requested_seeds": requested,
        "provenance": dict(provenance),
        "ranking_model_timing": {
            "started_at_utc": worker_started_at_utc,
            "completed_at_utc": utc_now(),
            "worker_wall_time_s": round(completed - worker_started, 3),
            "model_load_wall_time_s": round(model_load_wall_time_s, 3),
            "feature_prep_wall_time_s": round(feature_prep_wall_time_s, 3),
            "prediction_wall_time_s": round(prediction_wall_time_s, 3),
            "structure_write_wall_time_s": round(structure_write_wall_time_s, 3),
            "gpu_count": gpu_count,
        },
        "designs": records,
    }
    atomic_json(cell_dir / "ranking" / f"{model_name}.json", result)
    return result


def _run_structural_model(
    cell_dir: Path,
    *,
    model_spec: Mapping[str, Any],
    model: Any,
    build_features: FeatureBuilder,
    seeds: Sequence[int],
    provenance: Mapping[str, Any],
    worker_started: float,
    worker_started_at_utc: str,
    model_load_wall_time_s: float,
    gpu_count: int,
    self_consistency_spec: Mapping[str, Any],
    target_sequences: Sequence[str],
    target_residue_ranges: Sequence[Sequence[Sequence[int]]] | None,
) -> None:
    """Run one structure model while retaining only the current design's features."""
    import jax
    import jax.numpy as jnp
    from mosaic.common import TOKENS

    mpnn_result = json.loads(
        (cell_dir / "ranking" / "soluble_mpnn.json").read_text(encoding="utf-8")
    )
    token_index = {letter: index for index, letter in enumerate(TOKENS)}

    def make_predictor(design: Mapping[str, Any]) -> PredictSeed:
        sequence = str(design["sequence"])
        prepared = build_features(sequence)
        reference_pdb = cell_dir / str(design["source_structure"])
        one_hot = jax.nn.one_hot(
            jnp.asarray([token_index[letter] for letter in sequence]), len(TOKENS)
        )

        def predict(seed: int) -> Prediction:
            from binderopt.ranking.self_consistency import score_self_consistency

            key = jax.random.key(seed)
            prediction = model.predict(
                PSSM=one_hot,
                features=prepared.features,
                writer=prepared.writer,
                recycling_steps=int(model_spec["recycling_steps"]),
                sampling_steps=int(model_spec["sampling_steps"]),
                key=key,
            )
            score = protein_ipsae_min(one_hot, prediction.model_output, prepared.masks, key=key)
            structure = prediction.st.make_pdb_string()
            rmsd = self_consistency_spec["target_aligned_rmsd"]
            self_consistency = score_self_consistency(
                reference_pdb,
                structure,
                binder_length=len(sequence),
                target_sequences=target_sequences,
                atom_names=list(rmsd["atom_names"]),
                target_residue_ranges=target_residue_ranges,
            )
            return Prediction(score, structure, self_consistency)

        return predict

    run_model(
        cell_dir,
        model_name=str(model_spec["name"]),
        designs=mpnn_result["designs"],
        seeds=seeds,
        provenance=provenance,
        make_predictor=make_predictor,
        worker_started=worker_started,
        worker_started_at_utc=worker_started_at_utc,
        model_load_wall_time_s=model_load_wall_time_s,
        gpu_count=gpu_count,
    )


def run_ranking_model_worker(
    cell_dir: Path,
    *,
    recipe_name: str,
    model_spec: Mapping[str, Any],
    seeds: Sequence[int],
    base_seed: int,
    cache_root: Path,
    offline: bool,
    self_consistency_spec: Mapping[str, Any],
    gpu_count: int = 1,
) -> None:
    """Load one judge model once, then rank one complete design cell."""
    import os

    faulthandler.enable(all_threads=True)

    from hydra.utils import get_method

    from binderopt.assets.cache import configure_process_caches
    from binderopt.outputs import utc_now
    from binderopt.structure.features import build_esm_ranking_features
    from binderopt.structure.protenix import (
        build_protenix_ranking_features,
        configure_protenix_assets,
    )
    from binderopt.structure.targets import target_from_config

    worker_started = time.perf_counter()
    worker_started_at_utc = utc_now()
    configure_process_caches(cache_root, offline=offline)
    if model_spec["name"] == "protenix_v2":
        configured_root = Path(os.environ.get("MOSAIC_CACHE_DIR", cache_root)).expanduser()
        configure_protenix_assets(configured_root)
    model_load_started = time.perf_counter()
    model = get_method(str(model_spec["factory"]))()
    model_load_wall_time_s = time.perf_counter() - model_load_started

    provenance = {
        "recipe": recipe_name,
        "model": dict(model_spec),
        "ranking_seed": base_seed,
        "self_consistency": dict(self_consistency_spec),
    }
    result = json.loads((cell_dir / "result.json").read_text(encoding="utf-8"))
    target_config = result.get("config", {}).get("target")
    if not isinstance(target_config, Mapping):
        raise ValueError(f"design result has no resolved target config: {cell_dir}")
    target = target_from_config(target_config)
    if model_spec["name"] == "protenix_v2":
        def build_features(sequence: str):
            return build_protenix_ranking_features(sequence, target)
    else:
        def build_features(sequence: str):
            return build_esm_ranking_features(
                model,
                sequence,
                target,
                supports_msa=bool(model_spec["use_target_msa"]),
            )
    _run_structural_model(
        cell_dir,
        model_spec=model_spec,
        model=model,
        build_features=build_features,
        seeds=seeds,
        provenance=provenance,
        worker_started=worker_started,
        worker_started_at_utc=worker_started_at_utc,
        model_load_wall_time_s=model_load_wall_time_s,
        gpu_count=gpu_count,
        self_consistency_spec=self_consistency_spec,
        target_sequences=[chain.sequence for chain in target.chains],
        target_residue_ranges=target.self_consistency_residue_ranges,
    )
