"""Small read-only cache preflight delegated to Hugging Face."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


def configure_process_caches(cache_root: Path, *, offline: bool) -> None:
    """Point Mosaic and Hugging Face at one cache root for this process.

    Set `JAX_COMPILATION_CACHE_DIR` before importing JAX; later changes are ignored.
    """
    os.environ["MOSAIC_CACHE_DIR"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def check_cache(
    cache_root: Path,
    model_specs: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return cache misses without downloading or creating cache entries."""
    from huggingface_hub import snapshot_download

    errors: list[str] = []
    for spec in model_specs:
        checkpoint = spec.get("checkpoint")
        if not checkpoint:
            errors.append(f"model checkpoint is missing: {dict(spec)}")
            continue
        try:
            snapshot_download(
                str(checkpoint),
                cache_dir=cache_root / "huggingface",
                local_files_only=True,
            )
        except Exception as error:
            errors.append(f"missing Hugging Face checkpoint {checkpoint}: {error}")

    return errors


def check_ranking_cache(cache_root: Path, recipe: Mapping[str, Any]) -> list[str]:
    """Return ranking-recipe cache misses, including Protenix's non-Hub assets.

    Design preflight covers only the design ensemble, so without this a long
    design+rank job stages nothing for ranking and fails after the design stage.
    """
    from binderopt.structure.protenix import protenix_asset_paths

    specs = list(recipe["models"])
    hub_specs = [spec for spec in specs if spec["name"] != "protenix_v2"]
    errors = check_cache(cache_root, hub_specs)
    if any(spec["name"] == "protenix_v2" for spec in specs):
        errors.extend(
            f"missing Protenix asset {path}"
            for path in protenix_asset_paths(cache_root).values()
            if not path.is_file()
        )
    return errors




def compose_ranking_recipe(recipe_name: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base="1.3", config_dir=str(_config_root())):
        rank_config = compose("rank", overrides=[f"ranking={recipe_name}"])
        recipe = OmegaConf.to_container(rank_config.ranking, resolve=True)
    return cast(dict[str, Any], recipe)


def compose_profile(profile_name: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base="1.3", config_dir=str(_config_root())):
        design_config = compose("design", overrides=[f"profile={profile_name}"])
        profile = OmegaConf.to_container(design_config.profile, resolve=False)
        assert isinstance(profile, dict)
    return cast(dict[str, Any], profile)


def _config_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"
