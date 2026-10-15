"""Ranking assets must fail preflight rather than download on an offline node."""

from __future__ import annotations

from pathlib import Path

from binderopt.assets.cache import check_ranking_cache


def test_missing_protenix_assets_are_reported_without_downloading(tmp_path: Path) -> None:
    recipe = {
        "name": "full",
        "models": [
            {
                "name": "protenix_v2",
                "checkpoint": "protenix_v2",
                "factory": "mosaic.models.protenix.ProtenixV2",
            }
        ],
    }
    errors = check_ranking_cache(tmp_path, recipe)
    assert errors
    assert all("missing Protenix asset" in error for error in errors)
    assert any("protenix-v2.eqx" in error for error in errors)
    assert any("components.v20240608.cif" in error for error in errors)


def test_staged_protenix_assets_pass(tmp_path: Path) -> None:
    from binderopt.structure.protenix import protenix_asset_paths

    for path in protenix_asset_paths(tmp_path).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("staged", encoding="utf-8")
    recipe = {"name": "full", "models": [{"name": "protenix_v2", "checkpoint": "protenix_v2"}]}
    assert check_ranking_cache(tmp_path, recipe) == []
