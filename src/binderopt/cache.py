"""Check staged design and ranking model assets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from binderopt.assets.cache import (
    check_cache,
    check_ranking_cache,
    compose_profile,
    compose_ranking_recipe,
)


def main(arguments: list[str] | None = None) -> None:
    args = sys.argv[1:] if arguments is None else arguments
    if not args or args[0] != "check":
        raise SystemExit(
            "usage: python -m binderopt.cache check profile=<name> "
            "[ranking=<recipe>] [cache_root=<path>]"
        )
    values = dict(item.split("=", 1) for item in args[1:] if "=" in item)
    profile_name = values.get("profile", "reference")
    ranking_name = values.get("ranking")
    cache_root = Path(
        values.get("cache_root", os.environ.get("MOSAIC_CACHE_DIR", "~/.cache/mosaic"))
    ).expanduser()
    profile = compose_profile(profile_name)
    design = profile["design"]
    specs = [design["language_model"], *design["structure_models"]]
    errors = check_cache(cache_root, specs)
    checked = f"profile={profile_name}"
    if ranking_name is not None:
        errors.extend(check_ranking_cache(cache_root, compose_ranking_recipe(ranking_name)))
        checked += f" ranking={ranking_name}"
    if errors:
        raise SystemExit("cache preflight failed:\n- " + "\n- ".join(errors))
    print(f"cache preflight passed: {checked} root={cache_root}")


if __name__ == "__main__":
    main()
