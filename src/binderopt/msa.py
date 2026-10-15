"""Prepare and validate target-chain MSAs."""

from __future__ import annotations

import sys
from pathlib import Path

from binderopt.assets.msa import (
    experiment_target_ids,
    prefetch,
    protenix_msa_dir,
    require_esm_msa,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def main(arguments: list[str] | None = None) -> None:
    args = sys.argv[1:] if arguments is None else arguments
    if not args or args[0] not in {"check", "prefetch"}:
        raise SystemExit(
            "usage: python -m binderopt.msa {check,prefetch} "
            "{target=A,B|target=all|experiment=<name>}"
        )
    values = dict(item.split("=", 1) for item in args[1:] if "=" in item)
    experiment = values.get("experiment")
    target_value = values.get("target")
    if experiment is not None:
        target_ids = experiment_target_ids(experiment)
    elif target_value is None or target_value == "all":
        target_ids = sorted(path.stem for path in CONFIG_DIR.glob("target/*.yaml"))
    else:
        target_ids = target_value.split(",")
    import yaml

    sequences: set[str] = set()
    for target_id in target_ids:
        config = yaml.safe_load((CONFIG_DIR / "target" / f"{target_id}.yaml").read_text())
        sequences.update(chain["sequence"] for chain in config["chains"] if chain["use_msa"])
    for sequence in sorted(sequences):
        if args[0] == "prefetch":
            prefetch(sequence)
        require_esm_msa(sequence)
        protenix_msa_dir(sequence)
    print(f"{args[0]}: {len(sequences)} unique target-chain MSAs ready")


if __name__ == "__main__":
    main()
