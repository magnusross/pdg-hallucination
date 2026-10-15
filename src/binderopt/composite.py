"""Calculate optional target-cohort composite scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from binderopt.outputs import atomic_json
from binderopt.ranking.composite import compute_composite_ranking


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--ipsae-weight", type=float, required=True)
    parser.add_argument("--dockq-weight", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(arguments)
    output = args.output or args.input / "analysis" / "composite_ranking.json"
    result = compute_composite_ranking(
        args.input,
        ipsae_weight=args.ipsae_weight,
        dockq_weight=args.dockq_weight,
    )
    atomic_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
