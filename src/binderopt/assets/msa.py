"""Persistent sequence-keyed MSA lookup for ESMFold2Full and ProtenixV2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from binderopt.outputs import atomic_json, atomic_text

PROTENIX_MSA_FORMAT = "protenix-a3m-v1"


def msa_root() -> Path:
    return Path(os.environ.get("MOSAIC_CACHE_DIR", "~/.cache/mosaic")).expanduser() / "msa"


def esm_a3m_path(sequence: str, root: Path | None = None) -> Path:
    digest = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    return (msa_root() if root is None else root) / f"{digest}.a3m"


def require_esm_msa(sequence: str, root: Path | None = None) -> Path:
    path = esm_a3m_path(sequence, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing cached MSA for sequence {sequence[:12]}… ({len(sequence)} aa): {path}; "
            "run `python -m binderopt.msa prefetch target=<target>` where network is permitted"
        )
    return path


def protenix_msa_dir(sequence: str, root: Path | None = None) -> Path:
    """Adapt a validated Mosaic A3M to Protenix's persistent two-file layout."""
    source = require_esm_msa(sequence, root)
    digest = hashlib.sha256(
        f"{PROTENIX_MSA_FORMAT}\0{sequence}".encode()
    ).hexdigest()
    entry = (msa_root() if root is None else root) / "protenix" / digest
    pairing = entry / "pairing.a3m"
    non_pairing = entry / "non_pairing.a3m"
    manifest = entry / "manifest.json"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = {
        "format": PROTENIX_MSA_FORMAT,
        "sequence": sequence,
        "source": source.name,
        "source_sha256": source_sha,
    }
    if (
        manifest.is_file()
        and pairing.is_file()
        and non_pairing.is_file()
        and json.loads(manifest.read_text(encoding="utf-8")) == expected
    ):
        return entry

    atomic_text(pairing, f">query\n{sequence}\n")
    atomic_text(non_pairing, source.read_text(encoding="utf-8"))
    atomic_json(manifest, expected)
    return entry


def prefetch(sequence: str) -> Path:
    """Populate Mosaic's native A3M cache, then validate the resulting entry."""
    from mosaic.models.esmfold2 import _fetch_msa

    _fetch_msa(sequence)
    return require_esm_msa(sequence)


def experiment_target_ids(experiment: str) -> list[str]:
    """Read the target sweep of one experiment, so preflight matches what will run."""
    import yaml

    config = yaml.safe_load(
        (_config_root() / "experiment" / f"{experiment}.yaml").read_text(encoding="utf-8")
    )
    targets = config.get("hydra", {}).get("sweeper", {}).get("params", {}).get("target")
    if not targets:
        raise SystemExit(f"experiment {experiment} does not sweep over targets")
    return [item.strip() for item in str(targets).split(",") if item.strip()]




def _config_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"
