from __future__ import annotations

import json

import pytest

from binderopt.assets.msa import esm_a3m_path, protenix_msa_dir, require_esm_msa


def test_msa_identity_is_sequence_keyed_and_missing_is_actionable(tmp_path) -> None:
    assert esm_a3m_path("AAAA", tmp_path) != esm_a3m_path("AAAR", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"python -m binderopt\.msa prefetch"):
        require_esm_msa("AAAA", tmp_path)


def test_protenix_adapter_is_persistent_atomic_and_invalidates_source_change(tmp_path) -> None:
    sequence = "ARND"
    source = esm_a3m_path(sequence, tmp_path)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f">query\n{sequence}\n>hit\n{sequence}\n")
    entry = protenix_msa_dir(sequence, tmp_path)
    assert (entry / "pairing.a3m").read_text() == f">query\n{sequence}\n"
    assert ">hit" in (entry / "non_pairing.a3m").read_text()
    first = json.loads((entry / "manifest.json").read_text())

    source.write_text(f">query\n{sequence}\n>new\n{sequence}\n")
    assert protenix_msa_dir(sequence, tmp_path) == entry
    second = json.loads((entry / "manifest.json").read_text())
    assert first["source_sha256"] != second["source_sha256"]
    assert ">new" in (entry / "non_pairing.a3m").read_text()
