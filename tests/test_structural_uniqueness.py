"""Top-decile selection and binder extraction for Foldseek analysis."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_selects_after_pooling_and_extracts_only_binder_chain(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from analysis import structural_uniqueness as analysis

    select_top_designs = analysis.select_top_designs
    write_binder_chain = analysis.write_binder_chain

    rows = [
        {
            "target": "target",
            "optimizer": optimizer,
            "design_key": (seed, index),
            "ranking_score": float(score),
            "base_seed": seed,
            "design_index": index,
            "sequence": "A",
        }
        for optimizer in ("biohub", "pgd")
        for seed, scores in ((1, range(6)), (2, range(6, 11)))
        for index, score in enumerate(scores)
    ]
    pools = select_top_designs(rows, top_fraction=0.10)
    assert set(pools) == {("target", "biohub"), ("target", "pgd")}
    for selected in pools.values():
        assert [row["ranking_score"] for row in selected] == [10.0, 9.0]
        assert {row["base_seed"] for row in selected} == {2}

    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "SEQRES   1 A    1  ALA\n"
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"
        "TER       2      ALA A   1\n"
        "ATOM      3  CA  GLY B   1       4.000   5.000   6.000  1.00 90.00           C\n"
        "END\n"
    )
    binder = tmp_path / "binder.pdb"
    write_binder_chain(complex_pdb, binder, "A")
    text = binder.read_text()
    assert " A   1" in text
    assert " B   1" not in text

    created = []

    def fake_foldseek(command, *, text, capture_output):
        binders = Path(command[2])
        created.extend(path.stem for path in binders.iterdir())
        prefix = Path(command[3])
        (prefix.parent / f"{prefix.name}_cluster.tsv").write_text(
            "".join(f"{name}\t{name}\n" for name in created)
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(analysis, "_structure_for", lambda row, cache: complex_pdb)
    monkeypatch.setattr(analysis.subprocess, "run", fake_foldseek)
    assert (
        analysis.cluster_pool(
            pools["target", "biohub"],
            foldseek=Path("foldseek"),
            tm_threshold=0.6,
            cell_cache={},
        )
        == 2
    )
    assert created == ["target_biohub_seed02_design003", "target_biohub_seed02_design004"]
