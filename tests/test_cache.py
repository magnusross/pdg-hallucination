from __future__ import annotations

import os

from binderopt.assets.cache import check_cache


def test_preflight_delegates_checkpoint_to_huggingface(tmp_path, monkeypatch) -> None:
    calls = []

    def missing(repo, **kwargs):
        calls.append((repo, kwargs))
        raise FileNotFoundError("not staged")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    errors = check_cache(
        tmp_path,
        [{"checkpoint": "org/model"}],
    )
    assert calls == [
        (
            "org/model",
            {
                "cache_dir": tmp_path / "huggingface",
                "local_files_only": True,
            },
        )
    ]
    assert len(errors) == 1


def test_process_caches_point_mosaic_and_hugging_face_at_one_root(tmp_path, monkeypatch) -> None:
    from binderopt.assets.cache import configure_process_caches

    # monkeypatch snapshots these now and restores them at teardown, so the direct
    # os.environ writes inside configure_process_caches do not leak into other tests.
    monkeypatch.setenv("MOSAIC_CACHE_DIR", "")
    monkeypatch.setenv("HF_HUB_CACHE", "")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    root = tmp_path / "mosaic"
    configure_process_caches(root, offline=True)
    assert os.environ["MOSAIC_CACHE_DIR"] == str(root)
    assert os.environ["HF_HUB_CACHE"] == str(root / "huggingface")
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
