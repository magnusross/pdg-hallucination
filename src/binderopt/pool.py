"""Run pooled redesign, screening, selection, and survivor ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from binderopt.ranking.screening import (
    audit_pool,
    publish_json,
    rank_survivors,
    redesign_cell,
    screen_cell,
    select_pool,
    verify_ranked_pool,
)


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="pool",
)
def main(cfg: DictConfig) -> None:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    resolved = cast(dict[str, Any], container)
    cells = audit_pool(resolved)
    stage = resolved["stage"]
    path = Path(str(resolved["pool_dir"])) / "selection.json"
    if stage == "audit":
        print(f"{len(cells)} complete cells; {sum(len(c['designs']) for c in cells)} candidates")
    elif stage == "mpnn":
        redesign_cell(resolved, cells)
    elif stage == "screen":
        screen_cell(resolved, cells)
    elif stage in ("select", "rank", "verify"):
        manifest = select_pool(resolved, cells)
        if stage == "select":
            if path.exists():
                raise FileExistsError(f"completed pooled selection already exists: {path}")
            publish_json(path, manifest)
        else:
            if json.loads(path.read_text()) != manifest:
                raise ValueError("frozen selection differs from current scores or provenance")
            if stage == "rank":
                rank_survivors(resolved, cells, manifest)
            else:
                verify_ranked_pool(resolved, manifest)
    else:
        raise ValueError("stage must be audit, mpnn, screen, select, rank or verify")


if __name__ == "__main__":
    main()
