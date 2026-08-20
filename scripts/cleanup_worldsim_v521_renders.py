#!/usr/bin/env python3
"""P4/P9 后精确清理非 panel prediction 与可再生 shadow staging。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from motion_proj.worldsim_v521.census import sha256_file
from motion_proj.worldsim_v521.protocol import atomic_json


RUNS_ROOT = Path("/root/autodl-tmp/runs/worldsim_v521").resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_run_path(path: str | Path) -> Path:
    value = Path(path).resolve()
    if value == RUNS_ROOT or RUNS_ROOT not in value.parents:
        raise RuntimeError(f"cleanup target 越界：{value}")
    return value


def tree_lstat(root: Path) -> dict[str, int]:
    files = directories = symlinks = logical_bytes = 0
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        directories += len(dirnames)
        for name in filenames:
            path = Path(current) / name
            stat = path.lstat()
            files += 1
            symlinks += int(path.is_symlink())
            logical_bytes += int(stat.st_size)
    return {"files": files, "directories": directories, "symlinks": symlinks, "logical_lstat_bytes": logical_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args()
    run = assert_run_path(args.run_dir)
    registry = read_jsonl(args.registry.resolve())
    selected = {
        (row["scene"], int(row["canonical_sample_index"]), int(row["camera"]))
        for row in registry
        if row.get("selected_for_panel") and row.get("entity_kind") == "view" and row.get("panel_path")
    }
    if not selected:
        raise RuntimeError("没有已生成 panel 的 selected view，禁止 cleanup")
    removed = []
    retained = []
    for base in ("adgs", "streetgs"):
        root = run / "renders" / base
        for scene in sorted(root.iterdir()):
            for row in read_jsonl(scene / "RENDER_MAP.jsonl"):
                prediction = assert_run_path(row["prediction_path"])
                observed = sha256_file(prediction)
                if observed != row["prediction_sha256"]:
                    raise RuntimeError(f"prediction cleanup 前 hash 漂移：{prediction}")
                key = (row["scene"], int(row["frame"]), int(row["camera"]))
                item = {"path": str(prediction), "bytes": prediction.stat().st_size, "sha256": observed, "base": base, "key": list(key)}
                if key in selected:
                    retained.append(item)
                else:
                    prediction.unlink()
                    removed.append(item)
            if base == "adgs":
                for gt in sorted(scene.glob("model/test/ours_60000/gt/*.png")):
                    gt = assert_run_path(gt)
                    item = {"path": str(gt), "bytes": gt.stat().st_size, "sha256": sha256_file(gt), "base": base, "kind": "redundant_renderer_gt"}
                    gt.unlink()
                    removed.append(item)
    staging = assert_run_path(run / "staging")
    staging_inventory = tree_lstat(staging)
    shutil.rmtree(staging)
    atomic_json(
        run / "CLEANUP_MANIFEST.json",
        {
            "schema": "worldsim_v521_render_cleanup_v1",
            "run": str(run), "registry": str(args.registry.resolve()),
            "selected_view_keys": len(selected), "retained_predictions": retained,
            "removed_files": removed,
            "removed_file_bytes": sum(row["bytes"] for row in removed),
            "removed_staging": {"path": str(staging), **staging_inventory},
            "canonical_metrics_panels_audits_preserved": True,
            "recoverability": "regenerable_from_frozen_checkpoint_data_split_and_renderer",
        },
    )


if __name__ == "__main__":
    main()
