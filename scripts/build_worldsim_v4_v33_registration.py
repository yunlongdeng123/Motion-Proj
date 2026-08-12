#!/usr/bin/env python3
"""从完成的 V3.3 scene chain 生成 baseline matrix 注册记录。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.v33_replay import V33ReplayError, sha256_file


CORE_FILES = ("scene_chain.json", "render_manifest.json", "metrics.json")


def registration_record(run_dir: Path, *, expected_scene: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    chain_path = run_dir / "scene_chain.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    for path in (chain_path, summary_path, manifest_path, status_path):
        if not path.is_file():
            raise V33ReplayError(f"scene chain terminal file 缺失: {path}")
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if chain.get("schema_version") != "worldsim_v4_v33_scene_chain_v1":
        raise V33ReplayError("scene chain schema 漂移")
    if any(payload.get("scene") != expected_scene for payload in (chain, summary, status)):
        raise V33ReplayError("scene chain terminal scene 漂移")
    if summary.get("status") != "done" or status.get("status") != "done":
        raise V33ReplayError("scene chain 尚未完成")
    if status.get("summary_sha256") != sha256_file(summary_path):
        raise V33ReplayError("scene chain summary SHA 漂移")
    if status.get("manifest_sha256") != sha256_file(manifest_path):
        raise V33ReplayError("scene chain manifest SHA 漂移")
    if chain.get("test_quality_read") is not False or status.get(
        "test_quality_read"
    ) is not False:
        raise V33ReplayError("scene chain 未证明 test quality 未读")
    files = {}
    for name in CORE_FILES:
        path = run_dir / name
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "run": str(run_dir),
        "algorithm_commit": chain["algorithm_commit"],
        "base_checkpoint_sha256": chain["base_checkpoint_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "status_sha256": sha256_file(status_path),
        "files": files,
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = registration_record(args.run_dir, expected_scene=args.scene)
    atomic_json(args.output, record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
