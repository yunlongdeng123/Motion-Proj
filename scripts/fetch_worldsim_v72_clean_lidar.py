"""从官方 nuScenes 十个 blob 分卷中只提取冻结 clean split 的 keyframe LiDAR。"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ijson

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.data.splits import load_data_roles, require_role_access
from scripts.build_adgs_nuscenes_assets import scan_shards


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selected_scene_and_sample_tokens(
    metadata_root: Path, log_ids: set[str]
) -> tuple[list[str], set[str]]:
    scenes = json.loads((metadata_root / "scene.json").read_text(encoding="utf-8"))
    selected_scene_tokens = {
        str(row["token"]): str(row["name"])
        for row in scenes
        if str(row["log_token"]) in log_ids
    }
    samples = json.loads((metadata_root / "sample.json").read_text(encoding="utf-8"))
    sample_tokens = {
        str(row["token"])
        for row in samples
        if str(row["scene_token"]) in selected_scene_tokens
    }
    return list(selected_scene_tokens.values()), sample_tokens


def _required_keyframes(metadata_root: Path, sample_tokens: set[str]) -> set[str]:
    required: set[str] = set()
    with (metadata_root / "sample_data.json").open("rb") as handle:
        for row in ijson.items(handle, "item"):
            if (
                str(row["sample_token"]) in sample_tokens
                and bool(row["is_key_frame"])
                and str(row["filename"]).startswith("samples/LIDAR_TOP/")
            ):
                required.add(str(row["filename"]))
    return required


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    roles = load_data_roles(args.roles)
    dev_logs = require_role_access(roles, "nuscenes", "dev")
    route_logs = require_role_access(roles, "nuscenes", "route_select")
    selected_logs = set(dev_logs + route_logs)
    source_candidates = set(roles["datasets"]["nuscenes"]["group_roles"]["source_candidate_pool"])
    if selected_logs & source_candidates:
        raise RuntimeError("I/O 提取集合意外包含未打开 source-test candidates")
    scene_names, sample_tokens = _selected_scene_and_sample_tokens(
        args.metadata_root, selected_logs
    )
    required = _required_keyframes(args.metadata_root, sample_tokens)
    if not required:
        raise RuntimeError("clean split 没有解析出 LiDAR keyframes")

    run_dir = Path("/root/autodl-tmp/runs/worldsim_v72/WS-V72-P2-CLEAN-LIDAR-IO-01") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    manifest = {
        "schema_version": "worldsim_v72.clean_lidar_io.v1",
        "task_id": "WS-V72-P2-CLEAN-LIDAR-IO-01",
        "run_id": args.run_id,
        "status": "running",
        "roles": ["dev", "route_select"],
        "dev_log_count": len(dev_logs),
        "route_select_log_count": len(route_logs),
        "scene_count": len(scene_names),
        "required_keyframe_count": len(required),
        "quality_read": False,
        "source_test_read": False,
        "external_test_read": False,
        "git_commit": git_commit,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "archive_scan"})
    try:
        mapping, extracted = scan_shards(
            tar_dir=args.archive_root,
            members=required,
            index_path=args.index,
            dst=args.raw_root,
            workers=args.workers,
        )
        missing = [name for name in required if not (args.raw_root / name).is_file()]
        if missing:
            raise RuntimeError(f"提取完成后仍缺 {len(missing)} 个文件")
        total_bytes = sum((args.raw_root / name).stat().st_size for name in required)
        summary = {
            **manifest,
            "status": "done",
            "present_keyframe_count": len(required),
            "newly_extracted_keyframe_count": len(extracted),
            "total_bytes": total_bytes,
            "shards_used": sorted(set(mapping.values())),
            "index": str(args.index),
            "resources": {
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {**manifest, "status": "done", "summary": "summary.json"})
        _write_json(run_dir / "status.json", {"status": "done", "phase": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        _write_json(run_dir / "status.json", {"status": "failed", "phase": "archive_scan", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
