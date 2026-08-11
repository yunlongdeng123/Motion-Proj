#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v4.datasets.kitti import (
    TASK_ID,
    KittiAdapterError,
    build_tracking_manifest,
    canonical_json_bytes,
    detect_kitti_layout,
    sha256_file,
)


SNAPSHOT_RELPATHS = (
    "motion_proj/worldsim_v4/datasets/kitti.py",
    "scripts/audit_worldsim_v4_kitti.py",
    "scripts/build_worldsim_v4_kitti_manifest.py",
    "tests/test_worldsim_v4_kitti_track_id.py",
    "configs/worldsim_v4/kitti_adapter_v1.yaml",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _git(project_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise KittiAdapterError(process.stderr.strip())
    return process.stdout.strip()


def _markdown(layout: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# KITTI Layout Audit",
        "",
        f"- task: `{TASK_ID}`",
        f"- root: `{layout['dataset_root']}`",
        f"- layout: `{layout['layout']}`",
        f"- status: `{manifest['status']}`",
        "- download attempted: `false`",
        "- quality read / training: `false / false`",
        "",
    ]
    if layout["layout"] == "missing":
        lines.extend(
            [
                "公共路径当前不存在。adapter 代码与 synthetic contract tests 已落地，但不创建空目录、不下载数据、",
                "不输出质量表；挂载真实 KITTI 后必须用新 run 重新审计。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- detected sequences: `{layout.get('sequence_count', 0)}`",
                f"- cameras: `{', '.join(layout.get('camera_contract', []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def run(config_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    project_root = project_root.resolve()
    if run_dir.exists():
        raise KittiAdapterError(f"run 目录已存在，禁止复用：{run_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("task_id") != TASK_ID:
        raise KittiAdapterError("D1 config task_id 非法")
    requested_root = Path(config["dataset"]["root"]).absolute()
    dataset_root = requested_root.resolve()
    layout = detect_kitti_layout(requested_root)
    expected_status = config.get("freeze", {}).get("expected_status")
    if expected_status and layout["status"] != expected_status:
        raise KittiAdapterError(f"KITTI root state 漂移：{layout['status']} != {expected_status}")
    if layout["status"] == "ready" and layout["layout"] == "tracking_training":
        adapter_manifest = build_tracking_manifest(
            layout,
            smoke_count=int(config["protocol"]["adapter_smoke_sequences"]),
            formal_count=int(config["protocol"]["cross_domain_target_sequences"]),
        )
    else:
        adapter_manifest = {
            "schema_version": "worldsim_v4_kitti_manifest_v1",
            "task_id": TASK_ID,
            "status": "blocked",
            "reason": layout["status"],
            "dataset_root": str(dataset_root),
            "layout": layout["layout"],
            "camera_contract": ["image_02", "image_03"],
            "method_threshold_source": "frozen_nuscenes_only",
            "kitti_threshold_search": False,
            "download_attempted": False,
            "quality_read": False,
            "training": False,
        }
    terminal_status = "done" if adapter_manifest["status"] == "done" else "blocked"
    run_dir.mkdir(parents=True)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    _write_json(artifacts / "layout_audit.json", layout)
    _write_json(artifacts / "kitti_manifest.json", adapter_manifest)
    (artifacts / "KITTI_LAYOUT_AUDIT.md").write_text(
        _markdown(layout, adapter_manifest), encoding="utf-8"
    )
    event = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "event": "layout_audit_complete",
        "status": terminal_status,
        "reason": adapter_manifest.get("reason"),
    }
    (run_dir / "events.jsonl").write_bytes(canonical_json_bytes(event))
    shutil.copy2(config_path, run_dir / "resolved.yaml")
    snapshots = {}
    for relpath in SNAPSHOT_RELPATHS:
        source = project_root / relpath
        if not source.is_file():
            raise KittiAdapterError(f"source snapshot 缺失：{source}")
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    fingerprint = {
        "requested_root": str(requested_root),
        "dataset_root": str(dataset_root),
        "root_exists": dataset_root.exists(),
        "root_is_dir": dataset_root.is_dir(),
        "layout_audit_sha256": sha256_file(artifacts / "layout_audit.json"),
        "source_snapshots": snapshots,
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    finished = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": "worldsim_v4_d1_summary_v1",
        "task_id": TASK_ID,
        "status": terminal_status,
        "reason": adapter_manifest.get("reason"),
        "finished_at_utc": finished,
        "layout": layout,
        "adapter_manifest_sha256": sha256_file(artifacts / "kitti_manifest.json"),
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "project_git": {
            "head": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "dirty": _git(project_root, "status", "--porcelain") != "",
        },
        "download_attempted": False,
        "quality_read": False,
        "training": False,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": terminal_status,
            "finished_at_utc": finished,
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    files = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v4_d1_run_manifest_v1",
            "task_id": TASK_ID,
            "status": terminal_status,
            "artifacts": files,
            "download_attempted": False,
            "quality_read": False,
            "training": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="审计本地 KITTI layout，不下载")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.run_dir, args.project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
