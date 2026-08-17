#!/usr/bin/env python3
"""可续传下载并完整哈希冻结 V5.1 Stage B 官方 DINOv2 权重。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _write_text(
        path,
        existing + json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != "worldsim_v51_stage_b_dinov2_download_v1":
        raise ProtocolError("DINOv2 download schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("DINOv2 download task 漂移")
    if config.get("status") != "running":
        raise ProtocolError("DINOv2 download status 漂移")

    freeze_spec = config["input_freeze"]
    freeze_path = PROJECT / freeze_spec["path"]
    if not freeze_path.is_file() or sha256_file(freeze_path) != freeze_spec["sha256"]:
        raise ProtocolError("Stage B input freeze binding 漂移")
    freeze = load_yaml(freeze_path)
    if freeze.get("status") != freeze_spec["required_status"]:
        raise ProtocolError("Stage B input freeze 未完成")
    denominators = freeze["frozen_denominators"]
    if int(denominators["image_count"]) != int(
        freeze_spec["required_image_count"]
    ):
        raise ProtocolError("Stage B frozen image denominator 漂移")
    if int(denominators["checkpoint_count"]) != int(
        freeze_spec["required_checkpoint_count"]
    ):
        raise ProtocolError("Stage B frozen checkpoint denominator 漂移")

    asset = config["asset"]
    expected_target = Path(
        "/root/autodl-tmp/models/dinov2/dinov2_vitg14_reg4_pretrain.pth"
    )
    if Path(asset["target_path"]) != expected_target:
        raise ProtocolError("DINOv2 target path 漂移")
    if int(asset["content_length_bytes"]) != 4546140349:
        raise ProtocolError("DINOv2 expected bytes 漂移")
    if asset.get("etag_is_sha256") is not False:
        raise ProtocolError("multipart ETag 不得冒充 SHA-256")

    locks = config["locks"]
    for name in (
        "feature_extraction",
        "method_inference",
        "quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"DINOv2 asset quality/method lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    return config, freeze


def download_asset(
    config: dict[str, Any],
    run_dir: Path,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    if not os.environ.get("http_proxy") or not os.environ.get("https_proxy"):
        raise ProtocolError("必须先 source /etc/network_turbo")
    asset = config["asset"]
    target = Path(asset["target_path"])
    partial = Path(str(target) + str(asset["partial_suffix"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(asset["content_length_bytes"])
    minimum_free_after = int(config["resources"]["minimum_free_bytes_after_download"])
    disk_before = shutil.disk_usage(target.parent).free

    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_bytes:
            raise ProtocolError("已有 final asset 尺寸错误，禁止覆盖")
        cache_hit = True
        download_seconds = 0.0
        resumed_from_bytes = expected_bytes
    else:
        if partial.exists() and not partial.is_file():
            raise ProtocolError("DINOv2 partial 不是普通文件")
        resumed_from_bytes = partial.stat().st_size if partial.exists() else 0
        if resumed_from_bytes > expected_bytes:
            raise ProtocolError("DINOv2 partial 超过 expected bytes")
        remaining = expected_bytes - resumed_from_bytes
        if disk_before - remaining < minimum_free_after:
            raise ProtocolError("DINOv2 下载后磁盘安全余量不足")
        cache_hit = False
        events.append(
            {
                "event": "download_started",
                "at_utc": _utc_now(),
                "resumed_from_bytes": resumed_from_bytes,
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        command = [
            str(config["download"]["client"]),
            "--fail",
            "--location",
            "--retry",
            str(config["download"]["retry_count"]),
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            str(asset["url"]),
        ]
        started = time.monotonic()
        progress_rows = []
        with (run_dir / "download.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            last_report = -1
            while process.poll() is None:
                elapsed = int(time.monotonic() - started)
                if elapsed // 15 > last_report:
                    last_report = elapsed // 15
                    current = partial.stat().st_size if partial.exists() else 0
                    row = {
                        "metric": "download_progress",
                        "at_utc": _utc_now(),
                        "elapsed_seconds": elapsed,
                        "bytes": current,
                        "fraction": current / expected_bytes,
                    }
                    progress_rows.append(row)
                    _write_jsonl(run_dir / "metrics.jsonl", progress_rows)
                    print(
                        f"download_progress bytes={current}/{expected_bytes} "
                        f"elapsed={elapsed}s",
                        flush=True,
                    )
                time.sleep(2)
        download_seconds = time.monotonic() - started
        if process.returncode != 0:
            raise ProtocolError(f"curl download failed: exit={process.returncode}")
        if not partial.is_file() or partial.stat().st_size != expected_bytes:
            observed = partial.stat().st_size if partial.exists() else -1
            raise ProtocolError(f"DINOv2 download bytes 漂移: {observed}")

    verified_path = target if cache_hit else partial
    hash_started = time.monotonic()
    content_sha256 = sha256_file(verified_path)
    hash_seconds = time.monotonic() - hash_started
    if not cache_hit:
        partial.replace(target)
        events.append(
            {
                "event": "asset_published",
                "at_utc": _utc_now(),
                "target": str(target),
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
    disk_after = shutil.disk_usage(target.parent).free
    if target.stat().st_size != expected_bytes or sha256_file(target) != content_sha256:
        raise ProtocolError("DINOv2 atomic publish 后复核失败")
    return {
        "source_url": asset["url"],
        "target_path": str(target),
        "bytes": target.stat().st_size,
        "sha256": content_sha256,
        "cache_hit": cache_hit,
        "resumed_from_bytes": resumed_from_bytes,
        "download_seconds": download_seconds,
        "hash_seconds": hash_seconds,
        "disk_free_before_bytes": disk_before,
        "disk_free_after_bytes": disk_after,
        "network_turbo_proxy_configured": True,
        "etag": asset["etag"],
        "etag_is_sha256": False,
        "last_modified_utc": asset["last_modified_utc"],
        "s3_version_id": asset["s3_version_id"],
    }


def run(config_path: Path, run_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    if branch != V51_BRANCH:
        raise ProtocolError(f"必须在 {V51_BRANCH} 执行，当前为 {branch}")
    if status:
        raise ProtocolError("DINOv2 formal asset run 要求 clean worktree")
    config, input_freeze = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {"download": config, "input_freeze": input_freeze},
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    asset = download_asset(config, run_dir, events)
    _append_jsonl(
        run_dir / "metrics.jsonl",
        {
            "metric": "dinov2_asset_terminal",
            "bytes": asset["bytes"],
            "sha256": asset["sha256"],
            "cache_hit": asset["cache_hit"],
            "resumed_from_bytes": asset["resumed_from_bytes"],
            "download_seconds": asset["download_seconds"],
            "hash_seconds": asset["hash_seconds"],
        },
    )
    _write_json(
        run_dir / "artifacts/asset.json",
        {
            "schema_version": "worldsim_v51_dinov2_asset_v1",
            "task_id": config["task_id"],
            **asset,
        },
    )
    summary = {
        "schema_version": "worldsim_v51_dinov2_download_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "official_dinov2_vitg14_reg4_checkpoint_downloaded_and_sha256_frozen",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "input_freeze_sha256": config["input_freeze"]["sha256"],
        "asset": asset,
        "feature_extraction_started": False,
        "model_load_started": False,
        "method_inference_started": False,
        "quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "none",
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "fingerprint.json",
        {
            "schema_version": "worldsim_v51_dinov2_download_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "input_freeze_sha256": summary["input_freeze_sha256"],
            "asset_sha256": asset["sha256"],
            "asset_bytes": asset["bytes"],
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_b_dinov2_download_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    try:
        summary = run(args.config.resolve(), run_dir, events)
        events.append({"event": "run_done", "at_utc": _utc_now()})
        _write_jsonl(run_dir / "events.jsonl", events)
        manifest = {
            "schema_version": "worldsim_v51_dinov2_download_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_dinov2_download_status_v1",
                "task_id": summary["task_id"],
                "status": "done",
                "source_commit": summary["source_commit"],
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "finished_at_utc": _utc_now(),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_dinov2_download_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "partial_retained_for_resume": True,
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
