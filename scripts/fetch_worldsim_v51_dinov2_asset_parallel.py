#!/usr/bin/env python3
"""以固定 HTTP ranges 并行续传并冻结 V5.1 DINOv2 权重。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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


def _multipart_etag(path: Path, part_size: int) -> tuple[str, int]:
    part_digests = []
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(part_size), b""):
            part_digests.append(hashlib.md5(chunk).digest())  # noqa: S324 - S3 ETag
    if not part_digests:
        raise ProtocolError("multipart ETag 输入为空")
    digest = hashlib.md5(b"".join(part_digests)).hexdigest()  # noqa: S324
    return f"{digest}-{len(part_digests)}", len(part_digests)


def _ranges(start: int, stop: int, count: int) -> list[tuple[int, int]]:
    if start < 0 or stop <= start or count <= 0:
        raise ProtocolError("parallel range 参数非法")
    total = stop - start
    ranges = []
    for index in range(count):
        first = start + (total * index) // count
        last_exclusive = start + (total * (index + 1)) // count
        if last_exclusive <= first:
            raise ProtocolError("parallel range 为空")
        ranges.append((first, last_exclusive - 1))
    if ranges[0][0] != start or ranges[-1][1] != stop - 1:
        raise ProtocolError("parallel range 未覆盖完整 remainder")
    for left, right in zip(ranges, ranges[1:]):
        if left[1] + 1 != right[0]:
            raise ProtocolError("parallel range 存在 gap/overlap")
    return ranges


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != (
        "worldsim_v51_stage_b_dinov2_download_parallel_v1"
    ):
        raise ProtocolError("parallel DINOv2 schema 漂移")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("parallel DINOv2 task 漂移")
    if config.get("status") != "running":
        raise ProtocolError("parallel DINOv2 status 漂移")

    freeze_path = PROJECT / config["input_freeze"]["path"]
    if not freeze_path.is_file() or sha256_file(freeze_path) != config[
        "input_freeze"
    ]["sha256"]:
        raise ProtocolError("parallel DINOv2 input freeze 漂移")
    freeze = load_yaml(freeze_path)
    if freeze.get("status") != "done":
        raise ProtocolError("parallel DINOv2 input freeze 未完成")

    parent = config["blocked_parent_run"]
    parent_dir = Path(parent["path"])
    for relative, expected in parent["hashes"].items():
        path = parent_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"blocked r002 binding 漂移: {relative}")
    status = json.loads((parent_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "blocked" or not status.get(
        "partial_retained_for_resume"
    ):
        raise ProtocolError("blocked r002 terminal 漂移")

    target = Path(config["asset"]["target_path"])
    prefix = Path(config["frozen_prefix"]["path"])
    parts_dir = Path(config["parallel_download"]["parts_dir"])
    assembled = Path(config["parallel_download"]["assembled_path"])
    expected_target = Path(
        "/root/autodl-tmp/models/dinov2/dinov2_vitg14_reg4_pretrain.pth"
    )
    if target != expected_target or prefix != Path(str(target) + ".partial"):
        raise ProtocolError("parallel DINOv2 target/prefix path 漂移")
    if parts_dir.parent != target.parent or assembled.parent != target.parent:
        raise ProtocolError("parallel DINOv2 staging 必须留在 target filesystem")
    if not prefix.is_file():
        raise ProtocolError("parallel DINOv2 frozen prefix 缺失")
    if prefix.stat().st_size != int(config["frozen_prefix"]["bytes"]):
        raise ProtocolError("parallel DINOv2 prefix bytes 漂移")
    if sha256_file(prefix) != config["frozen_prefix"]["sha256"]:
        raise ProtocolError("parallel DINOv2 prefix SHA 漂移")
    if target.exists() or assembled.exists():
        raise ProtocolError("parallel DINOv2 final/assembled 已存在，禁止覆盖")
    if parts_dir.exists() and any(parts_dir.iterdir()):
        raise ProtocolError("parallel DINOv2 parts dir 非空")

    integrity = config["integrity"]["multipart_etag"]
    if integrity["expected"] != config["asset"]["etag"]:
        raise ProtocolError("parallel DINOv2 ETag binding 漂移")
    expected_bytes = int(config["asset"]["content_length_bytes"])
    expected_parts = (expected_bytes + int(integrity["part_size_bytes"]) - 1) // int(
        integrity["part_size_bytes"]
    )
    if expected_parts != int(integrity["expected_part_count"]):
        raise ProtocolError("parallel DINOv2 multipart part count 漂移")

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
            raise ProtocolError(f"parallel DINOv2 lock 漂移: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 必须保持 pending")
    return config, freeze


def fetch_segments(
    config: dict[str, Any], run_dir: Path, events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    if not os.environ.get("http_proxy") or not os.environ.get("https_proxy"):
        raise ProtocolError("必须先 source /etc/network_turbo")
    expected_bytes = int(config["asset"]["content_length_bytes"])
    prefix_bytes = int(config["frozen_prefix"]["bytes"])
    contract = config["parallel_download"]
    parts_dir = Path(contract["parts_dir"])
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = _ranges(prefix_bytes, expected_bytes, int(contract["segment_count"]))
    disk_before = shutil.disk_usage(parts_dir).free
    temporary_need = (expected_bytes - prefix_bytes) + expected_bytes
    minimum_after = int(
        config["resources"]["minimum_free_bytes_after_temporary_assembly"]
    )
    if disk_before - temporary_need < minimum_after:
        raise ProtocolError("parallel DINOv2 临时 assembly 后磁盘余量不足")

    processes = []
    logs = []
    parts = []
    started = time.monotonic()
    for index, (first, last) in enumerate(ranges):
        part = parts_dir / f"segment-{index:02d}.bin"
        log_path = run_dir / f"download-segment-{index:02d}.log"
        if part.exists() or log_path.exists():
            raise ProtocolError(f"parallel segment staging 已存在: {index}")
        log = log_path.open("wb")
        command = [
            str(contract["client"]),
            "--fail",
            "--location",
            "--retry",
            str(contract["retry_count"]),
            "--retry-all-errors",
            "--header",
            f"Accept-Encoding: {contract['accept_encoding']}",
            "--range",
            f"{first}-{last}",
            "--output",
            str(part),
            str(config["asset"]["url"]),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        logs.append(log)
        parts.append((part, first, last))
    events.append(
        {
            "event": "parallel_download_started",
            "at_utc": _utc_now(),
            "segment_count": len(parts),
            "prefix_bytes": prefix_bytes,
        }
    )
    _write_jsonl(run_dir / "events.jsonl", events)
    last_report = -1
    try:
        while any(process.poll() is None for process in processes):
            elapsed = int(time.monotonic() - started)
            if elapsed // 15 > last_report:
                last_report = elapsed // 15
                downloaded = sum(
                    part.stat().st_size if part.exists() else 0
                    for part, _, _ in parts
                )
                row = {
                    "metric": "parallel_download_progress",
                    "at_utc": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "prefix_bytes": prefix_bytes,
                    "segment_bytes": downloaded,
                    "total_reconstructed_bytes": prefix_bytes + downloaded,
                    "fraction": (prefix_bytes + downloaded) / expected_bytes,
                    "active_processes": sum(
                        process.poll() is None for process in processes
                    ),
                }
                _append_jsonl(run_dir / "metrics.jsonl", row)
                print(
                    f"parallel_progress bytes={prefix_bytes + downloaded}/"
                    f"{expected_bytes} active={row['active_processes']} "
                    f"elapsed={elapsed}s",
                    flush=True,
                )
            time.sleep(2)
    finally:
        for log in logs:
            log.close()
    failures = [
        {"segment": index, "exit": process.returncode}
        for index, process in enumerate(processes)
        if process.returncode != 0
    ]
    if failures:
        raise ProtocolError(f"parallel curl failed: {failures}")

    segment_records = []
    for index, (part, first, last) in enumerate(parts):
        expected = last - first + 1
        observed = part.stat().st_size if part.exists() else -1
        if observed != expected:
            raise ProtocolError(
                f"parallel segment bytes 漂移: {index} {observed}!={expected}"
            )
        segment_records.append(
            {
                "segment": index,
                "range_start": first,
                "range_end_inclusive": last,
                "bytes": observed,
                "sha256": sha256_file(part),
                "path": str(part),
            }
        )
    return segment_records, time.monotonic() - started


def assemble_verify_publish(
    config: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    target = Path(config["asset"]["target_path"])
    prefix = Path(config["frozen_prefix"]["path"])
    assembled = Path(config["parallel_download"]["assembled_path"])
    parts_dir = Path(config["parallel_download"]["parts_dir"])
    expected_bytes = int(config["asset"]["content_length_bytes"])
    assemble_started = time.monotonic()
    with assembled.open("xb") as destination:
        with prefix.open("rb") as source:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        for segment in segments:
            with Path(segment["path"]).open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    if assembled.stat().st_size != expected_bytes:
        raise ProtocolError("parallel assembled bytes 漂移")
    assemble_seconds = time.monotonic() - assemble_started

    hash_started = time.monotonic()
    full_sha256 = sha256_file(assembled)
    hash_seconds = time.monotonic() - hash_started
    etag_contract = config["integrity"]["multipart_etag"]
    etag_started = time.monotonic()
    multipart_etag, part_count = _multipart_etag(
        assembled, int(etag_contract["part_size_bytes"])
    )
    etag_seconds = time.monotonic() - etag_started
    if multipart_etag != etag_contract["expected"]:
        raise ProtocolError(
            f"parallel multipart ETag 漂移: {multipart_etag}"
        )
    if part_count != int(etag_contract["expected_part_count"]):
        raise ProtocolError("parallel multipart part count 漂移")

    assembled.replace(target)
    if target.stat().st_size != expected_bytes:
        raise ProtocolError("parallel atomic publish bytes 复核失败")
    cleanup_files = [prefix] + [Path(segment["path"]) for segment in segments]
    cleanup_bytes = sum(path.stat().st_size for path in cleanup_files)
    for path in cleanup_files:
        if path.parent not in {target.parent, parts_dir}:
            raise ProtocolError(f"parallel cleanup target 越界: {path}")
        path.unlink()
    parts_dir.rmdir()
    return {
        "target_path": str(target),
        "bytes": target.stat().st_size,
        "sha256": full_sha256,
        "multipart_etag": multipart_etag,
        "multipart_part_size_bytes": int(etag_contract["part_size_bytes"]),
        "multipart_part_count": part_count,
        "assemble_seconds": assemble_seconds,
        "sha256_seconds": hash_seconds,
        "multipart_etag_seconds": etag_seconds,
        "cleanup_file_count": len(cleanup_files),
        "cleanup_bytes": cleanup_bytes,
        "temporary_files_removed_after_publish": True,
        "disk_free_after_bytes": shutil.disk_usage(target.parent).free,
    }


def run(config_path: Path, run_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    if branch != V51_BRANCH:
        raise ProtocolError(f"必须在 {V51_BRANCH} 执行，当前为 {branch}")
    if status:
        raise ProtocolError("parallel DINOv2 formal run 要求 clean worktree")
    config, input_freeze = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {"parallel_download": config, "input_freeze": input_freeze},
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    segment_records, download_seconds = fetch_segments(config, run_dir, events)
    _write_json(
        run_dir / "artifacts/segments.json",
        {
            "schema_version": "worldsim_v51_dinov2_segments_v1",
            "task_id": config["task_id"],
            "records": segment_records,
        },
    )
    asset = assemble_verify_publish(config, segment_records)
    asset.update(
        {
            "source_url": config["asset"]["url"],
            "s3_version_id": config["asset"]["s3_version_id"],
            "last_modified_utc": config["asset"]["last_modified_utc"],
            "parallel_segment_count": len(segment_records),
            "frozen_prefix_bytes": int(config["frozen_prefix"]["bytes"]),
            "parallel_download_seconds": download_seconds,
            "network_turbo_proxy_configured": True,
        }
    )
    _write_json(
        run_dir / "artifacts/asset.json",
        {
            "schema_version": "worldsim_v51_dinov2_asset_v1",
            "task_id": config["task_id"],
            **asset,
        },
    )
    _append_jsonl(
        run_dir / "metrics.jsonl",
        {"metric": "parallel_asset_terminal", **asset},
    )
    events.append({"event": "asset_published", "at_utc": _utc_now()})
    _write_jsonl(run_dir / "events.jsonl", events)
    summary = {
        "schema_version": "worldsim_v51_dinov2_parallel_summary_v1",
        "task_id": config["task_id"],
        "status": "done",
        "conclusion": "official_dinov2_vitg14_reg4_parallel_download_sha256_and_s3_etag_frozen",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "input_freeze_sha256": config["input_freeze"]["sha256"],
        "blocked_parent_run_id": config["blocked_parent_run"]["run_id"],
        "asset": asset,
        "model_load_started": False,
        "feature_extraction_started": False,
        "method_inference_started": False,
        "quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": ["V51-F16"],
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "fingerprint.json",
        {
            "schema_version": "worldsim_v51_dinov2_parallel_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "input_freeze_sha256": summary["input_freeze_sha256"],
            "blocked_parent_status_sha256": config["blocked_parent_run"]["hashes"][
                "status.json"
            ],
            "asset_sha256": asset["sha256"],
            "asset_multipart_etag": asset["multipart_etag"],
            "asset_bytes": asset["bytes"],
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT
        / "configs/worldsim_v51/stage_b_dinov2_download_parallel_v1.yaml",
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
            "schema_version": "worldsim_v51_dinov2_parallel_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_dinov2_parallel_status_v1",
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
                "schema_version": "worldsim_v51_dinov2_parallel_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "staging_retained_for_forensics": True,
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
