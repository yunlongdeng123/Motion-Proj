#!/usr/bin/env python
"""从本地 KITTI Tracking ZIP 原子抽取 0000/0001 adapter smoke 子集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_ARCHIVES = {
    "calib": Path("/root/autodl-tmp/data_tracking_calib.zip"),
    "oxts": Path("/root/autodl-tmp/data_tracking_oxts.zip"),
    "label_02": Path("/root/autodl-tmp/data_tracking_label_2.zip"),
    "image_02": Path("/root/autodl-tmp/data_tracking_image_2.zip"),
    "image_03": Path("/root/autodl-tmp/data_tracking_image_3.zip"),
    "velodyne": Path("/root/autodl-tmp/data_tracking_velodyne.zip"),
}
DEFAULT_OUTPUT = Path("/root/autodl-tmp/data/worldsim_v5/kitti_tracking_smoke")
DEFAULT_MANIFEST = Path(
    "/root/autodl-tmp/data/worldsim_v5/manifests/kitti_tracking_smoke_raw_v1.json"
)
DEFAULT_ARCHIVE_AUDIT = (
    Path(__file__).resolve().parents[1] / "docs/KITTI_TRACKING_ARCHIVE_METADATA_V5.json"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_COMPONENTS = {"calib", "oxts", "label_02"}


class KittiSmokeExtractionError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise KittiSmokeExtractionError(
            process.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return process.stdout.strip()


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("ab") as handle:
        handle.write(
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise KittiSmokeExtractionError(f"ZIP member 路径不安全: {name}")
    return path


def selected_infos(
    archive: zipfile.ZipFile, component: str, sequences: tuple[str, ...]
) -> list[zipfile.ZipInfo]:
    if component in FILE_COMPONENTS:
        targets = {f"training/{component}/{sequence}.txt" for sequence in sequences}
        rows = [info for info in archive.infolist() if info.filename in targets]
        missing = targets - {info.filename for info in rows}
    else:
        prefixes = tuple(
            f"training/{component}/{sequence}/" for sequence in sequences
        )
        rows = [
            info
            for info in archive.infolist()
            if not info.is_dir() and info.filename.startswith(prefixes)
        ]
        found_sequences = {safe_member(info.filename).parts[2] for info in rows}
        missing = set(sequences) - found_sequences
    if missing:
        raise KittiSmokeExtractionError(
            f"{component} 缺少 sequences: {sorted(missing)}"
        )
    for info in rows:
        safe_member(info.filename)
        if info.file_size <= 0:
            raise KittiSmokeExtractionError(
                f"ZIP member 为空: {archive.filename}:{info.filename}"
            )
    return sorted(rows, key=lambda info: info.filename)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audited_archives(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 旧审计产物把 adapter 阻塞原因写进了 status；这里只做兼容读取，
    # 新增 run/status 仍严格使用 pending/running/blocked/done/rejected。
    legacy_status = payload.get("status") == "blocked_dataset_adapter"
    if (
        (payload.get("status") != "done" and not legacy_status)
        or payload.get("task_id") != "WS-V5-D1-KITTI-ARCHIVE-AUDIT-01"
    ):
        raise KittiSmokeExtractionError("KITTI archive audit 未完成或 task 漂移")
    gates = payload.get("gates", {})
    required_safe_gates = (
        "all_archives_present",
        "archive_sha256_recorded",
        "central_directories_readable",
        "expected_component_paths",
        "expected_sequence_sets",
        "no_duplicate_members",
        "safe_member_paths",
        "unencrypted_members",
    )
    if any(gates.get(name) is not True for name in required_safe_gates):
        raise KittiSmokeExtractionError("KITTI archive safe-extraction gates 未通过")
    if legacy_status and gates.get("sensor_frame_alignment") is not False:
        raise KittiSmokeExtractionError("legacy adapter 阻塞原因与审计事实不一致")
    records = {str(row["component"]): row for row in payload["archives"]}
    for component, archive in DEFAULT_ARCHIVES.items():
        row = records.get(component)
        if row is None:
            raise KittiSmokeExtractionError(f"archive audit 缺少 {component}")
        if Path(row["path"]) != archive or int(row["archive_bytes"]) != archive.stat().st_size:
            raise KittiSmokeExtractionError(f"archive audit path/bytes 漂移: {component}")
        if row.get("central_directory_readable") is not True or int(row.get("unsafe_member_count", -1)) != 0:
            raise KittiSmokeExtractionError(f"archive audit gate 未通过: {component}")
    return records, sha256_file(path)


def extract(
    *,
    archives: dict[str, Path],
    sequences: tuple[str, ...],
    output: Path,
    manifest_path: Path,
    archive_evidence: dict[str, dict[str, Any]] | None = None,
    archive_audit_sha256: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise KittiSmokeExtractionError(f"输出已存在，禁止覆盖: {output}")
    if manifest_path.exists():
        raise KittiSmokeExtractionError(f"manifest 已存在，禁止覆盖: {manifest_path}")
    staging = output.with_name(output.name + f".partial.{os.getpid()}")
    if staging.exists():
        raise KittiSmokeExtractionError(f"staging 已存在: {staging}")
    staging.mkdir(parents=True)
    archive_records = []
    files = []
    try:
        for component, archive_path in archives.items():
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                infos = selected_infos(archive, component, sequences)
                archive_records.append(
                    {
                        "component": component,
                        "archive": str(archive_path),
                        "archive_bytes": archive_path.stat().st_size,
                        "archive_sha256": (
                            archive_evidence[component]["sha256"]
                            if archive_evidence is not None
                            else sha256_file(archive_path)
                        ),
                        "archive_sha256_source": (
                            "frozen_archive_audit"
                            if archive_evidence is not None
                            else "recomputed_for_fixture"
                        ),
                        "selected_count": len(infos),
                        "selected_uncompressed_bytes": sum(
                            info.file_size for info in infos
                        ),
                    }
                )
                for info in infos:
                    relative = safe_member(info.filename)
                    target = staging.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.open(info) as source, target.open("xb") as sink:
                        while chunk := source.read(1 << 20):
                            sink.write(chunk)
                            digest.update(chunk)
                    if target.stat().st_size != info.file_size:
                        raise KittiSmokeExtractionError(
                            f"解压字节漂移: {info.filename}"
                        )
                    files.append(
                        {
                            "component": component,
                            "path": info.filename,
                            "bytes": info.file_size,
                            "crc32": f"{info.CRC:08x}",
                            "sha256": digest.hexdigest(),
                        }
                    )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
    except Exception:
        # 保留非空 staging 供失败取证，不做自动删除或覆盖。
        raise
    payload = {
        "schema_version": "worldsim_v5_kitti_tracking_smoke_raw_v1",
        "task_id": "WS-V5-D1-KITTI-ADAPTER-01",
        "status": "done",
        "purpose": "result-blind adapter smoke including known sensor-gap sequence",
        "sequences": list(sequences),
        "sequence_selection": {
            "0000": "nominal_alignment_smoke",
            "0001": "known_lidar_gap_abstain_smoke",
            "quality_based": False,
        },
        "output": str(output),
        "archives": archive_records,
        "archive_audit_sha256": archive_audit_sha256,
        "file_count": len(files),
        "uncompressed_bytes": sum(row["bytes"] for row in files),
        "files": sorted(files, key=lambda row: row["path"]),
        "sensor_payload_decoded_for_quality": False,
        "method_parameter_search": False,
        "complete": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(manifest_path, payload)
    return payload


def initialize_formal_run(
    run_dir: Path,
    *,
    output: Path,
    manifest: Path,
    archive_audit: Path,
    sequences: tuple[str, ...],
) -> None:
    if run_dir.exists():
        raise KittiSmokeExtractionError(f"run 目录已存在，禁止复用：{run_dir}")
    if git_output("status", "--porcelain"):
        raise KittiSmokeExtractionError("正式 KITTI extraction 要求 clean worktree")
    for name in ("artifacts", "source_snapshot"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    resolved = {
        "schema_version": "worldsim_v5_kitti_smoke_extraction_config_v1",
        "task_id": "WS-V5-D1-KITTI-ADAPTER-01",
        "status": "running",
        "stage": "two_sequence_selective_extraction",
        "sequences": list(sequences),
        "output": str(output),
        "dataset_manifest": str(manifest),
        "archive_audit": str(archive_audit),
        "archives": {key: str(value) for key, value in DEFAULT_ARCHIVES.items()},
        "checkpoint": "N/A_data_preparation",
        "quality_read": False,
        "method_training": False,
        "method_inference": False,
        "parameter_search": False,
    }
    atomic_json(run_dir / "resolved_config.json", resolved)
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "at_utc": now_utc(),
            "event": "kitti_smoke_selective_extraction_started",
            "sequences": list(sequences),
            "quality_read": False,
        },
    )


def finalize_formal_run(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    archive_audit: Path,
) -> dict[str, Any]:
    shutil.copy2(manifest_path, run_dir / "artifacts/raw_manifest.json")
    snapshot_relpaths = (
        "scripts/extract_worldsim_v5_kitti_smoke.py",
        "tests/test_worldsim_v5_kitti_smoke_extraction.py",
        "docs/KITTI_TRACKING_ARCHIVE_METADATA_V5.json",
    )
    snapshots = {}
    for relpath in snapshot_relpaths:
        source = PROJECT_ROOT / relpath
        target = run_dir / "source_snapshot" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[relpath] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    fingerprint = {
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.json"),
        "raw_manifest_file_sha256": sha256_file(manifest_path),
        "raw_manifest_content_sha256": payload["manifest_sha256"],
        "archive_audit_file_sha256": sha256_file(archive_audit),
        "source_snapshots": snapshots,
        "project_git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "dirty": False,
        },
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    summary = {
        "schema_version": "worldsim_v5_kitti_smoke_extraction_summary_v1",
        "task_id": "WS-V5-D1-KITTI-ADAPTER-01",
        "status": "done",
        "stage": "two_sequence_selective_extraction",
        "finished_at_utc": now_utc(),
        "sequences": payload["sequences"],
        "file_count": payload["file_count"],
        "uncompressed_bytes": payload["uncompressed_bytes"],
        "raw_manifest": str(manifest_path),
        "raw_manifest_file_sha256": sha256_file(manifest_path),
        "raw_manifest_content_sha256": payload["manifest_sha256"],
        "checkpoint": "N/A_data_preparation",
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "quality_read": False,
        "method_training_started": False,
        "method_inference_started": False,
        "parameter_search": False,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": summary["task_id"],
            "status": "done",
            "stage": summary["stage"],
            "finished_at_utc": summary["finished_at_utc"],
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "at_utc": now_utc(),
            "event": "kitti_smoke_selective_extraction_complete",
            "status": "done",
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_kitti_smoke_extraction_run_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "artifacts": artifacts,
            "quality_read": False,
            "method_training_started": False,
            "method_inference_started": False,
            "parameter_search": False,
        },
    )
    return summary


def record_formal_blocked(run_dir: Path, error: BaseException) -> None:
    if (run_dir / "status.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "at_utc": now_utc(),
        "event": "kitti_smoke_selective_extraction_blocked",
        "status": "blocked",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    append_jsonl(run_dir / "events.jsonl", event)
    fingerprint = {
        "project_head": git_output("rev-parse", "HEAD"),
        "error": event,
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    summary = {
        "schema_version": "worldsim_v5_kitti_smoke_extraction_summary_v1",
        "task_id": "WS-V5-D1-KITTI-ADAPTER-01",
        "status": "blocked",
        "stage": "two_sequence_selective_extraction",
        "reason": "selective_extraction_failed",
        "error": event,
        "checkpoint": "N/A_data_preparation",
        "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        "quality_read": False,
        "method_training_started": False,
        "method_inference_started": False,
        "parameter_search": False,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": summary["task_id"],
            "status": "blocked",
            "stage": summary["stage"],
            "finished_at_utc": now_utc(),
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_kitti_smoke_extraction_run_manifest_v1",
            "task_id": summary["task_id"],
            "status": "blocked",
            "artifacts": artifacts,
            "quality_read": False,
            "method_training_started": False,
            "method_inference_started": False,
            "parameter_search": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-audit", type=Path, default=DEFAULT_ARCHIVE_AUDIT)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    sequences = tuple(args.sequence or ["0000", "0001"])
    if sequences != ("0000", "0001"):
        raise KittiSmokeExtractionError(
            "首轮 adapter smoke identity 已冻结为 0000/0001"
        )
    run_dir = args.run_dir.resolve() if args.run_dir is not None else None
    existed_before = run_dir.exists() if run_dir is not None else False
    try:
        if run_dir is not None:
            initialize_formal_run(
                run_dir,
                output=args.output.resolve(),
                manifest=args.manifest.resolve(),
                archive_audit=args.archive_audit.resolve(),
                sequences=sequences,
            )
        evidence, audit_sha256 = audited_archives(args.archive_audit.resolve())
        payload = extract(
            archives=DEFAULT_ARCHIVES,
            sequences=sequences,
            output=args.output.resolve(),
            manifest_path=args.manifest.resolve(),
            archive_evidence=evidence,
            archive_audit_sha256=audit_sha256,
        )
        if run_dir is not None:
            finalize_formal_run(
                run_dir,
                payload,
                manifest_path=args.manifest.resolve(),
                archive_audit=args.archive_audit.resolve(),
            )
    except BaseException as error:
        if run_dir is not None and not existed_before:
            record_formal_blocked(run_dir, error)
        raise
    print(
        json.dumps(
            {
                "status": payload["status"],
                "file_count": payload["file_count"],
                "uncompressed_bytes": payload["uncompressed_bytes"],
                "manifest_sha256": payload["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
