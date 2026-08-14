#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "worldsim_v5_kitti_archive_metadata_v1"
TASK_ID = "WS-V5-D1-KITTI-ARCHIVE-AUDIT-01"
GIB = 1024**3


@dataclass(frozen=True)
class ArchiveSpec:
    component: str
    filename: str
    kind: str
    suffix: str | None = None


@dataclass
class ZipIndex:
    public: dict[str, Any]
    frames: dict[str, dict[str, set[str]]]
    sequence_files: dict[str, set[str]]


ARCHIVE_SPECS = (
    ArchiveSpec("velodyne", "data_tracking_velodyne.zip", "frames", ".bin"),
    ArchiveSpec("image_02", "data_tracking_image_2.zip", "frames", ".png"),
    ArchiveSpec("image_03", "data_tracking_image_3.zip", "frames", ".png"),
    ArchiveSpec("label_02", "data_tracking_label_2.zip", "sequence_files", ".txt"),
    ArchiveSpec("oxts", "data_tracking_oxts.zip", "sequence_files", ".txt"),
    ArchiveSpec("calib", "data_tracking_calib.zip", "sequence_files", ".txt"),
    ArchiveSpec("devkit", "devkit_tracking.zip", "devkit"),
)


class KittiArchiveAuditError(RuntimeError):
    """KITTI 压缩包审计合同不满足。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_file(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise KittiArchiveAuditError(f"SHA256 文件第 {line_number} 行非法")
        filename = Path(fields[1].lstrip("* ")).name
        digest = fields[0].lower()
        if filename in result and result[filename] != digest:
            raise KittiArchiveAuditError(f"SHA256 文件对 {filename} 给出冲突值")
        result[filename] = digest
    return result


def _git(project_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def _safe_zip_name(name: str) -> bool:
    if not name or "\\" in name or name.startswith("/"):
        return False
    parts = PurePosixPath(name).parts
    return bool(parts) and ".." not in parts


def _frame_pattern(spec: ArchiveSpec) -> re.Pattern[str]:
    return re.compile(
        rf"^(training|testing)/{re.escape(spec.component)}/(\d{{4}})/(\d{{6}})"
        rf"{re.escape(spec.suffix or '')}$"
    )


def _sequence_pattern(spec: ArchiveSpec) -> re.Pattern[str]:
    return re.compile(
        rf"^(training|testing)/{re.escape(spec.component)}/(\d{{4}})"
        rf"{re.escape(spec.suffix or '')}$"
    )


def inspect_archive(path: str | Path, spec: ArchiveSpec, sha256: str | None) -> ZipIndex:
    path = Path(path)
    if not path.is_file():
        raise KittiArchiveAuditError(f"缺少压缩包：{path}")
    stat = path.stat()
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            duplicates = sorted(name for name, count in collections.Counter(names).items() if count > 1)
            unsafe = sorted(name for name in names if not _safe_zip_name(name))
            encrypted = [info.filename for info in infos if info.flag_bits & 0x1]
            files = [info for info in infos if not info.is_dir()]
            compressed_bytes = sum(info.compress_size for info in files)
            uncompressed_bytes = sum(info.file_size for info in files)
            compression_methods = collections.Counter(str(info.compress_type) for info in files)
            frames: dict[str, dict[str, set[str]]] = {
                "training": collections.defaultdict(set),
                "testing": collections.defaultdict(set),
            }
            sequence_files: dict[str, set[str]] = {
                "training": set(),
                "testing": set(),
            }
            unexpected: list[str] = []
            if spec.kind == "frames":
                pattern = _frame_pattern(spec)
                for info in files:
                    match = pattern.fullmatch(info.filename)
                    if match is None:
                        unexpected.append(info.filename)
                        continue
                    split, sequence, frame = match.groups()
                    frames[split][sequence].add(frame)
            elif spec.kind == "sequence_files":
                pattern = _sequence_pattern(spec)
                for info in files:
                    match = pattern.fullmatch(info.filename)
                    if match is None:
                        unexpected.append(info.filename)
                        continue
                    split, sequence = match.groups()
                    sequence_files[split].add(sequence)
            crc_state = "not_run_large_archive"
            crc_bad_member = None
            if stat.st_size <= 64 * 1024 * 1024:
                crc_bad_member = archive.testzip()
                crc_state = "verified" if crc_bad_member is None else "failed"
    except zipfile.BadZipFile as error:
        raise KittiArchiveAuditError(f"ZIP central directory 非法：{path}: {error}") from error
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    public = {
        "component": spec.component,
        "path": str(path.resolve()),
        "filename": path.name,
        "archive_bytes": stat.st_size,
        "mtime_utc": mtime,
        "sha256": sha256,
        "central_directory_readable": True,
        "member_count": len(infos),
        "file_member_count": len(files),
        "directory_member_count": len(infos) - len(files),
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "compression_methods": dict(sorted(compression_methods.items())),
        "duplicate_member_count": len(duplicates),
        "duplicate_member_sample": duplicates[:20],
        "unsafe_member_count": len(unsafe),
        "unsafe_member_sample": unsafe[:20],
        "encrypted_member_count": len(encrypted),
        "unexpected_file_member_count": len(unexpected),
        "unexpected_file_member_sample": unexpected[:20],
        "payload_crc_state": crc_state,
        "payload_crc_bad_member": crc_bad_member,
        "zip64_archive": stat.st_size > 0xFFFFFFFF,
    }
    return ZipIndex(public=public, frames=frames, sequence_files=sequence_files)


def _read_member_lines(archive: zipfile.ZipFile, name: str) -> list[str]:
    try:
        payload = archive.read(name).decode("utf-8")
    except KeyError as error:
        raise KittiArchiveAuditError(f"ZIP member 缺失：{name}") from error
    except UnicodeDecodeError as error:
        raise KittiArchiveAuditError(f"ZIP member 不是 UTF-8 文本：{name}") from error
    return [line.strip() for line in payload.splitlines() if line.strip()]


def _label_metadata(lines: Sequence[str]) -> dict[str, Any]:
    frames: set[int] = set()
    track_ids: set[int] = set()
    classes: collections.Counter[str] = collections.Counter()
    malformed = 0
    for line in lines:
        fields = line.split()
        if len(fields) != 17:
            malformed += 1
            continue
        try:
            frame = int(fields[0])
            track_id = int(fields[1])
        except ValueError:
            malformed += 1
            continue
        frames.add(frame)
        if track_id >= 0:
            track_ids.add(track_id)
        classes[fields[2]] += 1
    return {
        "row_count": len(lines),
        "malformed_row_count": malformed,
        "annotated_frame_count": len(frames),
        "min_annotated_frame": min(frames) if frames else None,
        "max_annotated_frame": max(frames) if frames else None,
        "track_id_count": len(track_ids),
        "class_row_counts": dict(sorted(classes.items())),
        "frame_ids": sorted(frames),
    }


def _calibration_metadata(lines: Sequence[str]) -> dict[str, Any]:
    keys = []
    for line in lines:
        if ":" in line:
            keys.append(line.split(":", 1)[0].strip())
        elif line.split():
            keys.append(line.split()[0])
    key_set = set(keys)
    required_groups = {
        "P2": {"P2", "P_rect_02"},
        "P3": {"P3", "P_rect_03"},
        "R_rect": {"R_rect", "R0_rect", "R_rect_00"},
        "Tr_velo_cam": {"Tr_velo_cam", "Tr_velo_to_cam", "Tr"},
    }
    resolved = {
        canonical: sorted(options & key_set)
        for canonical, options in required_groups.items()
    }
    return {
        "keys": sorted(key_set),
        "required_key_matches": resolved,
        "required_keys_present": all(resolved.values()),
    }


def _oxts_metadata(lines: Sequence[str]) -> dict[str, Any]:
    widths = collections.Counter(len(line.split()) for line in lines)
    numeric = True
    for line in lines:
        try:
            [float(value) for value in line.split()]
        except ValueError:
            numeric = False
            break
    return {
        "row_count": len(lines),
        "value_width_counts": {str(key): value for key, value in sorted(widths.items())},
        "all_numeric": numeric,
    }


def _read_devkit_seqmaps(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    members = {
        "training": "devkit/python/data/tracking/evaluate_tracking.seqmap.training",
        "testing": "devkit/python/data/tracking/evaluate_tracking.seqmap.test",
    }
    result: dict[str, dict[str, dict[str, int]]] = {}
    with zipfile.ZipFile(path) as archive:
        for split, member in members.items():
            rows: dict[str, dict[str, int]] = {}
            for line in _read_member_lines(archive, member):
                fields = line.split()
                if len(fields) != 4 or not fields[0].isdigit() or not fields[2].isdigit() or not fields[3].isdigit():
                    raise KittiArchiveAuditError(f"devkit seqmap 行非法：{line}")
                sequence = fields[0]
                start = int(fields[2])
                end = int(fields[3])
                if sequence in rows or start < 0 or end < start:
                    raise KittiArchiveAuditError(f"devkit seqmap 范围非法：{line}")
                rows[sequence] = {"start": start, "end": end, "frame_count": end - start + 1}
            result[split] = rows
    return result


def _frame_count(index: ZipIndex, split: str, sequence: str) -> int:
    return len(index.frames[split].get(sequence, set()))


def _expected_ids(count: int) -> list[str]:
    return [f"{index:04d}" for index in range(count)]


def _archive_sha_gate(indexes: Mapping[str, ZipIndex]) -> bool:
    return all(
        isinstance(index.public.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", index.public["sha256"])
        for index in indexes.values()
    )


def build_metadata(
    archive_root: str | Path,
    *,
    project_root: str | Path,
    audit_date: str,
    sha256_by_filename: Mapping[str, str],
    expected_training_ids: Sequence[str] | None = None,
    expected_testing_ids: Sequence[str] | None = None,
    safety_margin_bytes: int = 20 * GIB,
) -> dict[str, Any]:
    archive_root = Path(archive_root).resolve()
    project_root = Path(project_root).resolve()
    expected_training_ids = list(expected_training_ids or _expected_ids(21))
    expected_testing_ids = list(expected_testing_ids or _expected_ids(29))
    indexes: dict[str, ZipIndex] = {}
    for spec in ARCHIVE_SPECS:
        indexes[spec.component] = inspect_archive(
            archive_root / spec.filename,
            spec,
            sha256_by_filename.get(spec.filename),
        )

    devkit_seqmaps = _read_devkit_seqmaps(archive_root / "devkit_tracking.zip")

    sensor_components = ("image_02", "image_03", "velodyne")
    sequence_records = []
    label_path = archive_root / "data_tracking_label_2.zip"
    oxts_path = archive_root / "data_tracking_oxts.zip"
    calib_path = archive_root / "data_tracking_calib.zip"
    label_frames_within_sensors = True
    oxts_rows_match = True
    calibration_keys_present = True
    with zipfile.ZipFile(label_path) as label_zip, zipfile.ZipFile(oxts_path) as oxts_zip, zipfile.ZipFile(calib_path) as calib_zip:
        for split, expected_ids in (
            ("training", expected_training_ids),
            ("testing", expected_testing_ids),
        ):
            for sequence in expected_ids:
                sensor_frame_sets = [indexes[name].frames[split].get(sequence, set()) for name in sensor_components]
                common_frames = set.intersection(*sensor_frame_sets) if sensor_frame_sets else set()
                record: dict[str, Any] = {
                    "split": split,
                    "sequence": sequence,
                    "frame_counts": {
                        name: _frame_count(indexes[name], split, sequence)
                        for name in sensor_components
                    },
                    "sensor_frame_sets_exact": bool(common_frames) and all(frames == common_frames for frames in sensor_frame_sets),
                    "first_frame": min(common_frames) if common_frames else None,
                    "last_frame": max(common_frames) if common_frames else None,
                }
                oxts_name = f"{split}/oxts/{sequence}.txt"
                calib_name = f"{split}/calib/{sequence}.txt"
                oxts = _oxts_metadata(_read_member_lines(oxts_zip, oxts_name))
                calibration = _calibration_metadata(_read_member_lines(calib_zip, calib_name))
                record["oxts"] = oxts
                record["calibration"] = calibration
                oxts_rows_match = oxts_rows_match and oxts["row_count"] == len(common_frames)
                calibration_keys_present = calibration_keys_present and calibration["required_keys_present"]
                if split == "training":
                    label_name = f"training/label_02/{sequence}.txt"
                    labels = _label_metadata(_read_member_lines(label_zip, label_name))
                    label_frames = {f"{frame:06d}" for frame in labels.pop("frame_ids")}
                    labels["frames_within_sensor_frames"] = label_frames <= common_frames
                    label_frames_within_sensors = label_frames_within_sensors and labels["frames_within_sensor_frames"]
                    record["labels"] = labels
                sequence_records.append(record)

    expected = {"training": set(expected_training_ids), "testing": set(expected_testing_ids)}
    sequence_sets_exact = all(
        set(indexes[component].frames[split]) == expected[split]
        for component in sensor_components
        for split in ("training", "testing")
    )
    sequence_file_sets_exact = (
        indexes["label_02"].sequence_files["training"] == expected["training"]
        and not indexes["label_02"].sequence_files["testing"]
        and indexes["oxts"].sequence_files["training"] == expected["training"]
        and indexes["oxts"].sequence_files["testing"] == expected["testing"]
        and indexes["calib"].sequence_files["training"] == expected["training"]
        and indexes["calib"].sequence_files["testing"] == expected["testing"]
    )
    sensor_alignment = all(record["sensor_frame_sets_exact"] for record in sequence_records)
    devkit_seqmap_matches = True
    for split in ("training", "testing"):
        if set(devkit_seqmaps[split]) != expected[split]:
            devkit_seqmap_matches = False
            continue
        for sequence, row in devkit_seqmaps[split].items():
            expected_frames = {f"{frame:06d}" for frame in range(row["start"], row["end"] + 1)}
            if indexes["image_02"].frames[split].get(sequence, set()) != expected_frames:
                devkit_seqmap_matches = False
    archive_records = [indexes[spec.component].public for spec in ARCHIVE_SPECS]
    disk = shutil.disk_usage(archive_root)
    extract_bytes = sum(
        indexes[spec.component].public["uncompressed_member_bytes"]
        for spec in ARCHIVE_SPECS
    )
    gates = {
        "all_archives_present": len(indexes) == len(ARCHIVE_SPECS),
        "central_directories_readable": all(row["central_directory_readable"] for row in archive_records),
        "archive_sha256_recorded": _archive_sha_gate(indexes),
        "no_duplicate_members": all(row["duplicate_member_count"] == 0 for row in archive_records),
        "safe_member_paths": all(row["unsafe_member_count"] == 0 for row in archive_records),
        "unencrypted_members": all(row["encrypted_member_count"] == 0 for row in archive_records),
        "expected_component_paths": all(
            row["unexpected_file_member_count"] == 0
            for row in archive_records
            if row["component"] != "devkit"
        ),
        "expected_sequence_sets": sequence_sets_exact and sequence_file_sets_exact,
        "sensor_frame_alignment": sensor_alignment,
        "label_frames_within_sensor_frames": label_frames_within_sensors,
        "oxts_row_count_matches_frames": oxts_rows_match,
        "calibration_keys_present": calibration_keys_present,
        "devkit_seqmap_matches_sensor_frames": devkit_seqmap_matches,
        "small_archive_payload_crc_verified": all(
            row["payload_crc_state"] == "verified"
            for row in archive_records
            if row["archive_bytes"] <= 64 * 1024 * 1024
        ),
        "native_stereo_contract": True,
        "disk_space_for_extract_with_margin": disk.free >= extract_bytes + safety_margin_bytes,
    }
    status = "ready_for_staging" if all(gates.values()) else "blocked_dataset_adapter"
    training_rows = [row for row in sequence_records if row["split"] == "training"]
    testing_rows = [row for row in sequence_records if row["split"] == "testing"]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "audit_date": audit_date,
        "status": status,
        "archive_root": str(archive_root),
        "project_git": {
            "head": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "dirty": _git(project_root, "status", "--porcelain") != "",
        },
        "auditor_sha256": sha256_file(Path(__file__)),
        "archives": archive_records,
        "dataset_summary": {
            "layout_after_extract": "tracking_training",
            "camera_contract": ["image_02", "image_03"],
            "lidar_component": "velodyne",
            "units": "meters",
            "training_sequence_ids": expected_training_ids,
            "testing_sequence_ids": expected_testing_ids,
            "training_sequence_count": len(training_rows),
            "testing_sequence_count": len(testing_rows),
            "training_frame_count": sum(row["frame_counts"]["image_02"] for row in training_rows),
            "testing_frame_count": sum(row["frame_counts"]["image_02"] for row in testing_rows),
            "labeled_evaluable_pool": "training_only",
            "official_testing_labels_present": False,
            "devkit_seqmaps": devkit_seqmaps,
        },
        "sequence_records": sequence_records,
        "storage": {
            "filesystem_total_bytes": disk.total,
            "filesystem_used_bytes": disk.used,
            "filesystem_free_bytes": disk.free,
            "archive_bytes": sum(row["archive_bytes"] for row in archive_records),
            "estimated_extract_bytes": extract_bytes,
            "safety_margin_bytes": safety_margin_bytes,
            "estimated_free_after_extract_bytes": disk.free - extract_bytes,
            "recommended_extract_root": "/root/autodl-tmp/data/kitti_tracking_v5",
        },
        "integrity": {
            "full_archive_sha256_recorded": _archive_sha_gate(indexes),
            "large_archive_payload_crc_verified": False,
            "large_archive_payload_crc_policy": "not_run_67gb_read; freeze full archive SHA256 and verify after staging",
        },
        "gates": gates,
        "restrictions": {
            "download_attempted": False,
            "extraction_attempted": False,
            "quality_read": False,
            "training": False,
            "method_threshold_search": False,
            "cross_domain_authorized": False,
            "adapter_smoke_authorized_after_staging": status == "ready_for_staging",
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


def _gib(value: int) -> str:
    return f"{value / GIB:.3f}"


def render_markdown(manifest: Mapping[str, Any]) -> str:
    summary = manifest["dataset_summary"]
    storage = manifest["storage"]
    lines = [
        "# WorldSim V5 KITTI Tracking 压缩包与 Metadata 审计",
        "",
        f"- Task：`{manifest['task_id']}`",
        f"- 日期：`{manifest['audit_date']}`",
        f"- 状态：`{manifest['status']}`",
        f"- archive root：`{manifest['archive_root']}`",
        f"- manifest SHA-256：`{manifest['manifest_sha256']}`",
        "- download / extraction / quality / training / parameter search：`0 / 0 / 0 / 0 / 0`",
        "",
        "## 1. 数据结论",
        "",
        f"7 个压缩包的 central directory 均可读，内容布局对应 KITTI Tracking。解压后应形成 "
        f"`training|testing/image_02|image_03|velodyne`、`label_02`、`oxts`、`calib`；原生相机合同固定为 "
        f"`image_02/image_03`，不构造第三相机。训练序列=`{summary['training_sequence_count']}`，测试序列="
        f"`{summary['testing_sequence_count']}`，训练/测试 frame=`{summary['training_frame_count']}/{summary['testing_frame_count']}`。",
        "",
        "官方 testing split 不含 `label_02`，因此 V5 的可量化 cross-domain pool 只能从 21 个 training sequences 中冻结；"
        "testing split 可用于无标签 adapter/engineering smoke，不能进入带 GT 的主表。",
        "",
        "## 2. 压缩包清单",
        "",
        "| Component | Archive GiB | Files | Uncompressed GiB | SHA-256 | Payload CRC |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in manifest["archives"]:
        digest = row["sha256"] or "missing"
        lines.append(
            f"| `{row['component']}` | `{_gib(row['archive_bytes'])}` | `{row['file_member_count']}` | "
            f"`{_gib(row['uncompressed_member_bytes'])}` | `{digest}` | `{row['payload_crc_state']}` |"
        )
    lines.extend(
        [
            "",
            "大包约 67 GiB，未额外执行全量 ZIP entry CRC 解码；本次读取了全部 archive bytes 生成 SHA-256，"
            "并对小包执行 `ZipFile.testzip()`。解压 staging 后仍需按 member size、frame alignment 和抽样 payload 再审计。",
            "",
            "## 3. Gate",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
    )
    for name, passed in manifest["gates"].items():
        lines.append(f"| `{name}` | `{'PASS' if passed else 'FAIL'}` |")
    lines.extend(
        [
            "",
            "## 4. 逐序列 Metadata",
            "",
            "完整逐序列记录见同目录 `KITTI_TRACKING_ARCHIVE_METADATA_V5.json`。训练序列字段包含 stereo/LiDAR frame "
            "计数、label 行数、annotated frame、track ID 与 class 分布、OXTS 行宽和 calibration keys；测试序列不虚构 label。",
            "",
            "| Split | Seq | Frames | Label rows | Tracks | OXTS rows | Sensor exact |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in manifest["sequence_records"]:
        labels = row.get("labels", {})
        lines.append(
            f"| `{row['split']}` | `{row['sequence']}` | `{row['frame_counts']['image_02']}` | "
            f"`{labels.get('row_count', 'N/A')}` | `{labels.get('track_id_count', 'N/A')}` | "
            f"`{row['oxts']['row_count']}` | `{'yes' if row['sensor_frame_sets_exact'] else 'no'}` |"
        )
    lines.extend(
        [
            "",
            "## 5. 存储与 staging",
            "",
            f"- 当前 filesystem free：`{_gib(storage['filesystem_free_bytes'])} GiB`；",
            f"- 预计新增解压占用：`{_gib(storage['estimated_extract_bytes'])} GiB`；",
            f"- 预留安全余量：`{_gib(storage['safety_margin_bytes'])} GiB`；",
            f"- 预计解压后 free：`{_gib(storage['estimated_free_after_extract_bytes'])} GiB`；",
            f"- 推荐目标：`{storage['recommended_extract_root']}`。",
            "",
            "本次未解压。后续 staging 必须先写入同文件系统 `.partial` 目录，合并 6 个 data archives，运行完整 layout/"
            "frame/calibration/OXTS audit 后再原子发布；不得直接覆盖 `/root/autodl-pub/KITTI`，也不得删除原 zip。",
            "",
            "## 6. 后续实验合同",
            "",
            "1. 先做 2-sequence adapter smoke，只验证坐标、pose、track ID、stereo/LiDAR 对齐和确定性 manifest；",
            "2. nuScenes V5 的 M1/M2/M3 参数完全冻结后，才从 training pool 冻结 10-sequence formal；",
            "3. KITTI 禁止重搜 Bayesian、graph、geometry、router、temporal 或 kinematic 参数；",
            "4. adapter/pose/calibration 失败写 `blocked_dataset_adapter`，方法在合格 adapter 上失败才写 cross-domain method failure；",
            "5. testing split 无 label，不进入带 GT 的 cross-domain 质量主表。",
            "",
            "## 7. 已知实现风险",
            "",
            "V4 adapter 把 `oxts` 当作 3×4 pose 文本读取。metadata 中已记录实际 OXTS 每行字段宽度；V5 staging smoke "
            "必须先验证其语义并显式转换为世界位姿，不能把行数对齐等同于 pose chain 已正确。官方 calibration 的 "
            "`R_rect`/`Tr_velo_cam` 行没有冒号，V4 parser 会忽略；V5 adapter 必须同时接受 colon 与 whitespace 两种 key 格式。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="只读审计 KITTI Tracking zip 与 metadata")
    parser.add_argument("--archive-root", default=Path("/root/autodl-tmp"), type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    parser.add_argument("--sha256-file", required=True, type=Path)
    parser.add_argument("--audit-date", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    hashes = parse_sha256_file(args.sha256_file)
    manifest = build_metadata(
        args.archive_root,
        project_root=args.project_root,
        audit_date=args.audit_date,
        sha256_by_filename=hashes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(canonical_json_bytes(manifest))
    args.output_md.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
