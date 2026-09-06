#!/usr/bin/env python3
"""按 ZIP Range 请求获取 LiDAR4D 所需的 KITTI-360 单序列切片。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from remotezip import RemoteZip


CALIBRATION_URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
    "384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip"
)
POSES_URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
    "89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip"
)
TIMESTAMPS_URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
    "data_3d_raw/data_timestamps_velodyne.zip"
)
VELODYNE_URL_TEMPLATE = (
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
    "data_3d_raw/{sequence}_velodyne.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sequence", default="2013_05_28_drive_0000_sync"
    )
    parser.add_argument("--frame-start", type=int, default=4950)
    parser.add_argument("--frame-end", type=int, default=5000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    temporary.replace(destination)


def safe_extract(archive_path: Path, output_root: Path) -> None:
    root = output_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            destination = (output_root / info.filename).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError(f"ZIP 路径越界：{info.filename}")
        archive.extractall(output_root)


def fetch_scans(
    url: str,
    output_root: Path,
    sequence: str,
    frame_start: int,
    frame_end: int,
) -> list[dict[str, object]]:
    destination_root = output_root / "data_3d_raw" / sequence / "velodyne_points" / "data"
    destination_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with RemoteZip(url) as archive:
        available = set(archive.namelist())
        for frame_id in range(frame_start, frame_end + 1):
            member = f"{sequence}/velodyne_points/data/{frame_id:010d}.bin"
            if member not in available:
                raise FileNotFoundError(f"远程 ZIP 缺少 {member}")
            destination = destination_root / f"{frame_id:010d}.bin"
            if not destination.is_file():
                temporary = destination.with_suffix(".bin.part")
                with archive.open(member) as source, temporary.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                temporary.replace(destination)
            size = destination.stat().st_size
            if size == 0 or size % 16 != 0:
                raise ValueError(f"点云文件尺寸非法：{destination} ({size} bytes)")
            records.append(
                {
                    "frame_id": frame_id,
                    "path": str(destination.relative_to(output_root)),
                    "bytes": size,
                    "points": size // 16,
                    "sha256": sha256(destination),
                }
            )
    return records


def main() -> None:
    args = parse_args()
    if args.frame_end < args.frame_start:
        raise ValueError("frame-end 必须大于等于 frame-start")
    output_root = args.output_root.resolve()
    cache_root = output_root.parent / ".kitti360_download_cache"
    small_archives = {
        "calibration": (
            CALIBRATION_URL,
            cache_root / "calibration.zip",
            output_root,
        ),
        "poses": (
            POSES_URL,
            cache_root / "data_poses.zip",
            output_root / "data_poses",
        ),
        "timestamps": (
            TIMESTAMPS_URL,
            cache_root / "data_timestamps_velodyne.zip",
            output_root / "data_3d_raw",
        ),
    }
    archive_records = {}
    for name, (url, path, extract_root) in small_archives.items():
        download(url, path)
        safe_extract(path, extract_root)
        archive_records[name] = {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "extract_root": str(extract_root),
        }

    velodyne_url = VELODYNE_URL_TEMPLATE.format(sequence=args.sequence)
    scans = fetch_scans(
        velodyne_url,
        output_root,
        args.sequence,
        args.frame_start,
        args.frame_end,
    )
    manifest = {
        "schema_version": "worldsim_v72_kitti360_slice_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "sequence": args.sequence,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "frame_count": len(scans),
        "velodyne_url": velodyne_url,
        "small_archives": archive_records,
        "scans": scans,
    }
    manifest_path = output_root.parent / (
        f"kitti360_{args.sequence}_{args.frame_start}_{args.frame_end}_manifest.json"
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "frame_count": len(scans),
                "scan_bytes": sum(int(item["bytes"]) for item in scans),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
