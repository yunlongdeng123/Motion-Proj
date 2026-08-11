#!/usr/bin/env python
"""Prepare the exact raw nuScenes payload required by a DriveStudio scene.

The public nuScenes trainval blobs are split across ten tar archives.  This
utility derives the required sensor filenames from frozen metadata, reuses
already-present payloads by hard link, scans only for the remaining members,
and emits a content-addressed manifest.  It never mutates the source dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import ijson


SENSORS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "LIDAR_TOP",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def iter_json_array(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def scene_sample_tokens(metadata: Path, scene_name: str) -> tuple[dict, set[str]]:
    scenes = load_json(metadata / "scene.json")
    matches = [row for row in scenes if row["name"] == scene_name]
    if len(matches) != 1:
        raise RuntimeError(f"scene {scene_name} must resolve once, got {len(matches)}")
    scene = matches[0]
    sample_by_token = {row["token"]: row for row in load_json(metadata / "sample.json")}
    tokens: set[str] = set()
    token = scene["first_sample_token"]
    while token:
        if token in tokens:
            raise RuntimeError(f"sample chain cycle at {token}")
        row = sample_by_token[token]
        if row["scene_token"] != scene["token"]:
            raise RuntimeError(f"sample {token} escaped scene {scene_name}")
        tokens.add(token)
        if token == scene["last_sample_token"]:
            break
        token = row["next"]
    if token != scene["last_sample_token"]:
        raise RuntimeError(f"sample chain did not reach last sample for {scene_name}")
    return scene, tokens


def collect_required(metadata: Path, scene_name: str) -> dict[str, Any]:
    scene, sample_tokens = scene_sample_tokens(metadata, scene_name)
    sensor_by_token = {row["token"]: row for row in load_json(metadata / "sensor.json")}
    channel_by_calibration = {
        row["token"]: sensor_by_token[row["sensor_token"]]["channel"]
        for row in load_json(metadata / "calibrated_sensor.json")
    }
    rows: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    seen_filenames: set[str] = set()
    for row in iter_json_array(metadata / "sample_data.json"):
        channel = channel_by_calibration.get(row["calibrated_sensor_token"])
        if row["sample_token"] not in sample_tokens or channel not in SENSORS:
            continue
        token, filename = row["token"], row["filename"]
        if token in seen_tokens or filename in seen_filenames:
            raise RuntimeError(f"duplicate sample_data identity: {token} {filename}")
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe sensor filename: {filename}")
        seen_tokens.add(token)
        seen_filenames.add(filename)
        rows.append(
            {
                "token": token,
                "sample_token": row["sample_token"],
                "channel": channel,
                "timestamp": int(row["timestamp"]),
                "filename": filename,
                "is_key_frame": bool(row["is_key_frame"]),
            }
        )
    rows.sort(key=lambda item: (item["channel"], item["timestamp"], item["token"]))
    counts = {channel: sum(row["channel"] == channel for row in rows) for channel in SENSORS}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"empty sensor chain for {scene_name}: {counts}")
    return {
        "scene_name": scene_name,
        "scene_token": scene["token"],
        "first_sample_token": scene["first_sample_token"],
        "last_sample_token": scene["last_sample_token"],
        "sample_count": len(sample_tokens),
        "sensor_counts": counts,
        "sample_data": rows,
    }


def load_asset_module(project_root: Path):
    script = project_root / "scripts" / "build_adgs_nuscenes_assets.py"
    spec = importlib.util.spec_from_file_location("dr_v2_asset_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", default="scene-0230")
    parser.add_argument("--scene-index", type=int, default=179)
    parser.add_argument("--meta-root", type=Path, default=Path("/root/autodl-tmp/data/nuscenes"))
    parser.add_argument(
        "--reuse-root",
        type=Path,
        action="append",
        default=[Path("/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1")],
    )
    parser.add_argument(
        "--tar-dir",
        type=Path,
        default=Path("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_raw_scene0230"),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/root/autodl-tmp/data/dynamic_editing_v2/manifests"),
    )
    parser.add_argument(
        "--anchor-index",
        type=Path,
        default=Path("/root/autodl-tmp/data/dynamic_recon/manifests/adgs_nuscenes_v1_member_shards.json"),
        help="Previously audited member→shard mappings used only when all scene anchors agree.",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    metadata = args.meta_root / "v1.0-trainval"
    payload = collect_required(metadata, args.scene_name)
    required = {row["filename"] for row in payload["sample_data"]}
    print(json.dumps({key: payload[key] for key in ("scene_name", "scene_token", "sample_count", "sensor_counts")}, sort_keys=True), flush=True)

    helpers = load_asset_module(project_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    helpers.link_metadata(metadata, args.out_root)
    auxiliary = helpers.link_auxiliary_files(args.meta_root, args.out_root)
    reuse_roots = [args.meta_root, *args.reuse_root]
    reused: dict[str, str] = {}
    for root in reuse_roots:
        linked = helpers.link_existing_files(root, args.out_root, required)
        for name in linked:
            reused[name] = str(root)

    anchor_evidence = {"index": str(args.anchor_index), "anchor_count": 0, "shards": []}
    if args.anchor_index.is_file():
        anchor_mapping = json.loads(args.anchor_index.read_text())
        anchors = {
            name: shard for name, shard in anchor_mapping.items() if name in required
        }
        anchor_shards = sorted(set(anchors.values()))
        anchor_evidence.update(
            {"anchor_count": len(anchors), "shards": anchor_shards}
        )
        if len(anchors) >= 50 and len(anchor_shards) == 1:
            shard_name = anchor_shards[0]
            prefix, suffix = "v1.0-trainval", "_blobs.tgz"
            if not (shard_name.startswith(prefix) and shard_name.endswith(suffix)):
                raise RuntimeError(f"invalid audited anchor shard name: {shard_name}")
            helpers.SHARDS = [shard_name[len(prefix):-len(suffix)]]
            anchor_evidence["scan_scope"] = "single_audited_scene_shard"
        else:
            anchor_evidence["scan_scope"] = "all_shards_fallback"

    index_path = args.manifest_dir / f"{args.scene_name}_member_shards.json"
    mapping, extracted = helpers.scan_shards(
        tar_dir=args.tar_dir,
        members=required,
        index_path=index_path,
        dst=args.out_root,
        workers=args.workers,
    )
    files = []
    for name in sorted(required):
        path = args.out_root / name
        if not (path.is_file() and path.stat().st_size > 0):
            raise RuntimeError(f"missing extracted member: {name}")
        files.append(
            {
                "filename": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shard": mapping[name],
                "source": "public_tar" if name in extracted else reused.get(name, "existing_output"),
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": f"DriveStudio/StreetGS {args.scene_name} native nuScenes input",
        "scene_name": args.scene_name,
        "scene_index": args.scene_index,
        "scene_token": payload["scene_token"],
        "sample_count": payload["sample_count"],
        "sensor_counts": payload["sensor_counts"],
        "raw_root": str(args.out_root),
        "metadata_root": str(metadata),
        "tar_dir": str(args.tar_dir),
        "workers": args.workers,
        "anchor_shard_evidence": anchor_evidence,
        "required_count": len(required),
        "present_count": len(files),
        "complete": len(files) == len(required),
        "metadata_files": sorted(path.name for path in metadata.iterdir() if path.is_file()),
        "auxiliary_files": auxiliary,
        "sample_data": payload["sample_data"],
        "files": files,
    }
    digest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    manifest_path = args.manifest_dir / f"{args.scene_name}_raw_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "done", "manifest": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"], "files": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
