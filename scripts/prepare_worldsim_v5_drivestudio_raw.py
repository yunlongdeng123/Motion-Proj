#!/usr/bin/env python
"""为 V5 development cohort 精确准备 DriveStudio 所需的 nuScenes 原始载荷。

脚本只读取已冻结的 scene identity 与官方 metadata，不读取图像质量或实验结果。
默认仅输出 readiness plan；显式传入 ``--extract`` 才会扫描公共 blob shard，并且
只落盘六路相机与 LIDAR_TOP 的目标成员。输出包含逐 scene 内容哈希、member→shard
映射以及可复核的 batch manifest。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import ijson
import yaml


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META_ROOT = Path("/root/autodl-tmp/data/worldsim_v4/nuscenes_meta")
DEFAULT_RAW_ROOT = Path("/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval")
DEFAULT_MANIFEST_DIR = Path("/root/autodl-tmp/data/worldsim_v5/manifests")
DEFAULT_TAR_DIR = Path("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval")
DEFAULT_REUSE_ROOTS = (
    Path("/root/autodl-tmp/data/nuscenes"),
    Path("/root/autodl-tmp/data/dynamic_recon/raw_subset/adgs_nuscenes_v1"),
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_development_scenes(config_path: Path) -> list[dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    binding = config.get("fresh_cohort_binding", {})
    rows = binding.get("development_scenes", [])
    result: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_indices: set[int] = set()
    for row in rows:
        name = str(row["scene"])
        index = int(row["scene_index"])
        if name in seen_names or index in seen_indices:
            raise RuntimeError(f"development scene identity 重复: {name}/{index}")
        seen_names.add(name)
        seen_indices.add(index)
        result.append({"scene_name": name, "scene_index": index})
    if len(result) != 8:
        raise RuntimeError(f"V5 development cohort 必须为 8 scenes，实际 {len(result)}")
    if binding.get("validation_quality_read") is not False:
        raise RuntimeError("validation_quality_read 必须保持 false")
    if binding.get("test_quality_read") is not False:
        raise RuntimeError("test_quality_read 必须保持 false")
    return result


def normalize_shard_name(value: Any) -> str | None:
    text = str(value)
    if text.isdigit() and len(text) <= 2:
        return f"v1.0-trainval{int(text):02d}_blobs.tgz"
    if text.startswith("v1.0-trainval") and text.endswith("_blobs.tgz"):
        return text
    return None


def iter_seed_mappings(paths: Iterable[Path]) -> Iterable[tuple[str, str]]:
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            for name, shard in payload.items():
                normalized = normalize_shard_name(shard)
                if isinstance(name, str) and normalized is not None:
                    yield name, normalized
            files = payload.get("files", [])
            if isinstance(files, list):
                for row in files:
                    if not isinstance(row, dict):
                        continue
                    normalized = normalize_shard_name(row.get("shard"))
                    name = row.get("filename")
                    if isinstance(name, str) and normalized is not None:
                        yield name, normalized


def seed_index(
    index_path: Path, required: set[str], seed_paths: Iterable[Path]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if index_path.is_file():
        for name, shard in iter_seed_mappings([index_path]):
            if name in required:
                mapping[name] = shard
    for name, shard in iter_seed_mappings(seed_paths):
        if name not in required:
            continue
        previous = mapping.get(name)
        if previous is not None and previous != shard:
            raise RuntimeError(f"member shard 冲突: {name}: {previous} != {shard}")
        mapping[name] = shard
    atomic_json(index_path, {name: mapping[name] for name in sorted(mapping)})
    return mapping


def collect_batch(preparer: Any, metadata: Path, scenes: list[dict[str, Any]]) -> dict:
    """单次流式扫描 sample_data，为全部冻结 development scenes 建清单。"""
    scene_by_name = {
        row["name"]: row
        for row in json.loads((metadata / "scene.json").read_text(encoding="utf-8"))
    }
    sample_by_token = {
        row["token"]: row
        for row in json.loads((metadata / "sample.json").read_text(encoding="utf-8"))
    }
    sensor_by_token = {
        row["token"]: row
        for row in json.loads((metadata / "sensor.json").read_text(encoding="utf-8"))
    }
    channel_by_calibration = {
        row["token"]: sensor_by_token[row["sensor_token"]]["channel"]
        for row in json.loads(
            (metadata / "calibrated_sensor.json").read_text(encoding="utf-8")
        )
    }

    payloads: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    sample_owner: dict[str, str] = {}
    for identity in scenes:
        scene_name = identity["scene_name"]
        scene = scene_by_name.get(scene_name)
        if scene is None:
            raise RuntimeError(f"metadata 中缺少 scene: {scene_name}")
        tokens: set[str] = set()
        token = scene["first_sample_token"]
        while token:
            if token in tokens or token in sample_owner:
                raise RuntimeError(f"sample chain 重复或跨 scene 重叠: {token}")
            sample = sample_by_token[token]
            if sample["scene_token"] != scene["token"]:
                raise RuntimeError(f"sample chain 逃离 scene: {scene_name}/{token}")
            tokens.add(token)
            sample_owner[token] = scene_name
            if token == scene["last_sample_token"]:
                break
            token = sample["next"]
        if token != scene["last_sample_token"]:
            raise RuntimeError(f"sample chain 未到达末尾: {scene_name}")
        payload = {
            "scene_name": scene_name,
            "scene_index": identity["scene_index"],
            "scene_token": scene["token"],
            "first_sample_token": scene["first_sample_token"],
            "last_sample_token": scene["last_sample_token"],
            "sample_count": len(tokens),
            "sample_data": [],
        }
        payloads.append(payload)
        payload_by_name[scene_name] = payload

    seen_sd_tokens: set[str] = set()
    seen_filenames: set[str] = set()
    with (metadata / "sample_data.json").open("rb") as handle:
        for row in ijson.items(handle, "item"):
            scene_name = sample_owner.get(row["sample_token"])
            channel = channel_by_calibration.get(row["calibrated_sensor_token"])
            if scene_name is None or channel not in preparer.SENSORS:
                continue
            token, filename = row["token"], row["filename"]
            if token in seen_sd_tokens or filename in seen_filenames:
                raise RuntimeError(f"sample_data identity 重复: {token}/{filename}")
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"不安全的 sensor filename: {filename}")
            seen_sd_tokens.add(token)
            seen_filenames.add(filename)
            payload_by_name[scene_name]["sample_data"].append(
                {
                    "token": token,
                    "sample_token": row["sample_token"],
                    "channel": channel,
                    "timestamp": int(row["timestamp"]),
                    "filename": filename,
                    "is_key_frame": bool(row["is_key_frame"]),
                }
            )

    for payload in payloads:
        rows = payload["sample_data"]
        rows.sort(key=lambda item: (item["channel"], item["timestamp"], item["token"]))
        counts = {
            channel: sum(row["channel"] == channel for row in rows)
            for channel in preparer.SENSORS
        }
        if any(count == 0 for count in counts.values()):
            raise RuntimeError(
                f"scene sensor chain 为空: {payload['scene_name']}: {counts}"
            )
        payload["sensor_counts"] = counts
    return {
        "scene_payloads": payloads,
        "required": set(seen_filenames),
    }


def hash_rows(
    raw_root: Path,
    filenames: list[str],
    mapping: dict[str, str],
    sources: dict[str, str],
    workers: int,
) -> list[dict[str, Any]]:
    def inspect(name: str) -> dict[str, Any]:
        path = raw_root / name
        if not (path.is_file() and path.stat().st_size > 0):
            raise RuntimeError(f"原始载荷缺失或为空: {path}")
        return {
            "filename": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "shard": mapping[name],
            "source": sources.get(name, "existing_raw"),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(inspect, sorted(filenames)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROJECT_ROOT
        / "configs/worldsim_v5/m1_structured_ownership_v1.yaml",
    )
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--tar-dir", type=Path, default=DEFAULT_TAR_DIR)
    parser.add_argument("--reuse-root", action="append", type=Path, default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="执行选择性 shard 扫描与抽取；缺省只输出 metadata readiness plan",
    )
    args = parser.parse_args()

    preparer = load_module(
        DEFAULT_PROJECT_ROOT / "scripts/prepare_dr_v2_drivestudio_scene.py",
        "worldsim_v5_scene_preparer",
    )
    helpers = load_module(
        DEFAULT_PROJECT_ROOT / "scripts/build_adgs_nuscenes_assets.py",
        "worldsim_v5_asset_helpers",
    )
    scenes = load_development_scenes(args.config)
    metadata = args.meta_root / "v1.0-trainval"
    batch = collect_batch(preparer, metadata, scenes)
    required: set[str] = batch["required"]
    scene_payloads: list[dict[str, Any]] = batch["scene_payloads"]
    present_before = {
        name
        for name in required
        if (args.raw_root / name).is_file()
        and (args.raw_root / name).stat().st_size > 0
    }
    plan = {
        "schema_version": "worldsim_v5_m1_raw_plan_v1",
        "identity_source": str(args.config),
        "identity_source_sha256": sha256_file(args.config),
        "quality_read": False,
        "scene_count": len(scenes),
        "required_count": len(required),
        "present_before_count": len(present_before),
        "missing_before_count": len(required - present_before),
        "raw_root": str(args.raw_root),
        "sensor_counts": {
            payload["scene_name"]: payload["sensor_counts"]
            for payload in scene_payloads
        },
    }
    print(json.dumps({"readiness_plan": plan}, ensure_ascii=False, indent=2), flush=True)
    if not args.extract:
        return

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    helpers.link_metadata(metadata, args.raw_root)
    auxiliary = helpers.link_auxiliary_files(args.meta_root, args.raw_root)
    sources = {name: "existing_raw" for name in present_before}
    for root in (*DEFAULT_REUSE_ROOTS, *args.reuse_root):
        for name in helpers.link_existing_files(root, args.raw_root, required):
            sources[name] = f"reused:{root}"

    index_path = args.manifest_dir / "m1_development_member_shards.json"
    seed_paths = sorted(
        (Path("/root/autodl-tmp/data/worldsim_v4/manifests")).glob("*.json")
    )
    seed_paths += sorted(
        (Path("/root/autodl-tmp/data/dynamic_editing_v2/manifests")).glob("*.json")
    )
    seed_paths += sorted(
        (Path("/root/autodl-tmp/data/dynamic_recon/manifests")).glob("*.json")
    )
    seeded = seed_index(index_path, required, seed_paths)
    print(f"[index] audited seed coverage={len(seeded)}/{len(required)}", flush=True)
    mapping, extracted = helpers.scan_shards(
        tar_dir=args.tar_dir,
        members=required,
        index_path=index_path,
        dst=args.raw_root,
        workers=args.workers,
    )
    for name in extracted:
        sources[name] = "public_tar"

    scene_manifests = []
    total_bytes = 0
    for payload in scene_payloads:
        filenames = [row["filename"] for row in payload["sample_data"]]
        files = hash_rows(
            args.raw_root, filenames, mapping, sources, args.hash_workers
        )
        manifest = {
            "schema_version": "worldsim_v5_m1_scene_raw_v1",
            "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
            "identity_source": str(args.config),
            "quality_read": False,
            "scene_name": payload["scene_name"],
            "scene_index": payload["scene_index"],
            "scene_token": payload["scene_token"],
            "sample_count": payload["sample_count"],
            "sensor_counts": payload["sensor_counts"],
            "required_count": len(files),
            "present_count": len(files),
            "complete": True,
            "raw_root": str(args.raw_root),
            "metadata_root": str(metadata),
            "tar_dir": str(args.tar_dir),
            "sample_data": payload["sample_data"],
            "files": files,
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        path = args.manifest_dir / f"{payload['scene_name']}_raw_manifest_v5.json"
        atomic_json(path, manifest)
        byte_count = sum(row["bytes"] for row in files)
        total_bytes += byte_count
        scene_manifests.append(
            {
                "scene_name": payload["scene_name"],
                "scene_index": payload["scene_index"],
                "manifest": str(path),
                "manifest_sha256": manifest["manifest_sha256"],
                "required_count": len(files),
                "bytes": byte_count,
            }
        )
        print(
            f"[manifest] {payload['scene_name']}: files={len(files)} bytes={byte_count}",
            flush=True,
        )

    batch_manifest = {
        "schema_version": "worldsim_v5_m1_raw_batch_v1",
        "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
        "identity_source": str(args.config),
        "identity_source_sha256": sha256_file(args.config),
        "quality_read": False,
        "raw_root": str(args.raw_root),
        "metadata_root": str(metadata),
        "member_shard_index": str(index_path),
        "member_shard_index_sha256": sha256_file(index_path),
        "scene_count": len(scene_manifests),
        "required_count": len(required),
        "present_count": sum(
            (args.raw_root / name).is_file()
            and (args.raw_root / name).stat().st_size > 0
            for name in required
        ),
        "extracted_count": len(extracted),
        "total_bytes": total_bytes,
        "auxiliary_files": auxiliary,
        "scenes": scene_manifests,
        "complete": True,
    }
    batch_manifest["manifest_sha256"] = sha256_json(batch_manifest)
    batch_path = args.manifest_dir / "m1_development_raw_batch_v1.json"
    atomic_json(batch_path, batch_manifest)
    print(
        json.dumps(
            {
                "status": "done",
                "manifest": str(batch_path),
                "manifest_sha256": batch_manifest["manifest_sha256"],
                "required_count": len(required),
                "total_bytes": total_bytes,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
