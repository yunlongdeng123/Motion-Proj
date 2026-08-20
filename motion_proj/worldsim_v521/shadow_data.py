"""构造只允许解码 Discovery 的 V5.2.1 renderer shadow data。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .census import CensusError, sha256_file, validate_discovery_record


FRAME_COUNT = 196
CAMERAS = (0, 1, 2)
TRAIN_REMAINDERS = (0, 1, 3)
METRIC_SIZE = (800, 450)
SOURCE_SIZE = (1600, 900)


def _symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _blank_assets(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rgb = root / "black_1600x900.jpg"
    mask = root / "zero_1600x900.png"
    if not rgb.is_file():
        Image.new("RGB", SOURCE_SIZE, (0, 0, 0)).save(rgb, quality=95, subsampling=0)
    if not mask.is_file():
        Image.new("L", SOURCE_SIZE, 0).save(mask)
    return {"rgb": rgb, "mask": mask}


def build_streetgs_shadow(
    *,
    source_scene: str | Path,
    shadow_root: str | Path,
    scene_index: int,
    discovery_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source = Path(source_scene).resolve()
    destination_root = Path(shadow_root).resolve()
    destination = destination_root / f"{int(scene_index):03d}"
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise CensusError(f"shadow 目标已存在：{destination}")
    for row in discovery_records:
        validate_discovery_record(row)
    discovery_frames = {int(row["frame"]) for row in discovery_records}
    if len(discovery_records) != len(discovery_frames) * len(CAMERAS):
        raise CensusError("Discovery scene registry 必须覆盖每个 sample 的三相机")
    destination_root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    skip = {"images", "dynamic_masks", "fine_dynamic_masks", "sky_masks"}
    for child in source.iterdir():
        if child.name not in skip:
            _symlink(child, partial / child.name)
    (partial / "images").mkdir()
    for category in ("all", "human", "vehicle"):
        (partial / "dynamic_masks" / category).mkdir(parents=True, exist_ok=True)
    (partial / "sky_masks").mkdir()
    blanks = _blank_assets(destination_root / ".v521_placeholders")
    linked_original = 0
    linked_placeholder = 0
    for frame in range(FRAME_COUNT):
        allow_pixels = frame % 5 in TRAIN_REMAINDERS or frame in discovery_frames
        for camera in CAMERAS:
            stem = f"{frame:03d}_{camera}"
            image_source = source / "images" / f"{stem}.jpg" if allow_pixels else blanks["rgb"]
            _symlink(image_source, partial / "images" / f"{stem}.jpg")
            for category in ("all", "human", "vehicle"):
                candidate = source / "dynamic_masks" / category / f"{stem}.png"
                mask_source = candidate if allow_pixels else blanks["mask"]
                if allow_pixels and not candidate.is_file():
                    raise CensusError(f"dynamic mask 缺失：{candidate}")
                _symlink(mask_source, partial / "dynamic_masks" / category / f"{stem}.png")
            sky_candidate = source / "sky_masks" / f"{stem}.png"
            sky_source = sky_candidate if allow_pixels else blanks["mask"]
            if allow_pixels and not sky_candidate.is_file():
                raise CensusError(f"sky mask 缺失：{sky_candidate}")
            _symlink(sky_source, partial / "sky_masks" / f"{stem}.png")
            linked_original += int(allow_pixels) * 5
            linked_placeholder += int(not allow_pixels) * 5
    manifest = {
        "schema": "worldsim_v521_streetgs_shadow_v1",
        "scene_index": int(scene_index),
        "source": str(source),
        "destination": str(destination),
        "discovery_frames": sorted(discovery_frames),
        "original_pixel_links": linked_original,
        "placeholder_pixel_links": linked_placeholder,
        "confirmation_original_pixel_links": 0,
        "heldout_original_pixel_links": 0,
        "quality_lock": "train_plus_discovery_only",
    }
    _atomic_json(partial / "V521_SHADOW_MANIFEST.json", manifest)
    os.replace(partial, destination)
    return manifest


def _scaled_intrinsic(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[values[0] * 0.5, 0.0, values[2] * 0.5], [0.0, values[1] * 0.5, values[3] * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _aligned_world_to_camera(camera_to_world: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.linalg.inv(origin) @ camera_to_world).astype(np.float32)


def build_adgs_discovery_adapter(
    *,
    train_adapter: str | Path,
    source_scene: str | Path,
    destination: str | Path,
    discovery_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train = Path(train_adapter).resolve()
    source = Path(source_scene).resolve()
    destination = Path(destination).resolve()
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise CensusError(f"AD-GS adapter 目标已存在：{destination}")
    rows = sorted(discovery_records, key=lambda row: (int(row["frame"]), int(row["camera"])))
    for row in rows:
        validate_discovery_record(row)
    partial.mkdir(parents=True)
    for folder in ("image", "semantic", "sky", "depth", "flow"):
        (partial / folder).mkdir()
    for folder in ("image", "semantic", "sky", "depth", "flow"):
        source_folder = train / folder
        for child in source_folder.iterdir():
            _symlink(child, partial / folder / child.name)
    _symlink(train / "points3d.ply", partial / "points3d.ply")
    meta = np.load(train / "meta.npz", allow_pickle=False)
    rotations = [value for value in meta["R"]]
    translations = [value for value in meta["T"]]
    intrinsics = [value for value in meta["K"]]
    timestamps = [float(value) for value in meta["time_stamps"]]
    flags = [bool(value) for value in meta["is_val_list"]]
    train_image_count = len(flags)
    if any(flags):
        raise CensusError("train-only AD-GS adapter 含 validation 行")
    zero_depth = partial / "depth" / ".v521_zero_depth.npy"
    np.save(zero_depth, np.zeros((METRIC_SIZE[1], METRIC_SIZE[0], 1), dtype=np.float32))
    origin = np.loadtxt(source / "extrinsics" / "000_0.txt")
    camera_intrinsics = {
        camera: _scaled_intrinsic(np.loadtxt(source / "intrinsics" / f"{camera}.txt"))
        for camera in CAMERAS
    }
    render_map: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        image_id = train_image_count + offset
        frame, camera = int(row["frame"]), int(row["camera"])
        stem = f"{frame:03d}_{camera}"
        with Image.open(source / "images" / f"{stem}.jpg") as opened:
            target = opened.convert("RGB").resize(METRIC_SIZE, Image.Resampling.LANCZOS)
            target.save(partial / "image" / f"{image_id:06d}.png")
        with Image.open(source / "dynamic_masks" / "all" / f"{stem}.png") as opened:
            dynamic = opened.convert("L").resize(METRIC_SIZE, Image.Resampling.NEAREST)
            np.save(partial / "semantic" / f"mask_{image_id:06d}.npy", (np.asarray(dynamic) > 0).astype(np.uint8))
        with Image.open(source / "sky_masks" / f"{stem}.png") as opened:
            sky = opened.convert("L").resize(METRIC_SIZE, Image.Resampling.NEAREST)
            np.save(partial / "sky" / f"mask_{image_id:06d}.npy", (np.asarray(sky) > 0).astype(np.uint8))
        _symlink(zero_depth, partial / "depth" / f"{image_id:06d}.npy")
        world_to_camera = _aligned_world_to_camera(np.loadtxt(source / "extrinsics" / f"{stem}.txt"), origin)
        rotations.append(world_to_camera[:3, :3])
        translations.append(world_to_camera[:3, 3])
        intrinsics.append(camera_intrinsics[camera])
        timestamps.append(float(frame))
        flags.append(True)
        render_map.append(
            {
                "render_index": offset,
                "image_id": image_id,
                "scene": row["scene"],
                "frame": frame,
                "camera": camera,
                "partition": "discovery",
                "target_source_sha256": row["target_sha256"],
            }
        )
    np.savez(
        partial / "meta.npz",
        R=np.stack(rotations).astype(np.float32),
        T=np.stack(translations).astype(np.float32),
        K=np.stack(intrinsics).astype(np.float32),
        time_stamps=np.asarray(timestamps, dtype=np.float32),
        is_val_list=np.asarray(flags, dtype=np.bool_),
    )
    manifest = {
        "schema": "worldsim_v521_adgs_discovery_adapter_v1",
        "source_train_adapter": str(train),
        "source_scene": str(source),
        "destination": str(destination),
        "train_image_count": train_image_count,
        "discovery_image_count": len(rows),
        "confirmation_image_count": 0,
        "meta_sha256": sha256_file(partial / "meta.npz"),
        "render_map": render_map,
        "quality_lock": "train_plus_discovery_only",
    }
    _atomic_json(partial / "V521_ADAPTER_MANIFEST.json", manifest)
    os.replace(partial, destination)
    return manifest
