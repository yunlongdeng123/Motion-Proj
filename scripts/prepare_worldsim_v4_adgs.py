#!/usr/bin/env python3
"""把冻结的 DriveStudio nuScenes 资产转换为 V4 matched AD-GS 输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
FRAME_COUNT = 196
CAMERAS = (0, 1, 2)
TARGET_SIZE = (800, 450)
PARTITION_MODULUS = 5
DEVELOPMENT_REMAINDER = 2
HELDOUT_REMAINDER = 4
TRAIN_REMAINDERS = (0, 1, 3)


class ADGSAdapterError(RuntimeError):
    """AD-GS 数据 adapter 合同失败。"""


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def partition_name(timestep: int) -> str:
    remainder = timestep % PARTITION_MODULUS
    if remainder == DEVELOPMENT_REMAINDER:
        return "development"
    if remainder == HELDOUT_REMAINDER:
        return "heldout"
    if remainder in TRAIN_REMAINDERS:
        return "train"
    raise ADGSAdapterError(f"未覆盖的 partition remainder: {remainder}")


def build_partition_flags(
    frame_count: int = FRAME_COUNT,
    cameras: Sequence[int] = CAMERAS,
    include_partitions: Sequence[str] = ("train", "development", "heldout"),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected = frozenset(include_partitions)
    valid = frozenset(("train", "development", "heldout"))
    if not selected or not selected <= valid or "train" not in selected:
        raise ADGSAdapterError("include_partitions 必须是包含 train 的非空合法子集")
    flags: list[bool] = []
    rows: list[dict[str, Any]] = []
    image_id = 0
    for timestep in range(frame_count):
        split = partition_name(timestep)
        if split not in selected:
            continue
        for camera in cameras:
            flags.append(split != "train")
            rows.append(
                {
                    "image_id": image_id,
                    "timestep": timestep,
                    "camera": int(camera),
                    "partition": split,
                }
            )
            image_id += 1
    return np.asarray(flags, dtype=np.bool_), rows


def scaled_intrinsic(values: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    if values.shape[0] < 4:
        raise ADGSAdapterError("intrinsics 至少需要 fx/fy/cx/cy")
    width, height = target_size
    scale_x = width / 1600.0
    scale_y = height / 900.0
    return np.asarray(
        [
            [values[0] * scale_x, 0.0, values[2] * scale_x],
            [0.0, values[1] * scale_y, values[3] * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def aligned_world_to_camera(camera_to_world: np.ndarray, origin: np.ndarray) -> np.ndarray:
    if camera_to_world.shape != (4, 4) or origin.shape != (4, 4):
        raise ADGSAdapterError("camera extrinsic 必须为 4x4")
    aligned_camera_to_world = np.linalg.inv(origin) @ camera_to_world
    return np.linalg.inv(aligned_camera_to_world).astype(np.float32)


def project_visible_colors(
    points_world: np.ndarray,
    intrinsics: Sequence[np.ndarray],
    world_to_cameras: Sequence[np.ndarray],
    images: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(intrinsics) == len(world_to_cameras) == len(images)):
        raise ADGSAdapterError("投影输入相机数不一致")
    color_sum = np.zeros((points_world.shape[0], 3), dtype=np.float32)
    color_count = np.zeros(points_world.shape[0], dtype=np.int32)
    homogeneous = np.concatenate(
        [points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    for intrinsic, world_to_camera, image in zip(
        intrinsics, world_to_cameras, images
    ):
        camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
        depth = camera_points[:, 2]
        projected = (intrinsic @ camera_points.T).T
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
        height, width = image.shape[:2]
        visible = (
            (depth > 0.0)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
        )
        indices = np.flatnonzero(visible)
        if not len(indices):
            continue
        pixels = np.rint(uv[indices]).astype(np.int64)
        color_sum[indices] += image[pixels[:, 1], pixels[:, 0], :3].astype(
            np.float32
        )
        color_count[indices] += 1
    visible_any = color_count > 0
    colors = np.zeros_like(color_sum, dtype=np.uint8)
    colors[visible_any] = np.clip(
        color_sum[visible_any] / color_count[visible_any, None], 0, 255
    ).astype(np.uint8)
    return visible_any, colors


def store_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("t", "f4"),
    ]
    normals = np.zeros_like(xyz, dtype=np.float32)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate(
        [
            xyz.astype(np.float32),
            normals,
            rgb.astype(np.uint8),
            timestamps.astype(np.float32).reshape(-1, 1),
        ],
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def expected_source_paths(
    source: Path,
    frame_count: int,
    cameras: Sequence[int],
) -> Iterable[Path]:
    for camera in cameras:
        yield source / "intrinsics" / f"{camera}.txt"
    for timestep in range(frame_count):
        yield source / "lidar" / f"{timestep:03d}.bin"
        yield source / "lidar_pose" / f"{timestep:03d}.txt"
        for camera in cameras:
            stem = f"{timestep:03d}_{camera}"
            yield source / "images" / f"{stem}.jpg"
            yield source / "extrinsics" / f"{stem}.txt"
            yield source / "sky_masks" / f"{stem}.png"
            yield source / "dynamic_masks" / "all" / f"{stem}.png"


def prepare_scene(
    source: Path,
    destination: Path,
    frame_count: int = FRAME_COUNT,
    cameras: Sequence[int] = CAMERAS,
    target_size: tuple[int, int] = TARGET_SIZE,
    include_partitions: Sequence[str] = ("train", "development", "heldout"),
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise ADGSAdapterError("目标或 partial 已存在，禁止覆盖")
    missing = [str(path) for path in expected_source_paths(source, frame_count, cameras) if not path.is_file()]
    if missing:
        raise ADGSAdapterError(f"源数据缺失 {len(missing)} 项，首项: {missing[0]}")

    for folder in ("image", "semantic", "sky", "depth", "flow"):
        (partial / folder).mkdir(parents=True, exist_ok=False)

    origin = np.loadtxt(source / "extrinsics" / "000_0.txt")
    camera_intrinsics = {
        camera: scaled_intrinsic(
            np.loadtxt(source / "intrinsics" / f"{camera}.txt"),
            target_size,
        )
        for camera in cameras
    }
    selected_partitions = frozenset(include_partitions)
    flags, partition_rows = build_partition_flags(
        frame_count, cameras, include_partitions
    )
    intrinsics_rows: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    timestamps: list[float] = []
    pcd_xyz: list[np.ndarray] = []
    pcd_rgb: list[np.ndarray] = []
    pcd_time: list[np.ndarray] = []
    image_id = 0

    for timestep in range(frame_count):
        split = partition_name(timestep)
        if split != "train" and split not in selected_partitions:
            continue
        frame_images: list[np.ndarray] = []
        frame_w2c: list[np.ndarray] = []
        frame_intrinsics: list[np.ndarray] = []
        for camera in cameras:
            stem = f"{timestep:03d}_{camera}"
            image = Image.open(source / "images" / f"{stem}.jpg").convert("RGB")
            if image.size != (1600, 900):
                raise ADGSAdapterError(f"图像分辨率漂移: {stem}={image.size}")
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)

            camera_to_world = np.loadtxt(
                source / "extrinsics" / f"{stem}.txt"
            )
            world_to_camera = aligned_world_to_camera(camera_to_world, origin)
            intrinsic = camera_intrinsics[camera]
            if split in selected_partitions:
                image.save(partial / "image" / f"{image_id:06d}.png")
                semantic = Image.open(
                    source / "dynamic_masks" / "all" / f"{stem}.png"
                ).convert("L").resize(target_size, Image.Resampling.NEAREST)
                sky = Image.open(source / "sky_masks" / f"{stem}.png").convert(
                    "L"
                ).resize(target_size, Image.Resampling.NEAREST)
                np.save(
                    partial / "semantic" / f"mask_{image_id:06d}.npy",
                    (np.asarray(semantic) > 0).astype(np.uint8),
                )
                np.save(
                    partial / "sky" / f"mask_{image_id:06d}.npy",
                    (np.asarray(sky) > 0).astype(np.uint8),
                )
                intrinsics_rows.append(intrinsic)
                rotations.append(world_to_camera[:3, :3])
                translations.append(world_to_camera[:3, 3])
                timestamps.append(float(timestep))
                image_id += 1
            frame_images.append(image_array)
            frame_w2c.append(world_to_camera)
            frame_intrinsics.append(intrinsic)

        if split != "train":
            continue
        lidar = np.fromfile(
            source / "lidar" / f"{timestep:03d}.bin", dtype=np.float32
        ).reshape(-1, 4)[:, :3]
        lidar_to_world = np.loadtxt(
            source / "lidar_pose" / f"{timestep:03d}.txt"
        )
        aligned_lidar_to_world = np.linalg.inv(origin) @ lidar_to_world
        points_world = (
            aligned_lidar_to_world[:3, :3] @ lidar.T
            + aligned_lidar_to_world[:3, 3:4]
        ).T.astype(np.float32)
        visible, colors = project_visible_colors(
            points_world, frame_intrinsics, frame_w2c, frame_images
        )
        pcd_xyz.append(points_world[visible])
        pcd_rgb.append(colors[visible])
        pcd_time.append(
            np.full(int(visible.sum()), float(timestep), dtype=np.float32)
        )

    xyz = np.concatenate(pcd_xyz, axis=0)
    rgb = np.concatenate(pcd_rgb, axis=0)
    point_times = np.concatenate(pcd_time, axis=0)
    store_ply(partial / "points3d.ply", xyz, rgb, point_times)
    np.savez(
        partial / "meta.npz",
        R=np.stack(rotations).astype(np.float32),
        T=np.stack(translations).astype(np.float32),
        K=np.stack(intrinsics_rows).astype(np.float32),
        time_stamps=np.asarray(timestamps, dtype=np.float32),
        is_val_list=flags,
    )
    write_json(
        partial / "partition.json",
        {
            "schema_version": "worldsim_v4_adgs_partition_v1",
            "task_id": TASK_ID,
            "modulus": PARTITION_MODULUS,
            "development_remainder": DEVELOPMENT_REMAINDER,
            "heldout_remainder": HELDOUT_REMAINDER,
            "train_remainders": list(TRAIN_REMAINDERS),
            "included_partitions": sorted(selected_partitions),
            "rows": partition_rows,
        },
    )
    counts = {
        split: sum(row["partition"] == split for row in partition_rows)
        for split in ("train", "development", "heldout")
    }
    manifest = {
        "schema_version": "worldsim_v4_adgs_adapter_manifest_v1",
        "task_id": TASK_ID,
        "source": str(source),
        "destination": str(destination),
        "frame_count": frame_count,
        "cameras": list(cameras),
        "target_size": list(target_size),
        "included_partitions": sorted(selected_partitions),
        "image_count": image_id,
        "partition_image_counts": counts,
        "point_count": int(xyz.shape[0]),
        "point_time_min": float(point_times.min()),
        "point_time_max": float(point_times.max()),
        "meta_sha256": sha256_file(partial / "meta.npz"),
        "points3d_sha256": sha256_file(partial / "points3d.ply"),
        "partition_sha256": sha256_file(partial / "partition.json"),
    }
    write_json(partial / "adapter_manifest.json", manifest)
    os.replace(partial, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "development", "heldout"),
        default=("train", "development", "heldout"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_scene(
                args.source,
                args.destination,
                include_partitions=args.partitions,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
