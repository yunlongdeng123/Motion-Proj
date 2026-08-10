"""S1 sidecar 的纯 NumPy schema、分类与一致性校验。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


NEGATIVE = np.int8(0)
CORE_POSITIVE = np.int8(1)
SEMANTIC_POSITIVE = np.int8(2)
AMBIGUOUS = np.int8(3)

LABEL_NAMES = {
    int(NEGATIVE): "NEGATIVE",
    int(CORE_POSITIVE): "CORE_POSITIVE",
    int(SEMANTIC_POSITIVE): "SEMANTIC_POSITIVE",
    int(AMBIGUOUS): "AMBIGUOUS",
}


def validate_actor_identity_contract(
    *,
    role: str,
    actor_config: Mapping[str, Any],
    dataset_instance: Mapping[str, Any],
    registry_actor: Mapping[str, Any],
) -> None:
    """冻结 dataset ID、instance token 与 D2 rigid index 的一一对应。"""
    expected_token = str(actor_config["instance_token"])
    dataset_token = str(dataset_instance.get("id", ""))
    registry_token = str(registry_actor.get("instance_token", ""))
    if dataset_token != expected_token:
        raise ValueError(
            f"{role} dataset_instance_id/token 错配: "
            f"expected={expected_token} actual={dataset_token}"
        )
    if registry_token != expected_token:
        raise ValueError(
            f"{role} actor registry token 错配: "
            f"expected={expected_token} actual={registry_token}"
        )
    expected_index = int(actor_config["rigid_model_index"])
    actual_index = int(registry_actor["rigid_model_index"])
    if actual_index != expected_index:
        raise ValueError(
            f"{role} rigid_model_index 错配: "
            f"expected={expected_index} actual={actual_index}"
        )
    expected_class = str(actor_config["class_name"])
    for source_name, row in (
        ("instances_info", dataset_instance),
        ("actor_registry", registry_actor),
    ):
        actual_class = str(row.get("class_name", ""))
        if actual_class != expected_class:
            raise ValueError(
                f"{role} {source_name} class 错配: "
                f"expected={expected_class} actual={actual_class}"
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_disjoint_split(
    train_frames: list[int] | tuple[int, ...],
    heldout_frames: list[int] | tuple[int, ...],
) -> None:
    train = {int(value) for value in train_frames}
    heldout = {int(value) for value in heldout_frames}
    overlap = sorted(train & heldout)
    if overlap:
        raise ValueError(f"S1 train/heldout 泄漏: {overlap}")


def contiguous_blocks(frames: list[int] | tuple[int, ...]) -> list[list[int]]:
    ordered = sorted({int(value) for value in frames})
    if not ordered:
        return []
    blocks: list[list[int]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame == blocks[-1][-1] + 1:
            blocks[-1].append(frame)
        else:
            blocks.append([frame])
    return blocks


def box_corners(size_xyz: np.ndarray) -> np.ndarray:
    size = np.asarray(size_xyz, dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("3D box size 必须是有限正数 (3,)")
    half = size / 2.0
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * half


def project_box_prompt(
    *,
    obj_to_world: np.ndarray,
    box_size: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    minimum_depth_m: float = 0.1,
    padding_fraction: float = 0.05,
    minimum_side_pixels: float = 4.0,
) -> np.ndarray | None:
    """把 DriveStudio object box 投影成裁剪后的 SAM xyxy prompt。"""
    o2w = np.asarray(obj_to_world, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if o2w.shape != (4, 4) or c2w.shape != (4, 4) or k.shape != (3, 3):
        raise ValueError("投影矩阵 shape 不合法")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像尺寸必须为正")

    corners = box_corners(np.asarray(box_size, dtype=np.float64))
    homogeneous = np.concatenate([corners, np.ones((8, 1))], axis=1)
    world = (o2w @ homogeneous.T).T
    camera = (np.linalg.inv(c2w) @ world.T).T[:, :3]
    visible = camera[:, 2] > float(minimum_depth_m)
    if int(visible.sum()) < 2:
        return None
    projected = (k @ camera[visible].T).T
    xy = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(xy).all():
        return None
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    extent = hi - lo
    lo -= extent * float(padding_fraction)
    hi += extent * float(padding_fraction)
    lo = np.maximum(lo, [0.0, 0.0])
    hi = np.minimum(hi, [float(image_width - 1), float(image_height - 1)])
    if np.any((hi - lo) < float(minimum_side_pixels)):
        return None
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)


def semantic_posterior(
    semantic_mass: np.ndarray,
    visible_mass: np.ndarray,
) -> np.ndarray:
    semantic = np.asarray(semantic_mass, dtype=np.float64)
    visible = np.asarray(visible_mass, dtype=np.float64)
    if semantic.shape != visible.shape:
        raise ValueError("semantic/visible mass shape 不一致")
    if np.any(semantic < 0) or np.any(visible < 0):
        raise ValueError("贡献质量不能为负")
    result = np.zeros_like(semantic, dtype=np.float64)
    np.divide(semantic, visible, out=result, where=visible > 0)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def binary_inner_boundary(mask: np.ndarray) -> np.ndarray:
    """返回四邻域定义的前景内边界，不引入额外图像依赖。"""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("binary mask 必须是二维数组")
    padded = np.pad(value, 1, mode="constant", constant_values=False)
    interior = (
        value
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return value & ~interior


def classify_gaussians(
    *,
    posterior: np.ndarray,
    semantic_mass: np.ndarray,
    positive_view_count: np.ndarray,
    core_mask: np.ndarray,
    semantic_threshold: float,
    ambiguous_threshold: float,
    minimum_semantic_mass: float,
    minimum_positive_views: int,
) -> np.ndarray:
    posterior = np.asarray(posterior, dtype=np.float32)
    mass = np.asarray(semantic_mass, dtype=np.float64)
    views = np.asarray(positive_view_count, dtype=np.int64)
    core = np.asarray(core_mask, dtype=bool)
    if not (posterior.shape == mass.shape == views.shape == core.shape):
        raise ValueError("S1 分类数组 shape 不一致")
    if not 0 <= ambiguous_threshold <= semantic_threshold <= 1:
        raise ValueError("S1 posterior 阈值顺序不合法")
    labels = np.full(posterior.shape, NEGATIVE, dtype=np.int8)
    evidence = (mass >= minimum_semantic_mass) & (views >= minimum_positive_views)
    labels[(posterior >= ambiguous_threshold) & ~evidence] = AMBIGUOUS
    labels[(posterior >= ambiguous_threshold) & evidence] = AMBIGUOUS
    labels[(posterior >= semantic_threshold) & evidence] = SEMANTIC_POSITIVE
    labels[core] = CORE_POSITIVE
    return labels


def label_counts(labels: np.ndarray) -> Mapping[str, int]:
    values = np.asarray(labels, dtype=np.int8)
    unknown = sorted(set(int(value) for value in np.unique(values)) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"未知 S1 label: {unknown}")
    return {
        name: int((values == numeric).sum())
        for numeric, name in LABEL_NAMES.items()
    }
