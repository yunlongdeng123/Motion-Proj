"""RoadPatch-Lite 的确定性 3D donor 索引、搜索与空间 delta。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA_VERSION = "worldsim_v33_roadpatch_index_v1"
DELTA_SCHEMA_VERSION = "worldsim_v33_roadpatch_delta_v1"
PROVENANCE_GENERATED_BY_PATCH_REUSE = np.uint8(2)
SH_C0 = 0.28209479177387814

# scene-0230 的 DriveStudio world 以首帧前相机为原点，保持 OpenCV x-right、
# y-down、z-forward；因此 BEV 是 x/z，竖直轴是 y。不能沿用传统 z-up 假设。
BEV_AXES = (0, 2)
VERTICAL_AXIS = 1

FEATURE_NAMES = (
    "mean_vertical",
    "std_vertical",
    "plane_normal_x",
    "plane_normal_y",
    "plane_normal_z",
    "plane_residual",
    "gaussian_density",
    "mean_scale",
    "max_scale",
    "opacity_mean",
    "opacity_std",
    "vertical_range",
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "gradient_energy_proxy",
    "actor_semantic_mean",
    "actor_semantic_max",
    "background_semantic_mean",
    "train_view_observation_mean",
    "multi_camera_count_mean",
    "log_visibility_mass_mean",
    "tangent_cos",
    "tangent_sin",
    "tangent_confidence",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

EXCLUDE_SPARSE = np.uint16(1 << 0)
EXCLUDE_ACTOR_SEMANTIC = np.uint16(1 << 1)
EXCLUDE_LOW_VISIBILITY = np.uint16(1 << 2)
EXCLUDE_LOW_VIEW_SUPPORT = np.uint16(1 << 3)
EXCLUDE_GEOMETRY_OUTLIER = np.uint16(1 << 4)
EXCLUDE_SCALE_OUTLIER = np.uint16(1 << 5)
EXCLUDE_GENERATED = np.uint16(1 << 6)


@dataclass(frozen=True)
class PatchIndex:
    """紧凑保存 1/2/4 m donor patches 与其原生 Gaussian 行映射。"""

    patch_ids: np.ndarray
    patch_sizes_m: np.ndarray
    cells_bev: np.ndarray
    bounds_bev: np.ndarray
    centers_xyz: np.ndarray
    features: np.ndarray
    exclusion_flags: np.ndarray
    row_offsets: np.ndarray
    flat_indices: np.ndarray
    coarse_chunk_ids: np.ndarray

    def validate(self, *, background_count: int | None = None) -> None:
        count = int(self.patch_ids.shape[0])
        expected = {
            "patch_sizes_m": (count,),
            "cells_bev": (count, 2),
            "bounds_bev": (count, 4),
            "centers_xyz": (count, 3),
            "features": (count, len(FEATURE_NAMES)),
            "exclusion_flags": (count,),
            "row_offsets": (count + 1,),
            "coarse_chunk_ids": (count,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape 非法: {getattr(self, name).shape}")
        if self.flat_indices.ndim != 1:
            raise ValueError("flat_indices 必须是一维")
        if count and len(set(self.patch_ids.tolist())) != count:
            raise ValueError("patch_id 必须唯一")
        if not np.array_equal(self.row_offsets, np.sort(self.row_offsets)):
            raise ValueError("row_offsets 必须单调")
        if int(self.row_offsets[0]) != 0 or int(self.row_offsets[-1]) != len(
            self.flat_indices
        ):
            raise ValueError("row_offsets 与 flat_indices 不对齐")
        if not np.isfinite(self.features).all() or not np.isfinite(
            self.centers_xyz
        ).all():
            raise ValueError("patch index 存在非有限数值")
        if not set(np.unique(self.patch_sizes_m).tolist()) <= {1.0, 2.0, 4.0}:
            raise ValueError("patch size 超出冻结集合")
        if background_count is not None and self.flat_indices.size:
            if int(self.flat_indices.min()) < 0 or int(self.flat_indices.max()) >= int(
                background_count
            ):
                raise ValueError("donor flat index 越界")

    def rows(self, patch_index: int) -> np.ndarray:
        start = int(self.row_offsets[patch_index])
        end = int(self.row_offsets[patch_index + 1])
        return self.flat_indices[start:end]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "patch_ids": self.patch_ids,
            "patch_sizes_m": self.patch_sizes_m,
            "cells_bev": self.cells_bev,
            "bounds_bev": self.bounds_bev,
            "centers_xyz": self.centers_xyz,
            "features": self.features,
            "feature_names": np.asarray(FEATURE_NAMES, dtype="U64"),
            "exclusion_flags": self.exclusion_flags,
            "row_offsets": self.row_offsets,
            "flat_indices": self.flat_indices,
            "coarse_chunk_ids": self.coarse_chunk_ids,
            "bev_axes": np.asarray(BEV_AXES, dtype=np.int8),
            "vertical_axis": np.asarray(VERTICAL_AXIS, dtype=np.int8),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype="U64"),
        }


@dataclass(frozen=True)
class HoleAnchor:
    """由保守对象 mask 与 first-hit depth 建立的局部道路洞锚点。"""

    center_xyz: np.ndarray
    bounds_bev: np.ndarray
    patch_size_m: float
    tangent_yaw: float
    tangent_confidence: float
    context_rgb_mean: np.ndarray
    context_rgb_std: np.ndarray
    valid_point_count: int
    cross_view_observed_pixels: int


@dataclass(frozen=True)
class DonorCandidate:
    patch_index: int
    patch_id: str
    distance: float
    geometry_distance: float
    appearance_distance: float
    semantic_distance: float
    visibility_distance: float
    yaw_radians: float
    vertical_offset_m: float


def sigmoid(value: np.ndarray) -> np.ndarray:
    value64 = np.asarray(value, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-value64))).astype(np.float32)


def rgb_from_sh_dc(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float32) * SH_C0 + 0.5, 0.0, 1.0)


def sh_dc_from_rgb(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) - 0.5) / SH_C0


def sha256_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """对字段名、dtype、shape 和连续字节做无歧义 SHA。"""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_save_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                entry = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(
                    entry,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    os.replace(temporary, target)


def atomic_save_patch_index(path: str | Path, index: PatchIndex) -> None:
    index.validate()
    _atomic_save_npz(path, index.as_arrays())


def load_patch_index(path: str | Path) -> PatchIndex:
    with np.load(path, allow_pickle=False) as payload:
        feature_names = tuple(payload["feature_names"].tolist())
        if feature_names != FEATURE_NAMES:
            raise ValueError("patch feature schema 漂移")
        if str(payload["schema_version"].item()) != SCHEMA_VERSION:
            raise ValueError("patch index schema version 漂移")
        if tuple(payload["bev_axes"].tolist()) != BEV_AXES or int(
            payload["vertical_axis"].item()
        ) != VERTICAL_AXIS:
            raise ValueError("patch 坐标轴合同漂移")
        index = PatchIndex(
            patch_ids=payload["patch_ids"],
            patch_sizes_m=payload["patch_sizes_m"],
            cells_bev=payload["cells_bev"],
            bounds_bev=payload["bounds_bev"],
            centers_xyz=payload["centers_xyz"],
            features=payload["features"],
            exclusion_flags=payload["exclusion_flags"],
            row_offsets=payload["row_offsets"],
            flat_indices=payload["flat_indices"],
            coarse_chunk_ids=payload["coarse_chunk_ids"],
        )
    index.validate()
    return index


def _fit_local_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """在 x/z 上拟合 y；输出 y-down 坐标中的单位法向与 RMS。"""

    if points.shape[0] < 3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32), float(
            np.finfo(np.float32).max
        )
    design = np.column_stack(
        [points[:, BEV_AXES[0]], points[:, BEV_AXES[1]], np.ones(points.shape[0])]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design.astype(np.float64), points[:, VERTICAL_AXIS].astype(np.float64), rcond=None
    )
    if rank < 3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32), float(
            np.finfo(np.float32).max
        )
    predicted = design @ coefficients
    residual = float(
        np.sqrt(np.mean(np.square(predicted - points[:, VERTICAL_AXIS])))
    )
    normal = np.array([-coefficients[0], 1.0, -coefficients[1]], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return normal.astype(np.float32), residual


def _tangent(points: np.ndarray) -> tuple[float, float]:
    bev = points[:, BEV_AXES].astype(np.float64)
    if bev.shape[0] < 3:
        return 0.0, 0.0
    covariance = np.cov(bev, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    major = vectors[:, order[-1]]
    confidence = float(
        max(values[order[-1]] - values[order[-2]], 0.0)
        / max(values[order[-1]], 1e-12)
    )
    yaw = float(np.arctan2(major[1], major[0]))
    # 道路切向无方向，统一到 [-pi/2, pi/2)。
    yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)
    return yaw, confidence


def _coarse_chunk_id(points: np.ndarray, size_m: float = 50.0) -> str:
    cells = np.unique(
        np.floor(points[:, BEV_AXES].astype(np.float64) / float(size_m)).astype(
            np.int64
        ),
        axis=0,
    )
    return ";".join(f"xz:{int(x)}:{int(z)}" for x, z in cells)


def native_row_eligibility(
    *,
    raw_scales: np.ndarray,
    actor_semantic_score: np.ndarray,
    train_view_observation_count: np.ndarray,
    visibility_mass: np.ndarray,
    native_donor_mask: np.ndarray,
    thresholds: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """在成 patch 前逐行 fail-closed，避免天空/立面异常点污染整块路面。"""

    raw_scales = np.asarray(raw_scales)
    count = int(raw_scales.shape[0])
    if raw_scales.shape != (count, 3):
        raise ValueError("raw_scales 必须是 (N,3)")
    actor = np.asarray(actor_semantic_score).reshape(-1)
    observations = np.asarray(train_view_observation_count).reshape(-1)
    mass = np.asarray(visibility_mass).reshape(-1)
    native = np.asarray(native_donor_mask, dtype=bool).reshape(-1)
    if any(value.shape != (count,) for value in (actor, observations, mass, native)):
        raise ValueError("row eligibility 输入行数不一致")
    maximum_scale = np.exp(raw_scales.astype(np.float64)).max(axis=1)
    reasons = {
        "actor_semantic": actor > float(thresholds["maximum_actor_semantic"]),
        "low_visibility": mass < float(thresholds["minimum_visibility_mass"]),
        "low_view_support": observations
        < float(thresholds["minimum_train_view_observations"]),
        "scale_outlier": maximum_scale > float(thresholds["maximum_scale_m"]),
        "generated": ~native,
    }
    eligible = ~np.logical_or.reduce(tuple(reasons.values()))
    return eligible, reasons


def _dominant_vertical_layer(
    rows: np.ndarray, means: np.ndarray, maximum_range_m: float
) -> np.ndarray:
    """返回 y 轴上覆盖点数最多的确定性闭区间，宽度不超过冻结上限。"""

    if rows.size <= 1:
        return rows
    ordered = rows[
        np.lexsort((rows, means[rows, VERTICAL_AXIS].astype(np.float64)))
    ]
    vertical = means[ordered, VERTICAL_AXIS].astype(np.float64)
    best_start = 0
    best_end = 1
    start = 0
    for end in range(vertical.size):
        while vertical[end] - vertical[start] > float(maximum_range_m):
            start += 1
        if end + 1 - start > best_end - best_start:
            best_start, best_end = start, end + 1
    return np.sort(ordered[best_start:best_end].astype(np.int64))


def build_patch_index(
    *,
    means: np.ndarray,
    raw_scales: np.ndarray,
    raw_opacities: np.ndarray,
    features_dc: np.ndarray,
    actor_semantic_score: np.ndarray,
    train_view_observation_count: np.ndarray,
    visibility_mass: np.ndarray,
    multi_camera_count: np.ndarray,
    native_donor_mask: np.ndarray,
    patch_sizes_m: Sequence[float],
    thresholds: Mapping[str, float],
) -> PatchIndex:
    """从 immutable Background 建立全场景、可审计的多尺度 patch index。"""

    means = np.asarray(means, dtype=np.float32)
    count = int(means.shape[0])
    if means.shape != (count, 3) or count == 0:
        raise ValueError("Background means 必须为非空 (N,3)")
    arrays = {
        "raw_scales": np.asarray(raw_scales),
        "raw_opacities": np.asarray(raw_opacities).reshape(-1),
        "features_dc": np.asarray(features_dc),
        "actor_semantic_score": np.asarray(actor_semantic_score).reshape(-1),
        "train_view_observation_count": np.asarray(
            train_view_observation_count
        ).reshape(-1),
        "visibility_mass": np.asarray(visibility_mass).reshape(-1),
        "multi_camera_count": np.asarray(multi_camera_count).reshape(-1),
        "native_donor_mask": np.asarray(native_donor_mask, dtype=bool).reshape(-1),
    }
    if any(value.shape[0] != count for value in arrays.values()):
        raise ValueError("patch feature 输入与 Background 行数不一致")
    if arrays["raw_scales"].shape != (count, 3) or arrays[
        "features_dc"
    ].shape != (count, 3):
        raise ValueError("scale/SH-DC schema 非 (N,3)")
    sizes = tuple(float(value) for value in patch_sizes_m)
    if sizes != (1.0, 2.0, 4.0):
        raise ValueError("patch size 必须严格冻结为 1/2/4 m")

    scales = np.exp(arrays["raw_scales"].astype(np.float64)).astype(np.float32)
    opacities = sigmoid(arrays["raw_opacities"])
    rgb = rgb_from_sh_dc(arrays["features_dc"])
    eligible, _ = native_row_eligibility(
        raw_scales=arrays["raw_scales"],
        actor_semantic_score=arrays["actor_semantic_score"],
        train_view_observation_count=arrays["train_view_observation_count"],
        visibility_mass=arrays["visibility_mass"],
        native_donor_mask=arrays["native_donor_mask"],
        thresholds=thresholds,
    )
    eligible_rows = np.flatnonzero(eligible).astype(np.int64)
    if eligible_rows.size == 0:
        raise ValueError("逐行 fail-closed 后没有原生 donor Gaussian")
    patch_ids: list[str] = []
    patch_sizes: list[float] = []
    cells_out: list[np.ndarray] = []
    bounds_out: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    features: list[np.ndarray] = []
    flags_out: list[np.uint16] = []
    rows_out: list[np.ndarray] = []
    chunk_ids: list[str] = []

    for size in sizes:
        cells = np.floor(means[eligible_rows][:, BEV_AXES].astype(np.float64) / size).astype(
            np.int64
        )
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        ordered_cells = cells[order]
        starts = np.r_[
            0, 1 + np.flatnonzero(np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1))
        ]
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts.tolist(), ends.tolist()):
            rows = np.sort(eligible_rows[order[start:end]].astype(np.int64))
            rows = _dominant_vertical_layer(
                rows,
                means,
                float(thresholds["maximum_vertical_range_m"]),
            )
            cell = ordered_cells[start]
            points = means[rows]
            normal, plane_residual = _fit_local_plane(points)
            tangent_yaw, tangent_confidence = _tangent(points)
            point_scales = scales[rows]
            point_opacity = opacities[rows]
            point_rgb = rgb[rows]
            semantic = arrays["actor_semantic_score"][rows]
            observations = arrays["train_view_observation_count"][rows]
            mass = arrays["visibility_mass"][rows]
            cameras = arrays["multi_camera_count"][rows]
            vertical = points[:, VERTICAL_AXIS]
            feature = np.array(
                [
                    float(vertical.mean()),
                    float(vertical.std()),
                    *normal.tolist(),
                    plane_residual,
                    float(len(rows) / (size * size)),
                    float(point_scales.mean()),
                    float(point_scales.max()),
                    float(point_opacity.mean()),
                    float(point_opacity.std()),
                    float(vertical.max() - vertical.min()),
                    *point_rgb.mean(axis=0).tolist(),
                    *point_rgb.std(axis=0).tolist(),
                    float(point_rgb.std(axis=0).mean()),
                    float(semantic.mean()),
                    float(semantic.max()),
                    float(1.0 - semantic.mean()),
                    float(observations.mean()),
                    float(cameras.mean()),
                    float(np.log1p(mass).mean()),
                    float(np.cos(tangent_yaw)),
                    float(np.sin(tangent_yaw)),
                    tangent_confidence,
                ],
                dtype=np.float32,
            )
            flags = np.uint16(0)
            if len(rows) < int(thresholds["minimum_rows"]):
                flags |= EXCLUDE_SPARSE
            if int(
                np.sum(cameras >= float(thresholds["minimum_multi_camera_count"]))
            ) < int(thresholds["minimum_multi_camera_rows"]):
                flags |= EXCLUDE_LOW_VIEW_SUPPORT
            if plane_residual > float(
                thresholds["maximum_plane_residual_m"]
            ) or abs(float(normal[VERTICAL_AXIS])) < float(
                thresholds["minimum_abs_plane_normal_vertical"]
            ) or float(vertical.max() - vertical.min()) > float(
                thresholds["maximum_vertical_range_m"]
            ):
                flags |= EXCLUDE_GEOMETRY_OUTLIER
            if not bool(eligible[rows].all()):
                raise RuntimeError("patch rows 混入逐行 fail-closed 排除项")
            bounds = np.array(
                [cell[0] * size, cell[1] * size, (cell[0] + 1) * size, (cell[1] + 1) * size],
                dtype=np.float32,
            )
            patch_ids.append(
                f"p{int(size)}-x{int(cell[0]):+07d}-z{int(cell[1]):+07d}"
            )
            patch_sizes.append(size)
            cells_out.append(cell.astype(np.int32))
            bounds_out.append(bounds)
            centers.append(points.mean(axis=0).astype(np.float32))
            features.append(feature)
            flags_out.append(flags)
            rows_out.append(rows)
            chunk_ids.append(_coarse_chunk_id(points))

    offsets = np.zeros(len(rows_out) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(rows) for rows in rows_out], dtype=np.int64)
    index = PatchIndex(
        patch_ids=np.asarray(patch_ids, dtype="U40"),
        patch_sizes_m=np.asarray(patch_sizes, dtype=np.float32),
        cells_bev=np.asarray(cells_out, dtype=np.int32),
        bounds_bev=np.asarray(bounds_out, dtype=np.float32),
        centers_xyz=np.asarray(centers, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        exclusion_flags=np.asarray(flags_out, dtype=np.uint16),
        row_offsets=offsets,
        flat_indices=np.concatenate(rows_out).astype(np.int64),
        coarse_chunk_ids=np.asarray(chunk_ids, dtype="U256"),
    )
    index.validate(background_count=count)
    return index


def conservative_delete_mask(
    instance_mask: np.ndarray, semantic_mask: np.ndarray
) -> np.ndarray:
    """用 S1 identity 与 SAM2 actor 证据交集抑制实例场 false positive。"""

    instance = np.asarray(instance_mask, dtype=bool)
    semantic = np.asarray(semantic_mask, dtype=bool)
    if instance.shape != semantic.shape or instance.ndim != 2:
        raise ValueError("delete masks 必须是同 shape 二维数组")
    result = instance & semantic
    if not result.any():
        raise ValueError("S1/SAM2 交集为空，delete mask ABSTAIN")
    return result


def _unproject(
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    y, x = np.nonzero(valid)
    z = np.asarray(depth, dtype=np.float64)[y, x]
    k = np.asarray(intrinsics, dtype=np.float64)
    camera = np.column_stack(
        [(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
    )
    homogeneous = np.column_stack([camera, np.ones(camera.shape[0])])
    return (homogeneous @ np.asarray(camera_to_world, dtype=np.float64).T)[:, :3]


def build_hole_anchor(
    *,
    delete_mask: np.ndarray,
    first_hit_depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    cross_view_observed_pixels: int,
    patch_sizes_m: Sequence[float],
    bottom_quantile: float,
    robust_quantiles: tuple[float, float],
    minimum_anchor_pixels: int,
    minimum_cross_view_observed_pixels: int,
    ring_pixels: int,
) -> HoleAnchor:
    """只用对象底部 first-hit depth 建锚；无合法深度或超 4 m 时 ABSTAIN。"""

    from scipy.ndimage import binary_dilation

    mask = np.asarray(delete_mask, dtype=bool)
    depth = np.asarray(first_hit_depth)
    image = np.asarray(rgb)
    if mask.shape != depth.shape or image.shape[:2] != mask.shape:
        raise ValueError("anchor 输入 shape 不一致")
    if not 0.0 <= bottom_quantile < 1.0:
        raise ValueError("bottom_quantile 非法")
    if int(cross_view_observed_pixels) < int(minimum_cross_view_observed_pixels):
        raise ValueError("cross-view observed support 不足，ABSTAIN")
    y_rows = np.nonzero(mask)[0]
    if y_rows.size == 0:
        raise ValueError("delete mask 为空，ABSTAIN")
    cutoff = float(np.quantile(y_rows, bottom_quantile))
    row_grid = np.indices(mask.shape)[0]
    valid = (
        mask
        & (row_grid >= cutoff)
        & np.isfinite(depth)
        & (depth > 1e-4)
    )
    if int(valid.sum()) < int(minimum_anchor_pixels):
        raise ValueError("first-hit depth anchor 像素不足，ABSTAIN")
    points = _unproject(depth, valid, intrinsics, camera_to_world)
    low, high = (float(value) for value in robust_quantiles)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("robust quantiles 非法")
    bounds = np.array(
        [
            np.quantile(points[:, BEV_AXES[0]], low),
            np.quantile(points[:, BEV_AXES[1]], low),
            np.quantile(points[:, BEV_AXES[0]], high),
            np.quantile(points[:, BEV_AXES[1]], high),
        ],
        dtype=np.float32,
    )
    span = float(max(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    allowed = tuple(float(value) for value in patch_sizes_m)
    selected = next((size for size in allowed if span <= size), None)
    if selected is None:
        raise ValueError(f"hole BEV span={span:.6f}m 超过 4m，ABSTAIN")
    yaw, confidence = _tangent(points)
    ring = binary_dilation(mask, iterations=int(ring_pixels)) & ~mask
    if not ring.any():
        raise ValueError("delete mask context ring 为空")
    ring_rgb = image[ring].astype(np.float32) / 255.0
    center = np.median(points, axis=0).astype(np.float32)
    center[BEV_AXES[0]] = 0.5 * (bounds[0] + bounds[2])
    center[BEV_AXES[1]] = 0.5 * (bounds[1] + bounds[3])
    return HoleAnchor(
        center_xyz=center,
        bounds_bev=bounds,
        patch_size_m=float(selected),
        tangent_yaw=yaw,
        tangent_confidence=confidence,
        context_rgb_mean=ring_rgb.mean(axis=0).astype(np.float32),
        context_rgb_std=ring_rgb.std(axis=0).astype(np.float32),
        valid_point_count=int(points.shape[0]),
        cross_view_observed_pixels=int(cross_view_observed_pixels),
    )


def _bbox_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left[0] < right[2]
        and left[2] > right[0]
        and left[1] < right[3]
        and left[3] > right[1]
    )


def _wrapped_axis_yaw_delta(target: float, donor: float) -> float:
    delta = float((target - donor + np.pi / 2) % np.pi - np.pi / 2)
    return delta


def search_donors(
    *,
    index: PatchIndex,
    anchor: HoleAnchor,
    top_k: int,
    weights: Mapping[str, float],
    minimum_spatial_separation_m: float,
    minimum_tangent_confidence: float,
    maximum_abs_yaw_radians: float,
    maximum_abs_vertical_offset_m: float,
) -> list[DonorCandidate]:
    """fail-closed 排除后按冻结四组距离返回恰好 top-K。"""

    index.validate()
    if int(top_k) != 5:
        raise ValueError("RoadPatch-Lite top-K 必须冻结为 5")
    candidates: list[DonorCandidate] = []
    for row in np.flatnonzero(
        (index.patch_sizes_m == float(anchor.patch_size_m))
        & (index.exclusion_flags == 0)
    ):
        if _bbox_overlap(index.bounds_bev[row], anchor.bounds_bev):
            continue
        delta_bev = index.centers_xyz[row, list(BEV_AXES)] - anchor.center_xyz[
            list(BEV_AXES)
        ]
        if float(np.linalg.norm(delta_bev)) < float(minimum_spatial_separation_m):
            continue
        feature = index.features[row]
        vertical_offset = float(
            anchor.center_xyz[VERTICAL_AXIS]
            - feature[FEATURE_INDEX["mean_vertical"]]
        )
        if abs(vertical_offset) > float(maximum_abs_vertical_offset_m):
            continue
        donor_yaw = float(
            np.arctan2(
                feature[FEATURE_INDEX["tangent_sin"]],
                feature[FEATURE_INDEX["tangent_cos"]],
            )
        )
        if min(
            float(anchor.tangent_confidence),
            float(feature[FEATURE_INDEX["tangent_confidence"]]),
        ) >= float(minimum_tangent_confidence):
            yaw = _wrapped_axis_yaw_delta(anchor.tangent_yaw, donor_yaw)
            if abs(yaw) > float(maximum_abs_yaw_radians):
                yaw = 0.0
        else:
            yaw = 0.0
        normal = feature[
            [
                FEATURE_INDEX["plane_normal_x"],
                FEATURE_INDEX["plane_normal_y"],
                FEATURE_INDEX["plane_normal_z"],
            ]
        ]
        geometry = (
            float(1.0 - abs(normal[VERTICAL_AXIS]))
            + float(feature[FEATURE_INDEX["plane_residual"]])
            + 0.25 * abs(vertical_offset)
        )
        donor_mean = feature[
            [FEATURE_INDEX["rgb_mean_r"], FEATURE_INDEX["rgb_mean_g"], FEATURE_INDEX["rgb_mean_b"]]
        ]
        donor_std = feature[
            [FEATURE_INDEX["rgb_std_r"], FEATURE_INDEX["rgb_std_g"], FEATURE_INDEX["rgb_std_b"]]
        ]
        appearance = float(
            np.mean(np.abs(donor_mean - anchor.context_rgb_mean))
            + 0.5 * np.mean(np.abs(donor_std - anchor.context_rgb_std))
        )
        semantic = float(feature[FEATURE_INDEX["actor_semantic_mean"]])
        views = float(feature[FEATURE_INDEX["train_view_observation_mean"]])
        cameras = float(feature[FEATURE_INDEX["multi_camera_count_mean"]])
        mass = float(feature[FEATURE_INDEX["log_visibility_mass_mean"]])
        visibility = float(1.0 / (1.0 + views) + 1.0 / (1.0 + cameras) + 1.0 / (1.0 + mass))
        distance = (
            float(weights["geometry"]) * geometry
            + float(weights["appearance"]) * appearance
            + float(weights["semantic"]) * semantic
            + float(weights["visibility"]) * visibility
        )
        candidates.append(
            DonorCandidate(
                patch_index=int(row),
                patch_id=str(index.patch_ids[row]),
                distance=float(distance),
                geometry_distance=geometry,
                appearance_distance=appearance,
                semantic_distance=semantic,
                visibility_distance=visibility,
                yaw_radians=float(yaw),
                vertical_offset_m=vertical_offset,
            )
        )
    candidates.sort(key=lambda item: (item.distance, item.patch_id))
    if len(candidates) < int(top_k):
        raise ValueError(f"合法 donor 只有 {len(candidates)} 个，少于 top-K=5，ABSTAIN")
    return candidates[: int(top_k)]


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def materialize_patch_delta(
    *,
    index: PatchIndex,
    candidate: DonorCandidate,
    anchor: HoleAnchor,
    background_state: Mapping[str, np.ndarray],
    source_gaussian_ids: np.ndarray,
    target_role: str,
    opacity_feather_width_m: float,
    maximum_rgb_affine: float,
    minimum_scale_m: float,
    maximum_scale_m: float,
    duplicate_radius_m: float,
    base_tree: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """只复制 donor 行并生成刚性空间 delta；source arrays 永不原位修改。"""

    from scipy.spatial import cKDTree

    index.validate()
    rows = index.rows(candidate.patch_index)
    required = {
        "_means": (None, 3),
        "_scales": (None, 3),
        "_quats": (None, 4),
        "_features_dc": (None, 3),
        "_features_rest": None,
        "_opacities": None,
    }
    arrays = {name: np.asarray(background_state[name]) for name in required}
    count = arrays["_means"].shape[0]
    if any(value.shape[0] != count for value in arrays.values()):
        raise ValueError("Background state 行数不一致")
    if np.asarray(source_gaussian_ids).shape != (count,):
        raise ValueError("source Gaussian ID 与 Background 行数不一致")
    source_means = arrays["_means"][rows].astype(np.float64)
    donor_center = index.centers_xyz[candidate.patch_index].astype(np.float64)
    target_center = anchor.center_xyz.astype(np.float64)
    yaw = float(candidate.yaw_radians)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    transformed = (source_means - donor_center) @ rotation.T + target_center
    # 仅抑制真正近重合点，不用大半径把目标区域原有稀疏路面全部删掉。
    tree = base_tree
    if tree is None:
        tree = cKDTree(arrays["_means"].astype(np.float64))
    nearest, _ = tree.query(
        transformed, k=1, workers=-1
    )
    keep = nearest > float(duplicate_radius_m)
    if not keep.any():
        raise ValueError("duplicate suppression 后 delta 为空，ABSTAIN")
    rows = rows[keep]
    transformed = transformed[keep]
    relative_bev = source_means[keep][:, list(BEV_AXES)] - donor_center[list(BEV_AXES)]
    half = float(anchor.patch_size_m) / 2.0
    edge_distance = half - np.max(np.abs(relative_bev), axis=1)
    feather = np.clip(edge_distance / float(opacity_feather_width_m), 0.0, 1.0)

    source_rgb = rgb_from_sh_dc(arrays["_features_dc"][rows])
    donor_mean = source_rgb.mean(axis=0)
    rgb_shift = np.clip(
        anchor.context_rgb_mean - donor_mean,
        -float(maximum_rgb_affine),
        float(maximum_rgb_affine),
    )
    adjusted_rgb = np.clip(source_rgb + rgb_shift, 1e-5, 1.0 - 1e-5)
    source_opacity = sigmoid(arrays["_opacities"][rows].reshape(-1))
    opacity = np.clip(source_opacity * feather, 1e-5, 1.0 - 1e-5)
    raw_scale = arrays["_scales"][rows].copy()
    scale = np.clip(
        np.exp(raw_scale.astype(np.float64)),
        float(minimum_scale_m),
        float(maximum_scale_m),
    )
    quats = arrays["_quats"][rows].astype(np.float64)
    yaw_quat = np.zeros_like(quats)
    yaw_quat[:, 0] = np.cos(yaw / 2.0)
    yaw_quat[:, VERTICAL_AXIS + 1] = np.sin(yaw / 2.0)
    transformed_quats = _quaternion_multiply(yaw_quat, quats)
    transformed_quats /= np.maximum(
        np.linalg.norm(transformed_quats, axis=1, keepdims=True), 1e-12
    )
    delta = {
        "means": transformed.astype(np.float32),
        "raw_scales": np.log(scale).astype(np.float32),
        "quats": transformed_quats.astype(np.float32),
        "features_dc": sh_dc_from_rgb(adjusted_rgb).astype(np.float32),
        "features_rest": arrays["_features_rest"][rows].copy(),
        "raw_opacities": np.log(opacity / (1.0 - opacity)).astype(np.float32)[:, None],
        "source_flat_indices": rows.astype(np.int64),
        "source_gaussian_ids": np.asarray(source_gaussian_ids)[rows].astype(np.int64),
        "feather_weight": feather.astype(np.float32),
        "provenance_code": np.full(
            len(rows), PROVENANCE_GENERATED_BY_PATCH_REUSE, dtype=np.uint8
        ),
        "target_role": np.full(len(rows), str(target_role), dtype="U32"),
        "donor_patch_id": np.full(len(rows), candidate.patch_id, dtype="U40"),
        "donor_chunk_ids": np.full(
            len(rows), str(index.coarse_chunk_ids[candidate.patch_index]), dtype="U256"
        ),
    }
    source_hash = sha256_arrays(
        {
            name: arrays[name][rows]
            for name in (
                "_means",
                "_scales",
                "_quats",
                "_features_dc",
                "_features_rest",
                "_opacities",
            )
        }
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ donor_center
    manifest = {
        "schema_version": DELTA_SCHEMA_VERSION,
        "provenance": "GENERATED_BY_PATCH_REUSE",
        "target_role": str(target_role),
        "donor_patch_id": candidate.patch_id,
        "donor_chunk_ids": str(index.coarse_chunk_ids[candidate.patch_index]).split(";"),
        "donor_flat_indices_count": int(len(rows)),
        "source_gaussian_hash": source_hash,
        "transform": transform.tolist(),
        "yaw_radians": yaw,
        "vertical_axis": VERTICAL_AXIS,
        "vertical_offset_m": float(candidate.vertical_offset_m),
        "rgb_shift": rgb_shift.astype(float).tolist(),
        "duplicate_suppressed_count": int((~keep).sum()),
        "source_checkpoint_mutated": False,
    }
    return delta, manifest


def validate_patch_delta(delta: Mapping[str, np.ndarray]) -> None:
    required = {
        "means",
        "raw_scales",
        "quats",
        "features_dc",
        "features_rest",
        "raw_opacities",
        "source_flat_indices",
        "source_gaussian_ids",
        "feather_weight",
        "provenance_code",
        "target_role",
        "donor_patch_id",
        "donor_chunk_ids",
    }
    if set(delta) != required:
        raise ValueError(f"delta 字段漂移: {sorted(set(delta) ^ required)}")
    count = int(np.asarray(delta["means"]).shape[0])
    if count <= 0 or any(np.asarray(value).shape[0] != count for value in delta.values()):
        raise ValueError("delta 行数不一致或为空")
    if np.asarray(delta["means"]).shape != (count, 3):
        raise ValueError("delta means 非 (N,3)")
    if np.asarray(delta["quats"]).shape != (count, 4):
        raise ValueError("delta quaternion 非 (N,4)")
    if not np.all(
        np.asarray(delta["provenance_code"]) == PROVENANCE_GENERATED_BY_PATCH_REUSE
    ):
        raise ValueError("delta provenance 不是 patch reuse")
    numeric = (
        "means",
        "raw_scales",
        "quats",
        "features_dc",
        "features_rest",
        "raw_opacities",
        "feather_weight",
    )
    if any(not np.isfinite(np.asarray(delta[name])).all() for name in numeric):
        raise ValueError("delta 存在非有限数值")


def atomic_save_patch_delta(path: str | Path, delta: Mapping[str, np.ndarray]) -> None:
    validate_patch_delta(delta)
    _atomic_save_npz(path, delta)


def load_patch_delta(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        delta = {name: payload[name] for name in payload.files}
    validate_patch_delta(delta)
    return delta
