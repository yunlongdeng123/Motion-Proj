"""LUDVIG 语义的 learning-free Gaussian feature uplift 纯算子。

Provenance：LUDVIG commit ``4461fc515439bb498a75d71738a1e73cf7a452ed``
的 ``utils/solver.py::uplifting`` 与 ``apply_weights.cu``。上游受
``LICENSE.txt`` non-commercial 条款约束；本模块不 vendor 上游源码，只实现已冻结的
``sum(w*f)/(sum(w)+1e-8)`` 数学合同，并保留该来源声明。
"""

from __future__ import annotations

from typing import Any

import numpy as np


LUDVIG_UPSTREAM_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
LUDVIG_LICENSE = "non-commercial"


def _validate_uplift_inputs(
    gaussian_id: np.ndarray,
    view_id: np.ndarray,
    contribution_weight: np.ndarray,
    pixel_features: np.ndarray,
    gaussian_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.dtype[Any]]:
    gids = np.asarray(gaussian_id, dtype=np.int64)
    views = np.asarray(view_id, dtype=np.int64)
    weights = np.asarray(contribution_weight, dtype=np.float64)
    features_input = np.asarray(pixel_features)
    if gids.ndim != 1 or views.shape != gids.shape or weights.shape != gids.shape:
        raise ValueError("gaussian_id/view_id/contribution_weight 必须是一一对齐的一维数组")
    if features_input.ndim != 2 or features_input.shape[0] != gids.size:
        raise ValueError("pixel_features 必须为与 intersection 对齐的 [N,D]")
    if not np.issubdtype(features_input.dtype, np.floating):
        raise ValueError("pixel_features 必须使用浮点 dtype")
    if int(gaussian_count) <= 0:
        raise ValueError("gaussian_count 必须大于 0")
    if np.any((gids < 0) | (gids >= int(gaussian_count))):
        raise ValueError("gaussian_id 越界")
    if np.any(views < 0):
        raise ValueError("view_id 必须非负")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("contribution_weight 必须为有限非负数")
    if not np.isfinite(features_input).all():
        raise ValueError("pixel_features 必须全部 finite")
    return gids, views, weights, features_input.astype(np.float64), features_input.dtype


def uplift_b0_b1(
    *,
    gaussian_id: np.ndarray,
    view_id: np.ndarray,
    contribution_weight: np.ndarray,
    pixel_features: np.ndarray,
    gaussian_count: int,
    minimum_intersection_contribution: float = 1e-4,
    minimum_gaussian_view_mass: float = 1e-3,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """在共同 support 上同时计算 view-saturated B0 与 normalized-transpose B1。

    先按 ``(view_id, gaussian_id)`` 收缩重复 intersection，再应用 Gaussian-view
    mass floor。内部用 float64 和 canonical 排序，最终 feature 恢复输入浮点 dtype。
    """

    for name, value in (
        ("minimum_intersection_contribution", minimum_intersection_contribution),
        ("minimum_gaussian_view_mass", minimum_gaussian_view_mass),
        ("epsilon", epsilon),
    ):
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须为有限正数")
    gids, views, weights, features, output_dtype = _validate_uplift_inputs(
        gaussian_id,
        view_id,
        contribution_weight,
        pixel_features,
        gaussian_count,
    )
    feature_dimension = int(features.shape[1])
    supported_rows = weights >= float(minimum_intersection_contribution)
    gids = gids[supported_rows]
    views = views[supported_rows]
    weights = weights[supported_rows]
    features = features[supported_rows]

    b0 = np.zeros((int(gaussian_count), feature_dimension), dtype=np.float64)
    b1 = np.zeros_like(b0)
    b0_denominator = np.zeros(int(gaussian_count), dtype=np.float64)
    b1_denominator = np.zeros(int(gaussian_count), dtype=np.float64)
    supported_view_count = np.zeros(int(gaussian_count), dtype=np.int32)
    gaussian_view_count = 0
    supported_gaussian_view_count = 0
    total_mass = 0.0
    supported_mass = 0.0

    if gids.size:
        # 组内也按 weight/feature 排序，确保输入行或 chunk 排列不改变求和顺序。
        sort_keys = [features[:, index] for index in reversed(range(feature_dimension))]
        sort_keys.extend((weights, gids, views))
        order = np.lexsort(tuple(sort_keys))
        gids = gids[order]
        views = views[order]
        weights = weights[order]
        features = features[order]
        starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero((gids[1:] != gids[:-1]) | (views[1:] != views[:-1]))
                + 1,
            )
        )
        group_gids = gids[starts]
        group_views = views[starts]
        mass = np.add.reduceat(weights, starts)
        numerator = np.add.reduceat(weights[:, None] * features, starts, axis=0)
        gaussian_view_count = int(starts.size)
        total_mass = float(mass.sum())
        keep = mass >= float(minimum_gaussian_view_mass)
        group_gids = group_gids[keep]
        group_views = group_views[keep]
        mass = mass[keep]
        numerator = numerator[keep]
        supported_gaussian_view_count = int(keep.sum())
        supported_mass = float(mass.sum())

        if group_gids.size:
            group_order = np.lexsort((group_views, group_gids))
            group_gids = group_gids[group_order]
            mass = mass[group_order]
            numerator = numerator[group_order]
            gaussian_starts = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.flatnonzero(group_gids[1:] != group_gids[:-1]) + 1,
                )
            )
            unique_gids = group_gids[gaussian_starts]
            view_feature = numerator / (mass[:, None] + float(epsilon))
            saturation = -np.expm1(-mass)
            b0_numerator = np.add.reduceat(
                saturation[:, None] * view_feature, gaussian_starts, axis=0
            )
            b0_mass = np.add.reduceat(saturation, gaussian_starts)
            b1_numerator = np.add.reduceat(numerator, gaussian_starts, axis=0)
            b1_mass = np.add.reduceat(mass, gaussian_starts)
            counts = np.diff(np.append(gaussian_starts, group_gids.size))
            b0[unique_gids] = b0_numerator / (
                b0_mass[:, None] + float(epsilon)
            )
            b1[unique_gids] = b1_numerator / (
                b1_mass[:, None] + float(epsilon)
            )
            b0_denominator[unique_gids] = b0_mass
            b1_denominator[unique_gids] = b1_mass
            supported_view_count[unique_gids] = counts.astype(np.int32)

    report = {
        "input_intersection_count": int(np.asarray(gaussian_id).size),
        "supported_intersection_count": int(supported_rows.sum()),
        "gaussian_view_count_before_mass_floor": gaussian_view_count,
        "supported_gaussian_view_count": supported_gaussian_view_count,
        "dropped_gaussian_view_count": gaussian_view_count
        - supported_gaussian_view_count,
        "covered_gaussian_count": int(np.count_nonzero(b1_denominator > 0.0)),
        "total_contribution_mass_after_intersection_floor": total_mass,
        "supported_contribution_mass": supported_mass,
        "minimum_intersection_contribution": float(
            minimum_intersection_contribution
        ),
        "minimum_gaussian_view_mass": float(minimum_gaussian_view_mass),
        "epsilon": float(epsilon),
        "accumulator_dtype": "float64",
        "output_dtype": str(output_dtype),
        "ludvig_upstream_commit": LUDVIG_UPSTREAM_COMMIT,
        "ludvig_license": LUDVIG_LICENSE,
        "optional_pruning": False,
    }
    return {
        "b0_feature": b0.astype(output_dtype, copy=False),
        "b1_feature": b1.astype(output_dtype, copy=False),
        "b0_denominator": b0_denominator,
        "b1_denominator": b1_denominator,
        "supported_view_count": supported_view_count,
        "report": report,
    }


def sample_patch_grid_bilinear(
    patch_grid: np.ndarray,
    pixel_id: np.ndarray,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """按 ``torch.interpolate(..., align_corners=False)`` 懒采样 CHW patch grid。"""

    grid = np.asarray(patch_grid)
    pixels = np.asarray(pixel_id, dtype=np.int64)
    if grid.ndim != 3 or not np.issubdtype(grid.dtype, np.floating):
        raise ValueError("patch_grid 必须为浮点 [C,H_patch,W_patch]")
    if pixels.ndim != 1:
        raise ValueError("pixel_id 必须是一维数组")
    if int(image_height) <= 0 or int(image_width) <= 0:
        raise ValueError("image_height/image_width 必须大于 0")
    if not np.isfinite(grid).all():
        raise ValueError("patch_grid 必须全部 finite")
    pixel_count = int(image_height) * int(image_width)
    if np.any((pixels < 0) | (pixels >= pixel_count)):
        raise ValueError("pixel_id 越界")

    _, patch_height, patch_width = grid.shape
    pixel_x = pixels % int(image_width)
    pixel_y = pixels // int(image_width)
    source_x = (pixel_x.astype(np.float64) + 0.5) * (
        patch_width / float(image_width)
    ) - 0.5
    source_y = (pixel_y.astype(np.float64) + 0.5) * (
        patch_height / float(image_height)
    ) - 0.5
    x0_unclipped = np.floor(source_x).astype(np.int64)
    y0_unclipped = np.floor(source_y).astype(np.int64)
    x_weight = source_x - x0_unclipped
    y_weight = source_y - y0_unclipped
    x0 = np.clip(x0_unclipped, 0, patch_width - 1)
    y0 = np.clip(y0_unclipped, 0, patch_height - 1)
    x1 = np.clip(x0_unclipped + 1, 0, patch_width - 1)
    y1 = np.clip(y0_unclipped + 1, 0, patch_height - 1)
    values = grid.astype(np.float64, copy=False)
    top_left = values[:, y0, x0].T
    top_right = values[:, y0, x1].T
    bottom_left = values[:, y1, x0].T
    bottom_right = values[:, y1, x1].T
    top = top_left * (1.0 - x_weight[:, None]) + top_right * x_weight[:, None]
    bottom = (
        bottom_left * (1.0 - x_weight[:, None])
        + bottom_right * x_weight[:, None]
    )
    sampled = top * (1.0 - y_weight[:, None]) + bottom * y_weight[:, None]
    return sampled.astype(grid.dtype, copy=False)


def initialize_streaming_uplift(
    *, gaussian_count: int, feature_dimension: int
) -> dict[str, Any]:
    """创建逐 view 累积的 B0/B1 float64 状态。"""
    if int(gaussian_count) <= 0 or int(feature_dimension) <= 0:
        raise ValueError("streaming uplift shape 必须为正")
    shape = (int(gaussian_count), int(feature_dimension))
    return {
        "b0_numerator": np.zeros(shape, dtype=np.float64),
        "b0_denominator": np.zeros(int(gaussian_count), dtype=np.float64),
        "b1_numerator": np.zeros(shape, dtype=np.float64),
        "b1_denominator": np.zeros(int(gaussian_count), dtype=np.float64),
        "supported_view_count": np.zeros(int(gaussian_count), dtype=np.int32),
        "input_intersection_count": 0,
        "supported_intersection_count": 0,
        "gaussian_view_count_before_mass_floor": 0,
        "supported_gaussian_view_count": 0,
        "total_contribution_mass_after_intersection_floor": 0.0,
        "supported_contribution_mass": 0.0,
        "processed_view_count": 0,
    }


def accumulate_streaming_uplift_view(
    state: dict[str, Any],
    *,
    gaussian_id: np.ndarray,
    pixel_id: np.ndarray,
    contribution_weight: np.ndarray,
    patch_grid: np.ndarray,
    image_height: int,
    image_width: int,
    minimum_intersection_contribution: float = 1e-4,
    minimum_gaussian_view_mass: float = 1e-3,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """用稀疏 renderer transpose 累积一个 view，避免展开 `[intersection,D]`。"""
    from scipy import sparse

    gids = np.asarray(gaussian_id, dtype=np.int64)
    pixels = np.asarray(pixel_id, dtype=np.int64)
    weights = np.asarray(contribution_weight, dtype=np.float64)
    grid = np.asarray(patch_grid)
    gaussian_count, feature_dimension = state["b1_numerator"].shape
    if gids.ndim != 1 or pixels.shape != gids.shape or weights.shape != gids.shape:
        raise ValueError("streaming intersection arrays 必须一一对齐")
    if grid.ndim != 3 or grid.shape[0] != feature_dimension:
        raise ValueError("streaming patch grid dimension 漂移")
    if np.any((gids < 0) | (gids >= gaussian_count)):
        raise ValueError("streaming gaussian_id 越界")
    pixel_count = int(image_height) * int(image_width)
    if np.any((pixels < 0) | (pixels >= pixel_count)):
        raise ValueError("streaming pixel_id 越界")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("streaming contribution 非 finite 非负")
    for value in (
        minimum_intersection_contribution,
        minimum_gaussian_view_mass,
        epsilon,
    ):
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError("streaming floor/epsilon 必须为有限正数")

    selected = weights >= float(minimum_intersection_contribution)
    selected_gids = gids[selected]
    selected_pixels = pixels[selected]
    selected_weights = weights[selected]
    state["input_intersection_count"] += int(gids.size)
    state["supported_intersection_count"] += int(selected.sum())
    state["processed_view_count"] += 1
    if selected_gids.size == 0:
        return {
            "input_intersection_count": int(gids.size),
            "supported_intersection_count": 0,
            "gaussian_view_count_before_mass_floor": 0,
            "supported_gaussian_view_count": 0,
            "supported_contribution_mass": 0.0,
        }

    # 与 dense align_corners=False interpolation 等价，但只物化每 pixel feature，
    # 不物化可能达到数千万行的 intersection×feature 张量。
    dense_pixel_features = sample_patch_grid_bilinear(
        grid,
        np.arange(pixel_count, dtype=np.int64),
        image_height=int(image_height),
        image_width=int(image_width),
    ).astype(np.float64)
    transpose = sparse.coo_matrix(
        (selected_weights, (selected_gids, selected_pixels)),
        shape=(gaussian_count, pixel_count),
        dtype=np.float64,
    ).tocsr()
    mass = np.asarray(transpose.sum(axis=1)).reshape(-1)
    numerator = np.asarray(transpose @ dense_pixel_features, dtype=np.float64)
    before = mass > 0.0
    keep = mass >= float(minimum_gaussian_view_mass)
    before_count = int(before.sum())
    keep_count = int(keep.sum())
    total_mass = float(mass.sum())
    supported_mass = float(mass[keep].sum())
    state["gaussian_view_count_before_mass_floor"] += before_count
    state["supported_gaussian_view_count"] += keep_count
    state["total_contribution_mass_after_intersection_floor"] += total_mass
    state["supported_contribution_mass"] += supported_mass
    if keep_count:
        keep_ids = np.flatnonzero(keep)
        kept_mass = mass[keep_ids]
        kept_numerator = numerator[keep_ids]
        saturation = -np.expm1(-kept_mass)
        view_feature = kept_numerator / (kept_mass[:, None] + float(epsilon))
        state["b0_numerator"][keep_ids] += saturation[:, None] * view_feature
        state["b0_denominator"][keep_ids] += saturation
        state["b1_numerator"][keep_ids] += kept_numerator
        state["b1_denominator"][keep_ids] += kept_mass
        state["supported_view_count"][keep_ids] += 1
    return {
        "input_intersection_count": int(gids.size),
        "supported_intersection_count": int(selected.sum()),
        "gaussian_view_count_before_mass_floor": before_count,
        "supported_gaussian_view_count": keep_count,
        "supported_contribution_mass": supported_mass,
    }


def finalize_streaming_uplift(
    state: dict[str, Any], *, epsilon: float = 1e-8
) -> dict[str, Any]:
    """完成流式 B0/B1，并保留共同 coverage/denominator。"""
    b0_denominator = np.asarray(state["b0_denominator"], dtype=np.float64)
    b1_denominator = np.asarray(state["b1_denominator"], dtype=np.float64)
    b0 = np.zeros_like(state["b0_numerator"], dtype=np.float64)
    b1 = np.zeros_like(state["b1_numerator"], dtype=np.float64)
    covered = b1_denominator > 0.0
    b0[covered] = state["b0_numerator"][covered] / (
        b0_denominator[covered, None] + float(epsilon)
    )
    b1[covered] = state["b1_numerator"][covered] / (
        b1_denominator[covered, None] + float(epsilon)
    )
    report = {
        "input_intersection_count": int(state["input_intersection_count"]),
        "supported_intersection_count": int(state["supported_intersection_count"]),
        "gaussian_view_count_before_mass_floor": int(
            state["gaussian_view_count_before_mass_floor"]
        ),
        "supported_gaussian_view_count": int(
            state["supported_gaussian_view_count"]
        ),
        "dropped_gaussian_view_count": int(
            state["gaussian_view_count_before_mass_floor"]
            - state["supported_gaussian_view_count"]
        ),
        "covered_gaussian_count": int(covered.sum()),
        "total_contribution_mass_after_intersection_floor": float(
            state["total_contribution_mass_after_intersection_floor"]
        ),
        "supported_contribution_mass": float(
            state["supported_contribution_mass"]
        ),
        "processed_view_count": int(state["processed_view_count"]),
        "accumulator_dtype": "float64",
        "output_dtype": "float32",
        "ludvig_upstream_commit": LUDVIG_UPSTREAM_COMMIT,
        "ludvig_license": LUDVIG_LICENSE,
        "optional_pruning": False,
    }
    return {
        "b0_feature": b0.astype(np.float32),
        "b1_feature": b1.astype(np.float32),
        "b0_denominator": b0_denominator,
        "b1_denominator": b1_denominator,
        "supported_view_count": np.asarray(
            state["supported_view_count"], dtype=np.int32
        ),
        "report": report,
    }
