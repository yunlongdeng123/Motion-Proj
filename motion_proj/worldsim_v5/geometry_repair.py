"""V5 geometry-first repair 的局部曲面与分阶段几何审计。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree


SURFACE_MODELS = (
    "G0_ROBUST_PLANE",
    "G1_PIECEWISE_PLANE",
    "G3_ROBUST_QUADRATIC",
)
RISK_MAPPINGS = ("R0_CLIP", "R1_LOG", "R2_PSEUDO_HUBER", "R3_QUANTILE", "R4_CALIBRATED_LOG")


class GeometryRepairError(ValueError):
    """输入或几何支持不满足冻结合约。"""


@dataclass(frozen=True)
class SurfaceFitResult:
    """相机归一化平面上的 inverse-depth 曲面拟合结果。"""

    model: str
    depth: np.ndarray
    valid: np.ndarray
    coefficients: np.ndarray
    support_count: int
    iterations: int
    inverse_depth_residual_median: float
    inverse_depth_residual_p90: float
    surface_fit_residual_median_m: float
    surface_fit_residual_p90_m: float
    extrapolation_distance_mean: float
    extrapolation_distance_p95: float
    condition_number: float

    def audit(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "coefficients": self.coefficients.tolist(),
            "support_count": int(self.support_count),
            "iterations": int(self.iterations),
            "inverse_depth_residual_median": float(
                self.inverse_depth_residual_median
            ),
            "inverse_depth_residual_p90": float(self.inverse_depth_residual_p90),
            "surface_fit_residual_median_m": float(
                self.surface_fit_residual_median_m
            ),
            "surface_fit_residual_p90_m": float(self.surface_fit_residual_p90_m),
            "extrapolation_distance_mean": float(self.extrapolation_distance_mean),
            "extrapolation_distance_p95": float(self.extrapolation_distance_p95),
            "condition_number": float(self.condition_number),
            "predicted_pixel_count": int(self.valid.sum()),
        }


def _normalized_coordinates(shape: tuple[int, int], intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise GeometryRepairError("intrinsics 必须为有限 3x3 矩阵")
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        raise GeometryRepairError("intrinsics focal length 必须为正")
    rows, columns = np.indices(shape, dtype=np.float64)
    x = (columns - float(matrix[0, 2])) / fx
    y = (rows - float(matrix[1, 2])) / fy
    return x, y


def _design_matrix(x: np.ndarray, y: np.ndarray, model: str) -> np.ndarray:
    if model in {"G0_ROBUST_PLANE", "G1_PIECEWISE_PLANE"}:
        return np.column_stack((np.ones(x.size), x, y))
    if model == "G3_ROBUST_QUADRATIC":
        return np.column_stack((np.ones(x.size), x, y, x * x, x * y, y * y))
    raise GeometryRepairError(f"未冻结的 surface model: {model}")


def _weighted_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float]:
    root = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted = design * root[:, None]
    normal = weighted.T @ weighted
    if ridge > 0.0:
        normal = normal + np.eye(normal.shape[0], dtype=np.float64) * float(ridge)
    right = weighted.T @ (target * root)
    condition = float(np.linalg.cond(normal))
    if not math.isfinite(condition) or condition > 1e14:
        raise GeometryRepairError(f"surface normal matrix 退化: condition={condition}")
    return np.linalg.solve(normal, right), condition


def _robust_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    *,
    huber_delta: float,
    maximum_iterations: int,
    ridge: float,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    weights = np.ones(target.size, dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    condition = float("nan")
    iterations = 0
    for iterations in range(1, int(maximum_iterations) + 1):
        updated, condition = _weighted_lstsq(design, target, weights, ridge)
        residual = design @ updated - target
        scale = max(1.4826 * float(np.median(np.abs(residual))), 1e-8)
        normalized = np.abs(residual) / (float(huber_delta) * scale)
        new_weights = np.ones_like(normalized)
        tail = normalized > 1.0
        new_weights[tail] = 1.0 / normalized[tail]
        converged = np.allclose(updated, coefficients, rtol=0.0, atol=1e-12)
        coefficients = updated
        weights = new_weights
        if converged:
            break
    return coefficients, iterations, condition, design @ coefficients - target


def _piece_labels(target: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(target.shape, dtype=np.float64)
    target_rows, target_columns = np.nonzero(target)
    center_row = float(np.mean(target_rows))
    center_column = float(np.mean(target_columns))
    row_scale = max(float(target_rows.max() - target_rows.min() + 1), 1.0)
    column_scale = max(float(target_columns.max() - target_columns.min() + 1), 1.0)
    dy = (rows - center_row) / row_scale
    dx = (columns - center_column) / column_scale
    # 固定顺序为 top/right/bottom/left；argmax 在平局时确定性选更早的 side。
    return np.argmax(np.stack((-dy, dx, dy, -dx), axis=-1), axis=-1).astype(np.int8)


def _fit_piecewise_surface(
    *,
    values: np.ndarray,
    valid_support: np.ndarray,
    target: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    minimum_support_points: int,
    huber_delta: float,
    maximum_iterations: int,
    ridge: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> SurfaceFitResult:
    labels = _piece_labels(target)
    support_design = _design_matrix(
        x[valid_support], y[valid_support], "G0_ROBUST_PLANE"
    )
    inverse_depth = 1.0 / values[valid_support]
    global_fit = _robust_lstsq(
        support_design,
        inverse_depth,
        huber_delta=huber_delta,
        maximum_iterations=maximum_iterations,
        ridge=ridge,
    )
    coefficients: list[np.ndarray] = []
    iterations: list[int] = []
    conditions: list[float] = []
    minimum_piece_points = max(16, int(minimum_support_points) // 8)
    for piece in range(4):
        selected = valid_support & (labels == piece)
        if int(selected.sum()) < minimum_piece_points:
            coefficients.append(global_fit[0].copy())
            iterations.append(global_fit[1])
            conditions.append(global_fit[2])
            continue
        fit = _robust_lstsq(
            _design_matrix(x[selected], y[selected], "G0_ROBUST_PLANE"),
            1.0 / values[selected],
            huber_delta=huber_delta,
            maximum_iterations=maximum_iterations,
            ridge=ridge,
        )
        coefficients.append(fit[0])
        iterations.append(fit[1])
        conditions.append(fit[2])
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    support_labels = labels[valid_support]
    support_prediction = np.empty(inverse_depth.shape, dtype=np.float64)
    for piece in range(4):
        selected = support_labels == piece
        support_prediction[selected] = (
            support_design[selected] @ coefficient_array[piece]
        )
    support_valid = support_prediction > 1.0 / float(maximum_depth_m)
    metric_residual = np.full(inverse_depth.size, np.nan, dtype=np.float64)
    metric_residual[support_valid] = np.abs(
        1.0 / support_prediction[support_valid]
        - values[valid_support][support_valid]
    )
    finite_metric = metric_residual[np.isfinite(metric_residual)]
    if finite_metric.size == 0:
        raise GeometryRepairError("ABSTAIN_INVALID_PIECEWISE_SURFACE")

    target_labels = labels[target]
    target_design = _design_matrix(x[target], y[target], "G0_ROBUST_PLANE")
    target_inverse_depth = np.empty(target_design.shape[0], dtype=np.float64)
    for piece in range(4):
        selected = target_labels == piece
        target_inverse_depth[selected] = target_design[selected] @ coefficient_array[piece]
    target_depth = np.full(values.shape, np.nan, dtype=np.float32)
    target_indices = np.flatnonzero(target)
    allowed = (
        np.isfinite(target_inverse_depth)
        & (target_inverse_depth >= 1.0 / float(maximum_depth_m))
        & (target_inverse_depth <= 1.0 / float(minimum_depth_m))
    )
    target_depth.reshape(-1)[target_indices[allowed]] = (
        1.0 / target_inverse_depth[allowed]
    ).astype(np.float32)
    predicted_valid = np.isfinite(target_depth) & target
    if not predicted_valid.any():
        raise GeometryRepairError("ABSTAIN_NO_VALID_PIECEWISE_TARGET_SURFACE")
    support_xy = np.column_stack((x[valid_support], y[valid_support]))
    target_xy = np.column_stack((x[predicted_valid], y[predicted_valid]))
    distance, _ = cKDTree(support_xy).query(target_xy, k=1, workers=1)
    inverse_residual = np.abs(support_prediction - inverse_depth)
    return SurfaceFitResult(
        model="G1_PIECEWISE_PLANE",
        depth=target_depth,
        valid=predicted_valid,
        coefficients=coefficient_array,
        support_count=int(valid_support.sum()),
        iterations=max(iterations),
        inverse_depth_residual_median=float(np.median(inverse_residual)),
        inverse_depth_residual_p90=float(np.quantile(inverse_residual, 0.90)),
        surface_fit_residual_median_m=float(np.median(finite_metric)),
        surface_fit_residual_p90_m=float(np.quantile(finite_metric, 0.90)),
        extrapolation_distance_mean=float(np.mean(distance)),
        extrapolation_distance_p95=float(np.quantile(distance, 0.95)),
        condition_number=max(conditions),
    )


def fit_inverse_depth_surface(
    *,
    depth: np.ndarray,
    support_mask: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: np.ndarray,
    model: str = "G0_ROBUST_PLANE",
    minimum_support_points: int = 64,
    huber_delta: float = 1.5,
    maximum_iterations: int = 12,
    ridge: float = 1e-10,
    minimum_depth_m: float = 0.1,
    maximum_depth_m: float = 120.0,
) -> SurfaceFitResult:
    """用冻结 IRLS 拟合 inverse-depth；G0 对应真实 3D plane。"""

    values = np.asarray(depth, dtype=np.float64)
    support = np.asarray(support_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if values.ndim != 2 or support.shape != values.shape or target.shape != values.shape:
        raise GeometryRepairError("depth/support/target shape 不一致")
    if model not in SURFACE_MODELS:
        raise GeometryRepairError(f"surface model 不在冻结集合: {model}")
    valid_support = (
        support
        & np.isfinite(values)
        & (values >= float(minimum_depth_m))
        & (values <= float(maximum_depth_m))
    )
    required = max(
        int(minimum_support_points),
        3 if model in {"G0_ROBUST_PLANE", "G1_PIECEWISE_PLANE"} else 6,
    )
    count = int(valid_support.sum())
    if count < required:
        raise GeometryRepairError(f"ABSTAIN_INSUFFICIENT_GEOMETRY_SUPPORT:{count}<{required}")
    if not target.any():
        raise GeometryRepairError("ABSTAIN_EMPTY_TARGET_MASK")

    x, y = _normalized_coordinates(values.shape, intrinsics)
    if model == "G1_PIECEWISE_PLANE":
        return _fit_piecewise_surface(
            values=values,
            valid_support=valid_support,
            target=target,
            x=x,
            y=y,
            minimum_support_points=minimum_support_points,
            huber_delta=huber_delta,
            maximum_iterations=maximum_iterations,
            ridge=ridge,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
        )
    design = _design_matrix(x[valid_support], y[valid_support], model)
    inverse_depth = 1.0 / values[valid_support]
    coefficients, iterations, condition, residual = _robust_lstsq(
        design,
        inverse_depth,
        huber_delta=huber_delta,
        maximum_iterations=maximum_iterations,
        ridge=ridge,
    )

    support_prediction = design @ coefficients
    support_valid = support_prediction > 1.0 / float(maximum_depth_m)
    metric_residual = np.full(count, np.nan, dtype=np.float64)
    metric_residual[support_valid] = np.abs(
        1.0 / support_prediction[support_valid] - values[valid_support][support_valid]
    )
    finite_metric = metric_residual[np.isfinite(metric_residual)]
    if finite_metric.size == 0:
        raise GeometryRepairError("ABSTAIN_INVALID_SURFACE_PREDICTION")

    target_design = _design_matrix(x[target], y[target], model)
    target_inverse_depth = target_design @ coefficients
    target_depth = np.full(values.shape, np.nan, dtype=np.float32)
    target_indices = np.flatnonzero(target)
    allowed = (
        np.isfinite(target_inverse_depth)
        & (target_inverse_depth >= 1.0 / float(maximum_depth_m))
        & (target_inverse_depth <= 1.0 / float(minimum_depth_m))
    )
    flat_depth = target_depth.reshape(-1)
    flat_depth[target_indices[allowed]] = (1.0 / target_inverse_depth[allowed]).astype(
        np.float32
    )
    predicted_valid = np.isfinite(target_depth) & target
    if not predicted_valid.any():
        raise GeometryRepairError("ABSTAIN_NO_VALID_TARGET_SURFACE")

    support_xy = np.column_stack((x[valid_support], y[valid_support]))
    target_xy = np.column_stack((x[predicted_valid], y[predicted_valid]))
    distance, _ = cKDTree(support_xy).query(target_xy, k=1, workers=1)
    inverse_residual = np.abs(residual)
    return SurfaceFitResult(
        model=model,
        depth=target_depth,
        valid=predicted_valid,
        coefficients=coefficients.astype(np.float64),
        support_count=count,
        iterations=iterations,
        inverse_depth_residual_median=float(np.median(inverse_residual)),
        inverse_depth_residual_p90=float(np.quantile(inverse_residual, 0.90)),
        surface_fit_residual_median_m=float(np.median(finite_metric)),
        surface_fit_residual_p90_m=float(np.quantile(finite_metric, 0.90)),
        extrapolation_distance_mean=float(np.mean(distance)),
        extrapolation_distance_p95=float(np.quantile(distance, 0.95)),
        condition_number=condition,
    )


def depth_error_summary(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    selected = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(left)
        & np.isfinite(right)
        & (left > 1e-4)
        & (right > 1e-4)
    )
    count = int(selected.sum())
    if count == 0:
        raise GeometryRepairError("ABSTAIN_NO_REFERENCE_OVERLAP")
    absolute = np.abs(left[selected] - right[selected])
    relative = absolute / np.maximum(right[selected], 1e-4)
    return {
        "pixel_count": count,
        "mae_m": float(np.mean(absolute)),
        "median_m": float(np.median(absolute)),
        "relative_mean": float(np.mean(relative)),
        "p90_m": float(np.quantile(absolute, 0.90)),
        "p95_m": float(np.quantile(absolute, 0.95)),
    }


def staged_geometry_metrics(
    *,
    raw_surface_depth: np.ndarray,
    pre_gaussianization_depth: np.ndarray,
    post_gaussianization_render_depth: np.ndarray,
    reference_depth: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, Any]:
    raw = depth_error_summary(raw_surface_depth, reference_depth, evaluation_mask)
    pre = depth_error_summary(
        pre_gaussianization_depth, reference_depth, evaluation_mask
    )
    post = depth_error_summary(
        post_gaussianization_render_depth, reference_depth, evaluation_mask
    )
    representation = depth_error_summary(
        post_gaussianization_render_depth,
        pre_gaussianization_depth,
        evaluation_mask,
    )
    return {
        "raw_geometry_error": raw,
        "pre_gaussianization_geometry_error": pre,
        "post_gaussianization_render_error": post,
        "representation_gap": representation,
        "gaussianization_delta_mae_m": float(post["mae_m"] - pre["mae_m"]),
    }


def geometry_reference_confidence(
    *,
    observed_reference_pixels: int,
    target_pixels: int,
    lidar_agreement_mae_m: float | None,
    agreement_scale_m: float,
) -> float:
    if target_pixels <= 0 or observed_reference_pixels < 0 or agreement_scale_m <= 0.0:
        raise GeometryRepairError("reference confidence 参数非法")
    coverage = min(float(observed_reference_pixels) / float(target_pixels), 1.0)
    if lidar_agreement_mae_m is None or not math.isfinite(lidar_agreement_mae_m):
        agreement = 0.5
    else:
        agreement = math.exp(-max(float(lidar_agreement_mae_m), 0.0) / agreement_scale_m)
    return float(np.clip(math.sqrt(coverage * agreement), 0.0, 1.0))


def geometry_risk_mapping(
    errors_m: np.ndarray,
    *,
    mapping: str,
    scale_m: float,
    calibration_errors_m: np.ndarray | None = None,
) -> np.ndarray:
    """Development-only 风险映射；R1/R2 保留 clipping 丢失的坏尾排序。"""

    values = np.asarray(errors_m, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise GeometryRepairError("geometry errors 必须是一维有限非负数组")
    if mapping not in RISK_MAPPINGS or scale_m <= 0.0:
        raise GeometryRepairError("geometry risk mapping/scale 非法")
    normalized = values / float(scale_m)
    if mapping == "R0_CLIP":
        result = np.minimum(normalized, 1.0)
    elif mapping == "R1_LOG":
        result = np.log1p(normalized)
    elif mapping == "R2_PSEUDO_HUBER":
        result = np.sqrt(1.0 + normalized * normalized) - 1.0
    else:
        if calibration_errors_m is None:
            raise GeometryRepairError(f"{mapping} 需要 development calibration errors")
        calibration = np.sort(np.asarray(calibration_errors_m, dtype=np.float64))
        if (
            calibration.ndim != 1
            or calibration.size < 2
            or not np.isfinite(calibration).all()
            or np.any(calibration < 0.0)
        ):
            raise GeometryRepairError("calibration errors 非法")
        if mapping == "R3_QUANTILE":
            ranks = np.arange(1, calibration.size + 1, dtype=np.float64) / calibration.size
            result = np.interp(values, calibration, ranks, left=0.0, right=1.0)
        else:
            denominator = max(float(np.log1p(calibration[-1] / scale_m)), 1e-12)
            result = np.log1p(normalized) / denominator
    return np.asarray(result, dtype=np.float64)
