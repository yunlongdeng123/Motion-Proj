"""V5 geometry-first repair 的局部曲面与分阶段几何审计。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree


SURFACE_MODELS = ("G0_ROBUST_PLANE", "G3_ROBUST_QUADRATIC")
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
    if model == "G0_ROBUST_PLANE":
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
    required = max(int(minimum_support_points), 3 if model == SURFACE_MODELS[0] else 6)
    count = int(valid_support.sum())
    if count < required:
        raise GeometryRepairError(f"ABSTAIN_INSUFFICIENT_GEOMETRY_SUPPORT:{count}<{required}")
    if not target.any():
        raise GeometryRepairError("ABSTAIN_EMPTY_TARGET_MASK")

    x, y = _normalized_coordinates(values.shape, intrinsics)
    design = _design_matrix(x[valid_support], y[valid_support], model)
    inverse_depth = 1.0 / values[valid_support]
    weights = np.ones(count, dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    condition = float("nan")
    iterations = 0
    for iterations in range(1, int(maximum_iterations) + 1):
        updated, condition = _weighted_lstsq(design, inverse_depth, weights, ridge)
        residual = design @ updated - inverse_depth
        scale = max(1.4826 * float(np.median(np.abs(residual))), 1e-8)
        normalized = np.abs(residual) / (float(huber_delta) * scale)
        new_weights = np.ones_like(normalized)
        tail = normalized > 1.0
        new_weights[tail] = 1.0 / normalized[tail]
        if np.allclose(updated, coefficients, rtol=0.0, atol=1e-12):
            coefficients = updated
            weights = new_weights
            break
        coefficients = updated
        weights = new_weights

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
    inverse_residual = np.abs(design @ coefficients - inverse_depth)
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
