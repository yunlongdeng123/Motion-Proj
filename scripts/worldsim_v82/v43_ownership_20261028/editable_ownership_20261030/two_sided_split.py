"""独立 CPU 原型：给定 3D 平面的双侧截断高斯矩匹配初始化。

输入 (mu, Sigma, n, d) -> 一维正态截断 -> 两个 moment-matched 子高斯
                                      -> mass / opacity 候选 / lineage

本模块不寻找语义边界，不验证输入切面，不分类车/路，不批量拆分 R2，
不调用 native renderer。它不是 EVSplitting 的完整实现；论文还包含其
特定的边界溢出约束与优化目标。本模块只实现下述截断正态公式。
参考背景：https://orca.cardiff.ac.uk/174775/1/3680528.3687592.pdf

令 Y = n_hat.T X，s^2 = n_hat.T Sigma n_hat，a = (d_hat-n_hat.T mu)/s，
v = Sigma n_hat / s，phi 和 Phi 分别为标准正态 PDF 和 CDF。则：

  p_left = Phi(a),  lambda_left = phi(a) / p_left
  mu_left = mu - v lambda_left
  Sigma_left = Sigma - vv.T lambda_left (a + lambda_left)

  p_right = Phi(-a), lambda_right = phi(a) / p_right
  mu_right = mu + v lambda_right
  Sigma_right = Sigma + vv.T lambda_right (a - lambda_right)

用 p_left/right 混合两个子高斯，恰好保持归一化分布的 0/1/2 阶矩，
但不是原截断密度的精确表示：子高斯仍越过切面，像素 alpha 和多层
alpha compositing 没有精确保真保证。安全范围内的 opacity 也需另做
native 渲染验收，不能把本模块的数学测试当作渲染验收。

peak opacity 与归一化质量不同。省略公共常数 (2*pi)^(3/2) 后，
parent_mass = parent_opacity * sqrt(det(Sigma))；若保持这些质量，
child_peak = parent_opacity * p_child * sqrt(det(Sigma)/det(Sigma_child))。
本模块返回未裁剪的 opacity_candidate，超过 1 时显式标记，绝不
静默 clip，也不修改 covariance 迁就 native opacity 范围。

依赖仅 NumPy 和 Python 标准库。scale 是正的线性标准差，quat 为
原生 wxyz；不是 log-scale，也不是 xyzw。所有计算使用 float64。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import math
from typing import Any, Iterable

import numpy as np


METHOD = "two_sided_truncated_gaussian_moment_match"
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_EPS = np.finfo(np.float64).eps


class NumericalSplitError(ValueError):
    """float64 无法可靠表示输入或某个子高斯；不偷偷删侧或正则化。"""


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result.copy()


def _covariance(value: Any, name: str = "covariance") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite (3, 3) matrix")
    scale = float(np.max(np.abs(result)))
    if scale == 0.0 or float(np.max(np.abs(result - result.T))) > 1e-12 * scale:
        raise ValueError(f"{name} must be symmetric and positive definite")
    # 只消除已通过相对对称检查的浮点舍入；不加 jitter、不修复负特征值。
    result = 0.5 * result + 0.5 * result.T
    try:
        eigenvalues = np.linalg.eigvalsh(result)
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{name} must be positive definite")
    if eigenvalues[0] <= 1e-14 * eigenvalues[-1]:
        raise NumericalSplitError(f"{name} is numerically singular (condition >= 1e14)")
    return result


def _quaternion(value: Any, name: str = "quaternion_wxyz") -> np.ndarray:
    result = _vector(value, 4, name)
    length = math.hypot(*result)
    if length <= 1e-12 or not math.isfinite(length):
        raise ValueError(f"{name} has a zero, near-zero, or unrepresentable norm")
    return result / length


def quat_wxyz_to_rotation(quaternion_wxyz: Any) -> np.ndarray:
    """将 wxyz 四元数归一化后转为主动旋转矩阵；列对应 Gaussian 轴。"""
    w, x, y, z = _quaternion(quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float64,
    )


def quat_wxyz_scale_to_cov(quaternion_wxyz: Any, scales: Any) -> np.ndarray:
    """Sigma = R diag(scales**2) R.T；拒绝非正、非有限 scale。"""
    scales = _vector(scales, 3, "scales")
    if np.any(scales <= 0.0):
        raise ValueError("scales must be strictly positive linear standard deviations")
    root = quat_wxyz_to_rotation(quaternion_wxyz) * scales[np.newaxis, :]
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = root @ root.T
    return _covariance(covariance)


def _rotation_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """已知右手正交旋转矩阵的稳定四元数转换。"""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 2.0 * math.sqrt(1.0 + trace)
        quat = np.array([s/4, (rotation[2, 1]-rotation[1, 2])/s,
                         (rotation[0, 2]-rotation[2, 0])/s,
                         (rotation[1, 0]-rotation[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(0.0, 1 + rotation[i, i] - rotation[j, j] - rotation[k, k]))
        quat = np.empty(4, dtype=np.float64)
        quat[0] = (rotation[k, j] - rotation[j, k]) / s
        quat[i + 1] = s / 4.0
        quat[j + 1] = (rotation[j, i] + rotation[i, j]) / s
        quat[k + 1] = (rotation[k, i] + rotation[i, k]) / s
    return quat / math.hypot(*quat)


def _canonical_eigenbasis(eigenvalues: np.ndarray, basis: np.ndarray,
                          reference: np.ndarray | None) -> np.ndarray:
    """重复特征值用子空间投影定基；其它轴用明确的最大分量定符号。"""
    basis = basis.copy()
    start = 0
    while start < 3:
        stop = start + 1
        while stop < 3 and abs(eigenvalues[stop] - eigenvalues[start]) <= 8*_EPS*eigenvalues[0]:
            stop += 1
        if stop - start > 1:
            subspace = basis[:, start:stop]
            projector = subspace @ subspace.T
            chosen: list[np.ndarray] = []
            targets = np.eye(3) if reference is None else reference
            # reference 只在数值重根子空间内定基，不改变非退化轴的特征值。
            for target in targets.T:
                candidate = projector @ target
                for previous in chosen:
                    candidate -= previous * float(previous @ candidate)
                length = float(np.linalg.norm(candidate))
                if length > 1e-10:
                    chosen.append(candidate / length)
                if len(chosen) == stop - start:
                    break
            basis[:, start:stop] = np.column_stack(chosen)
        elif reference is None:
            column = basis[:, start]
            if column[int(np.argmax(np.abs(column)))] < 0:
                basis[:, start] *= -1.0
        start = stop
    return basis


def cov_to_quat_wxyz_scale(covariance: Any, *, reference_quaternion_wxyz: Any = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """子 covariance -> (wxyz quaternion, positive linear scales)。

    默认 scale 按降序，特征向量最大绝对分量为正，最后一轴必要时反向以
    保持右手系，quat 首个显著分量为正。重复特征值用世界轴子空间投影定基。
    提供 reference 时，从等价的 24 个右手轴排列中选最接近 reference 的
    旋转，并与 reference 的 quat 同号，减少符号和轴排列跳变。此时 scales
    顺序跟随该旋转，不保证降序。重根处旋转本来不唯一；数值重根阈值为
    8*eps*最大特征值，分解往返误差仍需按浮点容差验收。
    """
    covariance = _covariance(covariance)
    eigenvalues, basis = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues, basis = eigenvalues[order], basis[:, order]
    reference_quat = None
    reference = None
    if reference_quaternion_wxyz is not None:
        reference_quat = _quaternion(reference_quaternion_wxyz, "reference_quaternion_wxyz")
        reference = quat_wxyz_to_rotation(reference_quat)
    basis = _canonical_eigenbasis(eigenvalues, basis, reference)
    if reference is None:
        if np.linalg.det(basis) < 0:
            basis[:, -1] *= -1.0
    else:
        best_score = -math.inf
        best_basis, best_values = None, None
        for permutation in permutations(range(3)):
            for signs in product((-1.0, 1.0), repeat=3):
                candidate = basis[:, permutation] * np.asarray(signs)[np.newaxis, :]
                if np.linalg.det(candidate) <= 0:
                    continue
                score = float(np.sum(candidate * reference))
                if score > best_score + 8*_EPS:
                    best_score = score
                    best_basis = candidate
                    best_values = eigenvalues[list(permutation)]
        basis, eigenvalues = best_basis, best_values
    quaternion = _rotation_to_quat_wxyz(basis)
    if reference_quat is not None:
        if float(quaternion @ reference_quat) < 0:
            quaternion *= -1.0
    else:
        first = next(value for value in quaternion if abs(value) > 1e-14)
        if first < 0:
            quaternion *= -1.0
    return quaternion, np.sqrt(eigenvalues)


def _tail_statistics(t: float) -> tuple[float, float, float]:
    """t>=0 时返回 log Q(t)、E[Z|Z>t]、Var[Z|Z>t]。

    t>=5 用 Laplace 连分式 lambda=t+1/(t+2/(t+3/...))，同时求导，
    由 Var=-d(lambda-t)/dt 避免 1+t*lambda-lambda**2 的灾难相消。
    """
    log_phi = -0.5 * t * t - _LOG_SQRT_2PI
    if not math.isfinite(log_phi):
        raise NumericalSplitError("standardized plane offset is too large for float64 log-mass")
    if t < 5.0:
        log_mass = math.log(0.5 * math.erfc(t / _SQRT_2))
        inverse_mills = math.exp(log_phi - log_mass)
        variance = 1 + t*inverse_mills - inverse_mills*inverse_mills
    else:
        remainder, derivative = 0.0, 0.0
        for k in range(128, 0, -1):
            denominator = t + remainder
            next_derivative = -k * (1 + derivative) / (denominator * denominator)
            remainder = k / denominator
            derivative = next_derivative
        inverse_mills = t + remainder
        variance = -derivative
        log_mass = log_phi - math.log(inverse_mills)
    if variance <= 0 or not math.isfinite(variance):
        raise NumericalSplitError("conditional tail variance is not representable")
    return log_mass, inverse_mills, variance


def _exp_or_inf(log_value: float) -> float:
    if log_value == -math.inf:
        return 0.0
    if log_value > math.log(np.finfo(np.float64).max):
        return math.inf
    return math.exp(log_value)


def _json_floats(values: dict[str, float]) -> dict[str, Any]:
    """严格 JSON 用 null 表示非有限数，并保留明确的符号说明。"""
    result = {key: value if math.isfinite(value) else None for key, value in values.items()}
    nonfinite = {key: str(value) for key, value in values.items() if not math.isfinite(value)}
    if nonfinite:
        result["nonfinite_values"] = nonfinite
    return result


@dataclass(frozen=True)
class GaussianChild:
    child_id: str
    parent_id: str
    split_id: str
    side: str
    mean: np.ndarray
    covariance: np.ndarray
    mass_weight: float
    log_mass_weight: float
    reduced_mass: float
    log_reduced_mass: float
    opacity_candidate: float
    log_opacity_candidate: float
    opacity_candidate_safe: bool
    needs_review: bool
    review_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id, "parent_id": self.parent_id,
            "split_id": self.split_id, "side": self.side,
            "mean": self.mean.tolist(), "covariance": self.covariance.tolist(),
            **_json_floats({"mass_weight": self.mass_weight, "log_mass_weight": self.log_mass_weight,
                           "reduced_mass": self.reduced_mass, "log_reduced_mass": self.log_reduced_mass,
                           "opacity_candidate": self.opacity_candidate,
                           "log_opacity_candidate": self.log_opacity_candidate}),
            "opacity_candidate_safe": self.opacity_candidate_safe,
            "needs_review": self.needs_review, "review_reasons": list(self.review_reasons),
            "semantic_owner": None,
        }


@dataclass(frozen=True)
class TwoSidedSplit:
    parent_id: str
    split_id: str
    parent_mean: np.ndarray
    parent_covariance: np.ndarray
    parent_opacity: float
    parent_reduced_mass: float
    log_parent_reduced_mass: float
    plane_normal: np.ndarray
    plane_offset: float
    standardized_offset: float
    children: tuple[GaussianChild, GaussianChild]

    @property
    def mass_weights(self) -> np.ndarray:
        return np.asarray([child.mass_weight for child in self.children])

    @property
    def opacity_candidates(self) -> np.ndarray:
        return np.asarray([child.opacity_candidate for child in self.children])

    @property
    def needs_review(self) -> bool:
        return any(child.needs_review for child in self.children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": METHOD, "initialization_only": True,
            "exact_pixel_alpha_conservation": False,
            "semantic_boundary_discovery": False,
            "complete_evsplitting_implementation": False,
            "parent_id": self.parent_id, "split_id": self.split_id,
            "parent_mean": self.parent_mean.tolist(),
            "parent_covariance": self.parent_covariance.tolist(),
            "parent_opacity": self.parent_opacity,
            **_json_floats({"parent_reduced_mass": self.parent_reduced_mass,
                           "log_parent_reduced_mass": self.log_parent_reduced_mass}),
            "plane_normal_unit": self.plane_normal.tolist(),
            "plane_offset_unit": self.plane_offset,
            "standardized_offset": self.standardized_offset,
            "mass_weights": self.mass_weights.tolist(),
            "opacity_candidates": self.opacity_candidates.tolist(),
            "needs_review": self.needs_review,
            "lineage": {"parent_id": self.parent_id, "split_id": self.split_id,
                        "child_ids": [child.child_id for child in self.children]},
            "children": [child.to_dict() for child in self.children],
        }


def split_gaussian_two_sided(mean: Any, covariance: Any, plane_normal: Any,
                             plane_offset: float, *, parent_id: str,
                             parent_opacity: float = 1.0, split_id: str = "plane_split",
                             tail_review_threshold: float = 1e-8,
                             normal_min_norm: float = 1e-12) -> TwoSidedSplit:
    """对 n.T x=d 的 leq/gt 两侧各返回一个矩匹配子高斯，绝不删侧。

    极小 tail 仍返回 child 与 log_mass_weight；float64 质量下溢时
    mass_weight=0 并显式标记 mass_weight_underflow，不能据此删 child。
    非法或数值奇异输入抛 ValueError/NumericalSplitError，不自动修复。
    调用者负责确认三维切面和逐 child 语义归属；不能自动应用至 R2 集合。
    """
    mean = _vector(mean, 3, "mean")
    covariance = _covariance(covariance)
    normal = _vector(plane_normal, 3, "plane_normal")
    normal_min_norm = float(normal_min_norm)
    if not math.isfinite(normal_min_norm) or normal_min_norm <= 0:
        raise ValueError("normal_min_norm must be finite and positive")
    normal_length = math.hypot(*normal)
    if not math.isfinite(normal_length) or normal_length <= normal_min_norm:
        raise ValueError("plane_normal has a zero, near-zero, or unrepresentable norm")
    offset = float(plane_offset)
    opacity = float(parent_opacity)
    threshold = float(tail_review_threshold)
    if not math.isfinite(offset):
        raise ValueError("plane_offset must be finite")
    if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("parent_opacity must be finite and in [0, 1]")
    if not math.isfinite(threshold) or not 0.0 < threshold < 0.5:
        raise ValueError("tail_review_threshold must be in (0, 0.5)")
    if not isinstance(parent_id, str) or not parent_id.strip():
        raise ValueError("parent_id must be a nonempty string")
    if not isinstance(split_id, str) or not split_id.strip():
        raise ValueError("split_id must be a nonempty string")
    normal, offset = normal / normal_length, offset / normal_length
    root = np.linalg.cholesky(covariance)
    projected = root.T @ normal
    stddev = math.hypot(*projected)
    if not math.isfinite(offset) or stddev <= 0 or not math.isfinite(stddev):
        raise NumericalSplitError("normalized plane or projected standard deviation is unrepresentable")
    a = (offset - float(normal @ mean)) / stddev
    if not math.isfinite(a):
        raise NumericalSplitError("standardized plane offset is unrepresentable")
    log_tail, tail_mean, tail_variance = _tail_statistics(abs(a))
    tail_mass = math.exp(log_tail)
    log_bulk = math.log1p(-tail_mass)
    bulk_mass = -math.expm1(log_tail)
    bulk_mean = math.exp(-0.5*a*a - _LOG_SQRT_2PI - log_bulk)
    bulk_variance = 1 - abs(a)*bulk_mean - bulk_mean*bulk_mean
    if bulk_variance <= 0:
        raise NumericalSplitError("conditional bulk variance is not representable")
    tail_stats = (tail_mass, log_tail, tail_mean, tail_variance)
    bulk_stats = (bulk_mass, log_bulk, bulk_mean, bulk_variance)
    left, right = (bulk_stats, tail_stats) if a >= 0 else (tail_stats, bulk_stats)

    # 在 whitened 坐标中构建 root；比 Sigma-vv.T 的大数相消更稳定。
    direction = projected / stddev
    tangent = np.eye(3)[:, int(np.argmin(np.abs(direction)))]
    tangent = tangent - direction * float(direction @ tangent)
    tangent /= np.linalg.norm(tangent)
    frame = np.column_stack((direction, tangent, np.cross(direction, tangent)))
    axis_root = root @ frame
    shift_axis = root @ direction
    parent_logdet = float(np.linalg.slogdet(covariance)[1])
    log_opacity = math.log(opacity) if opacity > 0 else -math.inf
    log_parent_mass = log_opacity + 0.5*parent_logdet
    children: list[GaussianChild] = []
    for side, sign, statistics in (("leq", -1.0, left), ("gt", 1.0, right)):
        mass, log_mass, conditional_mean, conditional_variance = statistics
        child_root = axis_root.copy()
        child_root[:, 0] *= math.sqrt(conditional_variance)
        child_covariance = _covariance(child_root @ child_root.T, f"{side} child covariance")
        child_mean = mean + sign*shift_axis*conditional_mean
        if not np.all(np.isfinite(child_mean)):
            raise NumericalSplitError(f"{side} child mean is unrepresentable")
        child_logdet = float(np.linalg.slogdet(child_covariance)[1])
        log_candidate = log_opacity + log_mass + 0.5*(parent_logdet-child_logdet)
        candidate = _exp_or_inf(log_candidate)
        log_reduced_mass = log_parent_mass + log_mass
        reduced_mass = _exp_or_inf(log_reduced_mass)
        reasons = []
        if log_mass < math.log(threshold):
            reasons.append("tiny_tail_mass")
        if mass == 0:
            reasons.append("mass_weight_underflow_log_mass_retained")
        if not math.isfinite(candidate) or candidate > 1.0:
            reasons.append("unsafe_peak_opacity_candidate_not_clipped")
        if opacity > 0 and candidate == 0:
            reasons.append("opacity_candidate_underflow_log_candidate_retained")
        if not math.isfinite(reduced_mass):
            reasons.append("reduced_mass_overflow_log_mass_retained")
        elif opacity > 0 and reduced_mass == 0:
            reasons.append("reduced_mass_underflow_log_mass_retained")
        children.append(GaussianChild(
            child_id=f"{parent_id}/{split_id}/{side}", parent_id=parent_id,
            split_id=split_id, side=side, mean=child_mean, covariance=child_covariance,
            mass_weight=mass, log_mass_weight=log_mass, reduced_mass=reduced_mass,
            log_reduced_mass=log_reduced_mass, opacity_candidate=candidate,
            log_opacity_candidate=log_candidate,
            opacity_candidate_safe=math.isfinite(candidate) and 0 <= candidate <= 1,
            needs_review=bool(reasons), review_reasons=tuple(reasons),
        ))
    return TwoSidedSplit(
        parent_id=parent_id, split_id=split_id, parent_mean=mean,
        parent_covariance=covariance, parent_opacity=opacity,
        parent_reduced_mass=_exp_or_inf(log_parent_mass),
        log_parent_reduced_mass=log_parent_mass, plane_normal=normal,
        plane_offset=offset, standardized_offset=a, children=(children[0], children[1]),
    )


def raw_mixture_moments(children: Iterable[GaussianChild]
                        ) -> tuple[float, np.ndarray, np.ndarray]:
    """返回归一化质量权重的 0 阶、1 阶、原始 2 阶矩（不是 central covariance）。"""
    mass = 0.0
    first = np.zeros(3, dtype=np.float64)
    second = np.zeros((3, 3), dtype=np.float64)
    for child in children:
        mass += child.mass_weight
        first += child.mass_weight * child.mean
        second += child.mass_weight * (child.covariance + np.outer(child.mean, child.mean))
    return mass, first, second


# 短名称只作为已有 native 探针的适配；参数语义仍为 wxyz 和线性标准差。
covariance_from_quaternion_scale = quat_wxyz_scale_to_cov
covariance_to_quaternion_scale = cov_to_quat_wxyz_scale


def split_gaussian(mean: Any, covariance: Any, opacity: float, plane_normal: Any,
                   plane_offset: float, *, parent_id: str | int,
                   split_id: str = "plane_split", tail_review_threshold: float = 1e-8
                   ) -> dict[str, Any]:
    """供 native 探针使用的 JSON 字典适配器；不调用 native、不修改 parent。

    native_safe 只表示没有本模块发现的数值/极小尾部/opacity 风险；它不
    证明输入切面可靠，也不代表已验收原生像素重建。False 时调用者应保留
    parent，不能只使用一个 child，也不能裁剪 opacity 后称矩仍然守恒。
    """
    if isinstance(parent_id, (int, np.integer)) and not isinstance(parent_id, (bool, np.bool_)):
        parent_id = str(parent_id)
    result = split_gaussian_two_sided(mean, covariance, plane_normal, plane_offset,
                                     parent_id=parent_id, parent_opacity=opacity,
                                     split_id=split_id,
                                     tail_review_threshold=tail_review_threshold)
    payload = result.to_dict()
    payload["native_safe"] = not result.needs_review
    payload["native_safe_scope"] = "numeric_candidate_gate_only_renderer_and_plane_unvalidated"
    payload["warnings"] = [f"{child.side}: {reason}" for child in result.children
                           for reason in child.review_reasons]
    return payload
