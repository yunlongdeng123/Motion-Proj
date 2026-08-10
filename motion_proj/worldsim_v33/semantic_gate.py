"""V3.3 S5 语义门控 2D residual、区域指标与 fail-safe 选择。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


GATE_SCHEMA_VERSION = "worldsim_v33_semantic_gate_v1"


def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or not value.any():
        raise ValueError(f"{name} 必须是非空二维 mask")
    return value


def build_semantic_gate(
    actor_footprint: np.ndarray,
    *,
    boundary_inner_pixels: int,
    boundary_outer_pixels: int,
    contact_depth_pixels: int,
    contact_side_pixels: int,
    shadow_depth_pixels: int,
    shadow_side_pixels: int,
    boundary_weight: float,
    contact_weight: float,
    shadow_weight: float,
) -> dict[str, np.ndarray]:
    """从 3D actor footprint 构造边界、接地点和阴影 support gate。"""

    # SAM2 冻结环境只消费本模块的语义质量函数，并未安装 SciPy；把形态学
    # 依赖限制在 gate 构建路径，避免检测阶段产生无关的运行时耦合。
    from scipy.ndimage import binary_dilation, binary_erosion

    actor = _validate_mask(actor_footprint, "actor footprint")
    integer_values = (
        boundary_inner_pixels,
        boundary_outer_pixels,
        contact_depth_pixels,
        contact_side_pixels,
        shadow_depth_pixels,
        shadow_side_pixels,
    )
    if any(int(value) < 0 for value in integer_values):
        raise ValueError("semantic gate 像素半径不能为负")
    if boundary_inner_pixels == 0 or boundary_outer_pixels == 0:
        raise ValueError("semantic gate boundary 半径必须为正")
    weights = (boundary_weight, contact_weight, shadow_weight)
    if any(not 0.0 <= float(value) <= 1.0 for value in weights):
        raise ValueError("semantic gate 权重必须位于 [0,1]")

    inner = binary_erosion(actor, iterations=int(boundary_inner_pixels))
    # 极小或极细的 3D footprint 可能被腐蚀为空。S5 仍需给出可审计的
    # actor-interior 指标；此时退回完整 footprint，并把 boundary 限在其外侧。
    if not inner.any():
        inner = actor.copy()
    outer = binary_dilation(actor, iterations=int(boundary_outer_pixels))
    boundary = outer & ~inner
    ys, xs = np.nonzero(actor)
    y_bottom = int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())
    height, width = actor.shape

    contact = np.zeros_like(actor)
    contact[
        max(0, y_bottom - int(boundary_inner_pixels)) : min(
            height, y_bottom + int(contact_depth_pixels) + 1
        ),
        max(0, x_left - int(contact_side_pixels)) : min(
            width, x_right + int(contact_side_pixels) + 1
        ),
    ] = True
    shadow = np.zeros_like(actor)
    shadow[
        y_bottom : min(height, y_bottom + int(shadow_depth_pixels) + 1),
        max(0, x_left - int(shadow_side_pixels)) : min(
            width, x_right + int(shadow_side_pixels) + 1
        ),
    ] = True
    contact &= ~inner
    shadow &= ~actor

    gate = np.zeros(actor.shape, dtype=np.float32)
    gate[shadow] = np.maximum(gate[shadow], float(shadow_weight))
    gate[contact] = np.maximum(gate[contact], float(contact_weight))
    gate[boundary] = np.maximum(gate[boundary], float(boundary_weight))
    allowed = gate > 0
    if not allowed.any() or not boundary.any() or not contact.any():
        raise ValueError("semantic gate 区域退化")
    return {
        "schema_version": np.asarray(GATE_SCHEMA_VERSION, dtype="<U64"),
        "gate": gate,
        "actor_footprint": actor,
        "actor_interior": inner,
        "boundary_ring": boundary,
        "ground_contact": contact,
        "shadow_support": shadow,
        "allowed": allowed,
        "far_non_target": ~allowed,
    }


def validate_semantic_gate(regions: Mapping[str, np.ndarray]) -> None:
    required = {
        "schema_version",
        "gate",
        "actor_footprint",
        "actor_interior",
        "boundary_ring",
        "ground_contact",
        "shadow_support",
        "allowed",
        "far_non_target",
    }
    missing = required - set(regions)
    if missing:
        raise ValueError(f"semantic gate 缺字段: {sorted(missing)}")
    if str(np.asarray(regions["schema_version"]).item()) != GATE_SCHEMA_VERSION:
        raise ValueError("semantic gate schema version 漂移")
    gate = np.asarray(regions["gate"], dtype=np.float32)
    if gate.ndim != 2 or not np.isfinite(gate).all() or np.any(
        (gate < 0.0) | (gate > 1.0)
    ):
        raise ValueError("semantic gate 数值非法")
    shape = gate.shape
    for name in required - {"schema_version", "gate"}:
        if np.asarray(regions[name]).shape != shape:
            raise ValueError(f"semantic gate {name} shape 漂移")
    allowed = np.asarray(regions["allowed"], dtype=bool)
    far = np.asarray(regions["far_non_target"], dtype=bool)
    if not np.array_equal(allowed, gate > 0) or not np.array_equal(far, ~allowed):
        raise ValueError("semantic gate allowed/far 合同漂移")


def apply_gated_residual(
    raw: np.ndarray,
    unconstrained: np.ndarray,
    gate: np.ndarray,
    *,
    residual_cap_uint8: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行 input + gate*clamp(enhancer-input)，far 区保持逐位相同。"""

    source = np.asarray(raw)
    candidate = np.asarray(unconstrained)
    gate_value = np.asarray(gate, dtype=np.float32)
    if (
        source.dtype != np.uint8
        or candidate.dtype != np.uint8
        or source.shape != candidate.shape
        or source.ndim != 3
        or source.shape[2] != 3
        or gate_value.shape != source.shape[:2]
    ):
        raise ValueError("gated residual RGB/gate schema 不合法")
    cap = float(residual_cap_uint8)
    if cap <= 0 or cap > 255:
        raise ValueError("residual cap 非法")
    residual = candidate.astype(np.float32) - source.astype(np.float32)
    clipped = np.clip(residual, -cap, cap)
    blended = source.astype(np.float32) + gate_value[..., None] * clipped
    output = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    far = gate_value == 0
    far_changed = int(np.count_nonzero(np.any(output[far] != source[far], axis=1)))
    if far_changed:
        raise RuntimeError("semantic gate 修改了 far non-target pixel")
    changed = np.any(output != source, axis=2)
    return output, {
        "residual_cap_uint8": cap,
        "maximum_unconstrained_abs_residual_uint8": float(
            np.abs(residual).max()
        ),
        "maximum_applied_abs_residual_uint8": float(
            np.abs(output.astype(np.int16) - source.astype(np.int16)).max()
        ),
        "changed_pixels": int(changed.sum()),
        "changed_far_non_target_pixels": far_changed,
        "allowed_pixels": int((gate_value > 0).sum()),
    }


def region_l1(
    prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> float:
    selected = np.asarray(mask, dtype=bool)
    left, right = np.asarray(prediction), np.asarray(reference)
    if left.shape != right.shape or left.shape[:2] != selected.shape or not selected.any():
        raise ValueError("region L1 输入或 mask 非法")
    return float(
        np.abs(left.astype(np.float32) - right.astype(np.float32))[selected].mean()
    )


def region_psnr(
    prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> float:
    selected = np.asarray(mask, dtype=bool)
    left, right = np.asarray(prediction), np.asarray(reference)
    if left.shape != right.shape or left.shape[:2] != selected.shape or not selected.any():
        raise ValueError("region PSNR 输入或 mask 非法")
    difference = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.square(difference[selected]).mean())
    return float("inf") if mse == 0 else float(10.0 * math.log10(255.0**2 / mse))


def evaluate_semantic_gate(
    *,
    raw: np.ndarray,
    gated: np.ndarray,
    reference: np.ndarray,
    regions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    validate_semantic_gate(regions)
    names = (
        "boundary_ring",
        "ground_contact",
        "shadow_support",
        "actor_interior",
        "far_non_target",
    )
    output: dict[str, Any] = {}
    for name in names:
        mask = np.asarray(regions[name], dtype=bool)
        output[name] = {
            "raw_l1_uint8": region_l1(raw, reference, mask),
            "gated_l1_uint8": region_l1(gated, reference, mask),
            "raw_psnr_db": region_psnr(raw, reference, mask),
            "gated_psnr_db": region_psnr(gated, reference, mask),
            "l1_delta": region_l1(gated, reference, mask)
            - region_l1(raw, reference, mask),
        }
    output["raw_gated"] = {
        name: region_l1(gated, raw, np.asarray(regions[name], dtype=bool))
        for name in names
    }
    return output


def semantic_mass(logits: np.ndarray, reference_mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float32)
    mask = _validate_mask(reference_mask, "semantic reference mask")
    if values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError("semantic logits shape/有限性错误")
    probability = np.empty_like(values)
    positive = values >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    probability[~positive] = exponential / (1.0 + exponential)
    binary = values > 0
    return {
        "mean_probability_inside_reference": float(probability[mask].mean()),
        "positive_fraction_inside_reference": float(binary[mask].mean()),
        "positive_pixels_inside_reference": int(np.sum(binary & mask)),
        "predicted_positive_pixels": int(binary.sum()),
    }


def semantic_reintroduction_decision(
    *,
    raw: Mapping[str, float],
    unconstrained: Mapping[str, float],
    production: Mapping[str, float],
    minimum_candidate_mass_increase: float,
    minimum_candidate_positive_fraction_increase: float,
    maximum_production_mass_increase: float,
    maximum_production_positive_fraction_increase: float,
) -> dict[str, Any]:
    mass_name = "mean_probability_inside_reference"
    fraction_name = "positive_fraction_inside_reference"
    candidate_mass_delta = float(unconstrained[mass_name]) - float(raw[mass_name])
    candidate_fraction_delta = float(unconstrained[fraction_name]) - float(raw[fraction_name])
    production_mass_delta = float(production[mass_name]) - float(raw[mass_name])
    production_fraction_delta = float(production[fraction_name]) - float(raw[fraction_name])
    flagged = (
        candidate_mass_delta >= float(minimum_candidate_mass_increase)
        or candidate_fraction_delta
        >= float(minimum_candidate_positive_fraction_increase)
    )
    production_safe = (
        production_mass_delta <= float(maximum_production_mass_increase)
        and production_fraction_delta
        <= float(maximum_production_positive_fraction_increase)
    )
    return {
        "unconstrained_candidate_flagged": bool(flagged),
        "production_safe": bool(production_safe),
        "candidate_mass_delta": candidate_mass_delta,
        "candidate_positive_fraction_delta": candidate_fraction_delta,
        "production_mass_delta": production_mass_delta,
        "production_positive_fraction_delta": production_fraction_delta,
        "production_policy": "raw_3d_delete_render_fallback",
    }


def development_selection(
    rows: Sequence[Mapping[str, Any]], gates: Mapping[str, float]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("development selection rows 为空")
    boundary_delta = float(
        np.mean([row["metrics"]["boundary_ring"]["l1_delta"] for row in rows])
    )
    contact_delta = float(
        np.mean([row["metrics"]["ground_contact"]["l1_delta"] for row in rows])
    )
    actor_delta = float(
        np.mean([row["metrics"]["actor_interior"]["l1_delta"] for row in rows])
    )
    safeguards = {
        "boundary_non_degradation": boundary_delta
        <= float(gates["maximum_mean_boundary_l1_degradation"]),
        "contact_non_degradation": contact_delta
        <= float(gates["maximum_mean_contact_l1_degradation"]),
        "actor_interior_preserved": actor_delta
        <= float(gates["maximum_mean_actor_interior_l1_degradation"]),
        "far_non_target_exact": all(
            int(row["blend_audit"]["changed_far_non_target_pixels"]) == 0
            for row in rows
        ),
        "residual_cap_exact": all(
            float(row["blend_audit"]["maximum_applied_abs_residual_uint8"])
            <= float(gates["residual_cap_uint8"])
            for row in rows
        ),
        "delete_production_exact": all(bool(row["delete_raw_production_exact"]) for row in rows),
        "delete_semantic_safe": all(
            bool(row["semantic_reintroduction"]["production_safe"])
            for row in rows
        ),
    }
    improvement = (
        boundary_delta <= -float(gates["minimum_mean_l1_improvement"])
        or contact_delta <= -float(gates["minimum_mean_l1_improvement"])
    )
    selected = "G1_semantic_gate" if all(safeguards.values()) and improvement else "G0_raw_3d"
    return {
        "selected_arm": selected,
        "mean_boundary_l1_delta": boundary_delta,
        "mean_contact_l1_delta": contact_delta,
        "mean_actor_interior_l1_delta": actor_delta,
        "safeguards": safeguards,
        "improvement_gate": bool(improvement),
        "accepted": selected == "G1_semantic_gate",
        "no_gain": selected == "G0_raw_3d",
    }
