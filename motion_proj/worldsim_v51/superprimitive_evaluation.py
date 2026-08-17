"""E0b 对 U2/B3 主基线和 D0 机制基线的 matched H gate。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from motion_proj.worldsim_v51.progressive_evaluation import METRICS


def _metrics(value: Mapping[str, Any], label: str) -> dict[str, float]:
    result = {name: float(value[name]) for name in METRICS}
    if not all(np.isfinite(number) for number in result.values()):
        raise ValueError(f"non-finite E0b metric: {label}")
    return result


def _balanced(deltas: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in deltas], dtype=np.float64))
        for name in METRICS
    }


def evaluate_e0b_h_gate(
    scenes: Sequence[Mapping[str, Any]],
    primary_gate: Mapping[str, Any],
    mechanism_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """只有相对 U2/B3 与 raw D0 两组冻结门同时通过才接受 E0b。"""

    expected = int(primary_gate["scene_count"])
    if len(scenes) != expected or int(mechanism_gate["scene_count"]) != expected:
        raise ValueError("E0b H scene denominator drift")
    primary_deltas = []
    mechanism_deltas = []
    scene_rows = []
    for scene in scenes:
        arms = scene["evaluation_aggregate"]
        baseline = _metrics(arms["U2_B3_G0"], f"{scene['scene']}:G0")
        d0 = _metrics(arms["D0"], f"{scene['scene']}:D0")
        e0b = _metrics(arms["E0B"], f"{scene['scene']}:E0B")
        primary = {name: e0b[name] - baseline[name] for name in METRICS}
        mechanism = {name: e0b[name] - d0[name] for name in METRICS}
        primary_deltas.append(primary)
        mechanism_deltas.append(mechanism)
        scene_rows.append(
            {
                "scene": scene["scene"],
                "delta_vs_u2_b3_g0": primary,
                "delta_vs_d0": mechanism,
            }
        )

    primary_balanced = _balanced(primary_deltas)
    mechanism_balanced = _balanced(mechanism_deltas)
    positive_primary = sum(row["boundary_f1"] > 0.0 for row in primary_deltas)
    nonnegative_mechanism = sum(row["boundary_f1"] >= 0.0 for row in mechanism_deltas)
    primary_checks = {
        "positive_boundary_f1_scene_count": positive_primary
        >= int(primary_gate["minimum_positive_boundary_f1_scenes"]),
        "scene_balanced_boundary_f1_positive": primary_balanced["boundary_f1"]
        > float(primary_gate["minimum_scene_balanced_boundary_f1_delta_exclusive"]),
        "scene_balanced_iou_nonnegative": primary_balanced["iou_at_frozen_threshold"]
        >= float(primary_gate["minimum_scene_balanced_iou_delta"]),
        "scene_balanced_false_negative_safeguard": primary_balanced[
            "false_negative_semantic_mass"
        ]
        <= float(primary_gate["maximum_scene_balanced_false_negative_semantic_mass_delta"]),
    }
    mechanism_checks = {
        "nonnegative_boundary_f1_scene_count": nonnegative_mechanism
        >= int(mechanism_gate["minimum_nonnegative_boundary_f1_scenes"]),
        "scene_balanced_boundary_f1_positive": mechanism_balanced["boundary_f1"]
        > float(mechanism_gate["minimum_scene_balanced_boundary_f1_delta_exclusive"]),
        "scene_balanced_iou_nonnegative": mechanism_balanced["iou_at_frozen_threshold"]
        >= float(mechanism_gate["minimum_scene_balanced_iou_delta"]),
        "scene_balanced_false_negative_nonincreasing": mechanism_balanced[
            "false_negative_semantic_mass"
        ]
        <= float(
            mechanism_gate["maximum_scene_balanced_false_negative_semantic_mass_delta"]
        ),
    }
    primary_pass = bool(all(primary_checks.values()))
    mechanism_pass = bool(all(mechanism_checks.values()))
    return {
        "pass": primary_pass and mechanism_pass,
        "primary_gate": {
            "pass": primary_pass,
            "checks": primary_checks,
            "positive_boundary_f1_scene_count": positive_primary,
            "scene_balanced_delta_vs_u2_b3_g0": primary_balanced,
        },
        "mechanism_gate": {
            "pass": mechanism_pass,
            "checks": mechanism_checks,
            "nonnegative_boundary_f1_scene_count": nonnegative_mechanism,
            "scene_balanced_delta_vs_d0": mechanism_balanced,
        },
        "scenes": scene_rows,
    }
