"""Matched evaluation and frozen H promotion gate for V5.1 D0."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


METRICS = (
    "boundary_f1",
    "iou_at_frozen_threshold",
    "false_negative_semantic_mass",
    "false_positive_semantic_mass",
    "brier",
    "ece",
    "nll",
)


def _metric_mapping(value: Mapping[str, Any], label: str) -> dict[str, float]:
    result = {}
    for name in METRICS:
        number = float(value[name])
        if not np.isfinite(number):
            raise ValueError(f"non-finite D0 evaluation metric: {label}/{name}")
        result[name] = number
    return result


def evaluate_progressive_h_gate(
    scenes: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the normative section 23.1 gate against U2/B3 G0."""

    expected_scene_count = int(gate["scene_count"])
    if len(scenes) != expected_scene_count:
        raise ValueError("D0 H scene denominator drift")
    deltas = []
    scene_rows = []
    for scene in scenes:
        arms = scene["evaluation_aggregate"]
        baseline = _metric_mapping(arms["U2_B3_G0"], f"{scene['scene']}:G0")
        candidate = _metric_mapping(arms["D0"], f"{scene['scene']}:D0")
        delta = {name: candidate[name] - baseline[name] for name in METRICS}
        deltas.append(delta)
        scene_rows.append({"scene": scene["scene"], "delta_vs_u2_b3_g0": delta})

    scene_balanced = {
        name: float(np.mean([row[name] for row in deltas], dtype=np.float64))
        for name in METRICS
    }
    positive_boundary_count = sum(row["boundary_f1"] > 0.0 for row in deltas)
    checks = {
        "positive_boundary_f1_scene_count": positive_boundary_count
        >= int(gate["minimum_positive_boundary_f1_scenes"]),
        "scene_balanced_boundary_f1_positive": scene_balanced["boundary_f1"]
        > float(gate["minimum_scene_balanced_boundary_f1_delta_exclusive"]),
        "scene_balanced_iou_nonnegative": scene_balanced["iou_at_frozen_threshold"]
        >= float(gate["minimum_scene_balanced_iou_delta"]),
        "scene_balanced_false_negative_safeguard": scene_balanced[
            "false_negative_semantic_mass"
        ]
        <= float(gate["maximum_scene_balanced_false_negative_semantic_mass_delta"]),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "positive_boundary_f1_scene_count": positive_boundary_count,
        "scene_balanced_delta_vs_u2_b3_g0": scene_balanced,
        "scenes": scene_rows,
    }
