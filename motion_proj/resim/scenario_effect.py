"""只读取 3D realized state 的 cut-in/merge scenario-effect evaluator。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canonical_hash import canonical_sha256
from .safety_geometry import OrientedBox


@dataclass(frozen=True)
class ScenarioThresholds:
    event_min_consecutive_frames: int = 5
    min_lateral_gap_change_m: float = 0.5
    min_boundary_crossings: int = 1
    ttc_valid_range_s: tuple[float, float] = (1.0, 6.0)
    time_headway_valid_range_s: tuple[float, float] = (0.5, 4.0)


def _longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _corridor_intersection(box: OrientedBox, half_width_m: float) -> bool:
    corners_y = box.corners()[:, 1]
    return float(corners_y.min()) <= half_width_m and float(corners_y.max()) >= -half_width_m


def evaluate_scenario_effect(
    source: list[OrientedBox],
    realized: list[OrientedBox],
    *,
    ego_speed_mps: float,
    corridor_half_width_m: float,
    dt_s: float,
    thresholds: ScenarioThresholds,
    corridor_source: str,
) -> dict:
    if corridor_source not in {"hd_map", "proxy"}:
        raise ValueError("corridor_source 必须显式为 hd_map 或 proxy")
    if len(source) != len(realized) or not source:
        raise ValueError("source/realized 必须等长且非空")
    source_inside = np.asarray([
        _corridor_intersection(box, corridor_half_width_m) for box in source
    ])
    realized_inside = np.asarray([
        _corridor_intersection(box, corridor_half_width_m) for box in realized
    ])
    crossings = int(np.count_nonzero((~realized_inside[:-1]) & realized_inside[1:]))
    duration = _longest_true_run(realized_inside)
    source_gap = np.asarray([
        max(abs(box.center[1]) - box.dimensions_lwh[1] / 2 - corridor_half_width_m, 0.0)
        for box in source
    ])
    realized_gap = np.asarray([
        max(abs(box.center[1]) - box.dimensions_lwh[1] / 2 - corridor_half_width_m, 0.0)
        for box in realized
    ])
    lateral_gap_change = float(source_gap.min() - realized_gap.min())
    longitudinal = np.asarray([box.center[0] for box in realized])
    actor_speed = float(np.median(np.diff(longitudinal) / dt_s)) if len(realized) > 1 else 0.0
    closing_speed = ego_speed_mps - actor_speed
    positive_distance = longitudinal[longitudinal > 0]
    distance = float(positive_distance.min()) if len(positive_distance) else float("nan")
    if closing_speed > 1e-6 and np.isfinite(distance):
        headway_kind = "ttc"
        headway = distance / closing_speed
        valid_range = thresholds.ttc_valid_range_s
    elif ego_speed_mps > 1e-6 and np.isfinite(distance):
        headway_kind = "time_headway"
        headway = distance / ego_speed_mps
        valid_range = thresholds.time_headway_valid_range_s
    else:
        headway_kind = "unavailable"
        headway = None
        valid_range = (0.0, 0.0)
    source_label = int(_longest_true_run(source_inside) >= thresholds.event_min_consecutive_frames)
    realized_label = int(
        crossings >= thresholds.min_boundary_crossings
        and duration >= thresholds.event_min_consecutive_frames
        and lateral_gap_change >= thresholds.min_lateral_gap_change_m
        and headway is not None
        and valid_range[0] <= headway <= valid_range[1]
    )
    result = {
        "schema_version": "scenario-effect-v1",
        "corridor_source": corridor_source,
        "source_label": source_label,
        "realized_label": realized_label,
        "label_transition": f"{source_label}->{realized_label}",
        "boundary_crossings": crossings,
        "max_consecutive_inside_frames": duration,
        "lateral_gap_change_m": lateral_gap_change,
        "headway_kind": headway_kind,
        "headway_s": headway,
        "positive": source_label == 0 and realized_label == 1,
        "negative": source_label == 0 and realized_label == 0,
    }
    result["scenario_effect_hash"] = canonical_sha256(result)
    return result


def build_counterfactual_pair(
    *,
    scene_id: str,
    source_actor_id: int,
    positive_proposal_id: str,
    negative_proposal_id: str,
    positive_effect: dict,
    negative_effect: dict,
) -> dict:
    if not positive_effect.get("positive"):
        raise ValueError("positive proposal 未实现 0->1")
    if not negative_effect.get("negative"):
        raise ValueError("negative proposal 未保持 0->0")
    pair = {
        "schema_version": "counterfactual-pair-v1",
        "counterfactual_pair_id": (
            f"{scene_id}:{source_actor_id}:{positive_proposal_id}:{negative_proposal_id}"
        ),
        "scene_id": str(scene_id),
        "source_actor_id": int(source_actor_id),
        "positive_proposal_id": positive_proposal_id,
        "negative_proposal_id": negative_proposal_id,
        "requested_effect": {"positive": "0->1", "negative": "0->0"},
        "realized_effect": {
            "positive": positive_effect["label_transition"],
            "negative": negative_effect["label_transition"],
        },
        "scenario_effect_hash": canonical_sha256(
            {
                "positive": positive_effect["scenario_effect_hash"],
                "negative": negative_effect["scenario_effect_hash"],
            }
        ),
    }
    pair["pair_hash"] = canonical_sha256(pair)
    return pair
