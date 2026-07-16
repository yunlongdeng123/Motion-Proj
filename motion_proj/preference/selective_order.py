"""Strict/tie/incomparable 关系与 condition 级无环偏序图。"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

import torch

from .calibration import PRIMARY_COMPONENTS
from .common_support import CommonSupportWindow
from .residual_motion import MotionComponentEvidence


RELATIONS = {"strict", "tie", "incomparable"}


def video_quality_metrics(frames: torch.Tensor) -> dict[str, Any]:
    """冻结的轻量画质诊断；只做 comparability gate，不产生 physics reward。"""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames 必须是 [K,3,H,W]")
    finite = bool(torch.isfinite(frames).all())
    value = torch.nan_to_num(frames.float(), nan=0.0, posinf=1.0, neginf=-1.0).add(1.0).mul(0.5).clamp(0, 1)
    luminance = value.mean(dim=1)
    frame_luminance = luminance.mean(dim=(1, 2))
    black_frame_fraction = float((frame_luminance <= 0.02).float().mean())
    saturation_fraction = float(((value <= 0.01) | (value >= 0.99)).float().mean())
    temporal = (value[1:] - value[:-1]).abs().mean(dim=(1, 2, 3))
    dx = (luminance[:, :, 1:] - luminance[:, :, :-1]).abs().mean(dim=(1, 2))
    dy = (luminance[:, 1:, :] - luminance[:, :-1, :]).abs().mean(dim=(1, 2))
    sharpness = dx + dy
    sharpness_jump = (
        float((sharpness[1:] - sharpness[:-1]).abs().max()) if sharpness.shape[0] > 1 else 0.0
    )
    return {
        "finite": finite,
        "black_frame_fraction": black_frame_fraction,
        "saturation_fraction": saturation_fraction,
        "temporal_l1_median": float(temporal.median()) if int(temporal.numel()) else 0.0,
        "temporal_l1_max": float(temporal.max()) if int(temporal.numel()) else 0.0,
        "sharpness_median": float(sharpness.median()),
        "sharpness_jump_max": sharpness_jump,
    }


def quality_comparability(
    quality_a: Mapping[str, Any],
    quality_b: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """绝对 catastrophic gate 加 pairwise equivalence；任何失败均不可排序。"""
    reasons = []
    for side, quality in (("a", quality_a), ("b", quality_b)):
        if not bool(quality.get("finite")):
            reasons.append(f"{side}_nonfinite")
        if float(quality.get("black_frame_fraction", 1.0)) > float(settings["maximum_black_frame_fraction"]):
            reasons.append(f"{side}_black_frames")
        if float(quality.get("saturation_fraction", 1.0)) > float(settings["maximum_saturation_fraction"]):
            reasons.append(f"{side}_saturation")
        if float(quality.get("temporal_l1_max", float("inf"))) > float(settings["maximum_temporal_l1"]):
            reasons.append(f"{side}_temporal_jump")
        if float(quality.get("sharpness_jump_max", float("inf"))) > float(settings["maximum_sharpness_jump"]):
            reasons.append(f"{side}_sharpness_jump")
    if abs(float(quality_a["saturation_fraction"]) - float(quality_b["saturation_fraction"])) > float(
        settings["maximum_pair_saturation_difference"]
    ):
        reasons.append("pair_saturation_mismatch")
    if abs(float(quality_a["temporal_l1_median"]) - float(quality_b["temporal_l1_median"])) > float(
        settings["maximum_pair_temporal_l1_difference"]
    ):
        reasons.append("pair_temporal_mismatch")
    sharp_a = max(float(quality_a["sharpness_median"]), 1.0e-8)
    sharp_b = max(float(quality_b["sharpness_median"]), 1.0e-8)
    ratio = sharp_a / sharp_b
    if ratio < float(settings["minimum_pair_sharpness_ratio"]) or ratio > float(
        settings["maximum_pair_sharpness_ratio"]
    ):
        reasons.append("pair_sharpness_mismatch")
    return {"comparable": not reasons, "reasons": reasons, "sharpness_ratio_a_to_b": ratio}


def _finite_metric(metrics: Mapping[str, float | None], name: str) -> float | None:
    value = metrics.get(name)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _activity_noninferior(
    winner: Mapping[str, float | None],
    loser: Mapping[str, float | None],
    settings: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    reasons = []
    active_w = _finite_metric(winner, "active_fraction")
    active_l = _finite_metric(loser, "active_fraction")
    survival_w = _finite_metric(winner, "survival")
    survival_l = _finite_metric(loser, "survival")
    if active_w is None or active_l is None or active_w < active_l - float(settings["maximum_active_fraction_drop"]):
        reasons.append("active_fraction_drop")
    if survival_w is None or survival_l is None or survival_w < survival_l - float(settings["maximum_survival_drop"]):
        reasons.append("survival_drop")
    for name in ("residual_speed", "net_residual_displacement"):
        value_w = _finite_metric(winner, name)
        value_l = _finite_metric(loser, name)
        if value_w is None or value_l is None:
            reasons.append(f"{name}_missing")
            continue
        lower = value_l * (1.0 - float(settings["maximum_motion_drop_fraction"]))
        upper = value_l * (1.0 + float(settings["maximum_motion_increase_fraction"])) + 1.0e-6
        if value_w < lower:
            reasons.append(f"{name}_drop")
        if value_w > upper:
            reasons.append(f"{name}_increase")
    return not reasons, reasons


def _activity_equivalent(
    left: Mapping[str, float | None],
    right: Mapping[str, float | None],
    settings: Mapping[str, Any],
) -> bool:
    for name in ("active_fraction", "survival"):
        a, b = _finite_metric(left, name), _finite_metric(right, name)
        if a is None or b is None or abs(a - b) > float(settings["activity_equivalence_absolute"]):
            return False
    for name in ("residual_speed", "net_residual_displacement"):
        a, b = _finite_metric(left, name), _finite_metric(right, name)
        if a is None or b is None:
            return False
        relative = abs(a - b) / max(abs(a), abs(b), 1.0e-6)
        if relative > float(settings["motion_equivalence_fraction"]):
            return False
    return True


def decide_selective_relation(
    *,
    condition_id: str,
    edge_id: str,
    support: CommonSupportWindow,
    evidence: MotionComponentEvidence,
    intervals: Mapping[str, Mapping[str, Any]],
    ropes: Mapping[str, float],
    strict_threshold: float,
    quality: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """按 fixed gates 输出 strict/tie/incomparable，绝不强制 winner。"""
    base = {
        "condition_id": condition_id,
        "edge_id": edge_id,
        "candidate_a": support.candidate_a,
        "candidate_b": support.candidate_b,
        "start_frame": support.start_frame,
        "end_frame": support.end_frame,
        "winner_candidate_id": None,
        "loser_candidate_id": None,
        "strict_score": None,
        "high_confidence": False,
        "uses_future_gt": False,
    }
    gate_reasons = []
    if not support.valid:
        gate_reasons.append("support_insufficient")
    if not evidence.valid:
        gate_reasons.append(evidence.reason or "motion_evidence_invalid")
    if evidence.camera_distance_px is None:
        gate_reasons.append("camera_distance_invalid")
    elif evidence.camera_distance_px > float(settings["maximum_camera_distance_px"]):
        gate_reasons.append("camera_mismatch")
    if not bool(quality.get("comparable")):
        gate_reasons.append("quality_mismatch")
    if gate_reasons:
        return base | {
            "relation": "incomparable",
            "reason": gate_reasons[0],
            "gate_reasons": gate_reasons,
            "component_evidence": {},
        }

    component_scores = {}
    a_evidence: list[tuple[float, str]] = []
    b_evidence: list[tuple[float, str]] = []
    intervals_inside_rope = True
    for component in PRIMARY_COMPONENTS:
        row = intervals.get(component, {})
        lower = row.get("lower")
        upper = row.get("upper")
        rope = float(ropes[component])
        if lower is None or upper is None or not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
            intervals_inside_rope = False
            component_scores[component] = {"valid": False}
            continue
        lower = float(lower)
        upper = float(upper)
        score_a = lower / max(rope, 1.0e-12)
        score_b = -upper / max(rope, 1.0e-12)
        significant = bool(row.get("holm_significant", False))
        component_scores[component] = {
            "valid": True,
            "lower": lower,
            "upper": upper,
            "rope": rope,
            "score_a": score_a,
            "score_b": score_b,
            "holm_significant": significant,
        }
        if significant and score_a > strict_threshold:
            a_evidence.append((score_a, component))
        if significant and score_b > strict_threshold:
            b_evidence.append((score_b, component))
        intervals_inside_rope &= lower >= -rope and upper <= rope

    if a_evidence and b_evidence:
        return base | {
            "relation": "incomparable",
            "reason": "component_conflict",
            "gate_reasons": [],
            "component_evidence": component_scores,
        }
    direction = "a" if a_evidence else "b" if b_evidence else None
    if direction is not None:
        selected = a_evidence if direction == "a" else b_evidence
        score, component = max(selected)
        winner_activity = evidence.activity_a if direction == "a" else evidence.activity_b
        loser_activity = evidence.activity_b if direction == "a" else evidence.activity_a
        activity_ok, activity_reasons = _activity_noninferior(winner_activity, loser_activity, settings)
        counter_regression = False
        for name, row in component_scores.items():
            if not row.get("valid"):
                continue
            if direction == "a" and float(row["upper"]) < -float(row["rope"]):
                counter_regression = True
            if direction == "b" and float(row["lower"]) > float(row["rope"]):
                counter_regression = True
        if not activity_ok or counter_regression:
            return base | {
                "relation": "incomparable",
                "reason": "activity_noninferiority" if not activity_ok else "component_conflict",
                "gate_reasons": activity_reasons,
                "component_evidence": component_scores,
            }
        winner = support.candidate_a if direction == "a" else support.candidate_b
        loser = support.candidate_b if direction == "a" else support.candidate_a
        return base | {
            "relation": "strict",
            "reason": "strict_evidence",
            "winner_candidate_id": winner,
            "loser_candidate_id": loser,
            "strict_score": score,
            "strict_component": component,
            "high_confidence": score >= strict_threshold + float(settings["high_confidence_margin"]),
            "gate_reasons": [],
            "component_evidence": component_scores,
        }

    if intervals_inside_rope and _activity_equivalent(evidence.activity_a, evidence.activity_b, settings):
        return base | {
            "relation": "tie",
            "reason": "within_measurement_rope",
            "gate_reasons": [],
            "component_evidence": component_scores,
        }
    return base | {
        "relation": "incomparable",
        "reason": "uncertainty_too_wide",
        "gate_reasons": [],
        "component_evidence": component_scores,
    }


def _has_path(adjacency: Mapping[str, set[str]], source: str, target: str, *, skip: tuple[str, str] | None = None) -> bool:
    stack = [source]
    seen = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        for child in adjacency.get(node, set()):
            if skip == (node, child):
                continue
            stack.append(child)
    return False


def _cycle_nodes(nodes: set[str], edges: set[tuple[str, str]]) -> list[str]:
    indegree = {node: 0 for node in nodes}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] = indegree.get(target, 0) + 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = []
    while queue:
        node = queue.popleft()
        visited.append(node)
        for child in sorted(adjacency.get(node, set())):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return sorted(nodes - set(visited))


def build_condition_partial_order(
    condition_id: str,
    relation_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_tie_fraction: float = 0.5,
) -> dict[str, Any]:
    """合并窗口关系、拒绝反向边/cycle，并只在 DAG 上做 transitive reduction。"""
    nodes = {
        str(value)
        for row in relation_rows
        for value in (row.get("candidate_a"), row.get("candidate_b"))
        if value
    }
    pair_directions: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    strict_rows = []
    tie_count = 0
    incomparable_count = 0
    for row in relation_rows:
        relation = str(row.get("relation"))
        if relation not in RELATIONS:
            raise ValueError(f"未知 relation: {relation}")
        if relation == "strict":
            source = str(row["winner_candidate_id"])
            target = str(row["loser_candidate_id"])
            pair_directions[tuple(sorted((source, target)))].add((source, target))
            strict_rows.append(dict(row))
        elif relation == "tie":
            tie_count += 1
        else:
            incomparable_count += 1
    conflicting_pairs = [pair for pair, directions in pair_directions.items() if len(directions) > 1]
    if conflicting_pairs:
        return {
            "condition_id": condition_id,
            "status": "invalid_component_conflict",
            "nodes": sorted(nodes),
            "strict_edges": [],
            "reduced_edges": [],
            "cycle_nodes": [],
            "conflicting_pairs": [list(pair) for pair in conflicting_pairs],
            "tie_windows": tie_count,
            "incomparable_windows": incomparable_count,
            "uses_future_gt": False,
        }
    edges = {next(iter(directions)) for directions in pair_directions.values()}
    cycle = _cycle_nodes(nodes, edges)
    if cycle:
        return {
            "condition_id": condition_id,
            "status": "invalid_cycle",
            "nodes": sorted(nodes),
            "strict_edges": [list(edge) for edge in sorted(edges)],
            "reduced_edges": [],
            "cycle_nodes": cycle,
            "conflicting_pairs": [],
            "tie_windows": tie_count,
            "incomparable_windows": incomparable_count,
            "uses_future_gt": False,
        }
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
    reduced = {
        edge for edge in edges
        if not _has_path(adjacency, edge[0], edge[1], skip=edge)
    }
    compared = tie_count + incomparable_count
    tie_fraction = tie_count / max(compared, 1)
    status = "strict" if edges else "tie" if tie_count and tie_fraction >= minimum_tie_fraction else "incomparable"
    return {
        "condition_id": condition_id,
        "status": status,
        "nodes": sorted(nodes),
        "strict_edges": [list(edge) for edge in sorted(edges)],
        "reduced_edges": [list(edge) for edge in sorted(reduced)],
        "cycle_nodes": [],
        "conflicting_pairs": [],
        "tie_windows": tie_count,
        "incomparable_windows": incomparable_count,
        "tie_fraction_without_strict": tie_fraction,
        "strict_window_count": len(strict_rows),
        "uses_future_gt": False,
    }
