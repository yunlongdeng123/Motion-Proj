"""PA2 的无加权和偏好判定。

这里刻意把候选物理诊断与 pair 判定拆开：候选先满足硬可行性，再只以
P-UNC 的高置信投影能量作为 primary 改善量，同时对运动、存活、位移与
质量施加 non-inferiority。这样不能靠把视频静止化或牺牲可见轨迹取得标签。
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


DECISIVE_LABELS = frozenset({"a_wins", "b_wins"})
PAIR_LABELS = frozenset({"a_wins", "b_wins", "tie", "abstain", "invalid"})


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / max(abs(denominator), 1.0e-8)


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _window_energy(score: Mapping[str, Any], start: int, end: int) -> tuple[float | None, int]:
    values = score.get("projection_energy_by_frame")
    counts = score.get("projection_points_by_frame")
    if not isinstance(values, Sequence) or not isinstance(counts, Sequence):
        return None, 0
    if len(values) != len(counts) or start < 0 or end > len(values) or start >= end:
        return None, 0
    numerator, denominator = 0.0, 0
    for value, count in zip(values[start:end], counts[start:end]):
        scalar = _finite(value)
        points = int(count) if isinstance(count, int) and not isinstance(count, bool) else 0
        if scalar is None or points <= 0:
            continue
        numerator += scalar * points
        denominator += points
    return (numerator / denominator if denominator else None), denominator


def _window_mean(score: Mapping[str, Any], field: str, start: int, end: int) -> float | None:
    values = score.get(field)
    if not isinstance(values, Sequence) or start < 0 or end > len(values) or start >= end:
        return None
    return _mean([_finite(value) for value in values[start:end]])


def candidate_feasibility(
    score: Mapping[str, Any],
    base_score: Mapping[str, Any],
    quality: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """对单条 generated RGB candidate 应用 PA2 Stage-A 硬门槛。"""
    projection_points = int(score.get("projection_points", 0)) if isinstance(score.get("projection_points"), int) else 0
    primary_tracks = int(score.get("primary_track_count", 0)) if isinstance(score.get("primary_track_count"), int) else 0
    coverage = _finite(score.get("track_coverage"))
    median_length = _finite(score.get("median_track_length_frames"))
    confidence = _finite(score.get("scorer_confidence"))
    survival = _finite(score.get("survival_rate"))
    motion = _finite(score.get("motion_magnitude"))
    displacement = _finite(score.get("net_displacement"))
    base_survival = _finite(base_score.get("survival_rate"))
    base_motion = _finite(base_score.get("motion_magnitude"))
    base_displacement = _finite(base_score.get("net_displacement"))
    saturation = _finite(quality.get("saturation_fraction"))
    motion_ratio = _ratio(motion, base_motion)
    displacement_ratio = _ratio(displacement, base_displacement)
    invariants = score.get("punc_invariants") if isinstance(score.get("punc_invariants"), Mapping) else {}
    frame0_correction = _finite(invariants.get("frame0_correction_max_px"))
    visibility_changed = invariants.get("visibility_changed_count")
    checks = {
        "uses_future_gt_false": score.get("uses_future_gt") is False,
        "finite": bool(quality.get("finite")),
        "projection_energy": _finite(score.get("projection_energy")) is not None,
        "minimum_projection_points": projection_points >= int(thresholds["minimum_projection_points"]),
        "minimum_primary_tracks": primary_tracks >= int(thresholds["minimum_primary_tracks"]),
        "track_coverage": coverage is not None and coverage >= float(thresholds["minimum_track_coverage"]),
        "median_track_length": median_length is not None and median_length >= float(thresholds["minimum_median_track_length"]),
        "scorer_confidence": confidence is not None and confidence >= float(thresholds["minimum_scorer_confidence"]),
        "not_saturated": saturation is not None and saturation <= float(thresholds["maximum_saturation_fraction"]),
        "punc_frame0_exact": frame0_correction is not None and frame0_correction <= float(thresholds["frame0_max_correction_px"]),
        "punc_visibility_preserved": isinstance(visibility_changed, int) and not isinstance(visibility_changed, bool)
        and visibility_changed <= int(thresholds["maximum_visibility_changed_count"]),
        "survival_relative_to_base": (
            survival is not None and base_survival is not None
            and survival >= base_survival - float(thresholds["maximum_survival_drop_to_base"])
        ),
        "motion_relative_to_base": (
            motion_ratio is not None
            and float(thresholds["minimum_motion_ratio_to_base"]) <= motion_ratio <= float(thresholds["maximum_motion_ratio_to_base"])
        ),
        "displacement_relative_to_base": (
            displacement_ratio is not None
            and float(thresholds["minimum_displacement_ratio_to_base"]) <= displacement_ratio
            <= float(thresholds["maximum_displacement_ratio_to_base"])
        ),
    }
    return {
        "feasible": all(checks.values()),
        "checks": checks,
        "motion_ratio_to_base": motion_ratio,
        "displacement_ratio_to_base": displacement_ratio,
    }


def _noninferiority(
    winner: Mapping[str, Any],
    loser: Mapping[str, Any],
    winner_quality: Mapping[str, Any],
    loser_quality: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    winner_survival, loser_survival = _finite(winner.get("survival_rate")), _finite(loser.get("survival_rate"))
    winner_motion, loser_motion = _finite(winner.get("motion_magnitude")), _finite(loser.get("motion_magnitude"))
    winner_displacement, loser_displacement = _finite(winner.get("net_displacement")), _finite(loser.get("net_displacement"))
    winner_coverage, loser_coverage = _finite(winner.get("track_coverage")), _finite(loser.get("track_coverage"))
    winner_saturation = _finite(winner_quality.get("saturation_fraction"))
    loser_saturation = _finite(loser_quality.get("saturation_fraction"))
    return {
        "survival": (
            winner_survival is not None and loser_survival is not None
            and winner_survival >= loser_survival - float(thresholds["maximum_pair_survival_drop"])
        ),
        "motion": (
            winner_motion is not None and loser_motion is not None
            and winner_motion >= loser_motion * (1.0 - float(thresholds["maximum_pair_motion_drop_fraction"]))
        ),
        "displacement": (
            winner_displacement is not None and loser_displacement is not None
            and winner_displacement >= loser_displacement * (1.0 - float(thresholds["maximum_pair_displacement_drop_fraction"]))
        ),
        "coverage": (
            winner_coverage is not None and loser_coverage is not None
            and winner_coverage >= loser_coverage - float(thresholds["maximum_pair_coverage_drop"])
        ),
        "quality": (
            winner_saturation is not None and loser_saturation is not None
            and winner_saturation <= loser_saturation + float(thresholds["maximum_pair_saturation_increase"])
        ),
    }


def _pair_confidence(
    winner: Mapping[str, Any],
    loser: Mapping[str, Any],
    winner_quality: Mapping[str, Any],
    relative_margin: float,
    thresholds: Mapping[str, Any],
) -> float:
    track = min(
        _clip01(float(_finite(winner.get("scorer_confidence")) or 0.0)),
        _clip01(float(_finite(loser.get("scorer_confidence")) or 0.0)),
    )
    margin = _clip01(relative_margin / max(float(thresholds["confidence_reference_relative_margin"]), 1.0e-8))
    saturation = float(_finite(winner_quality.get("saturation_fraction")) or 1.0)
    quality = _clip01(1.0 - saturation / max(float(thresholds["maximum_saturation_fraction"]), 1.0e-8))
    return _clip01(track * margin * quality)


def decide_global_pair(
    *,
    candidate_a: str,
    candidate_b: str,
    score_a: Mapping[str, Any],
    score_b: Mapping[str, Any],
    feasibility_a: Mapping[str, Any],
    feasibility_b: Mapping[str, Any],
    quality_a: Mapping[str, Any],
    quality_b: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """用 primary projection margin + 约束非劣性给一对视频判定。

    此函数没有线性 reward，且两个不 feasible 的候选永远不会产生 winner。
    """
    energy_a, energy_b = _finite(score_a.get("projection_energy")), _finite(score_b.get("projection_energy"))
    common = {
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "feasibility_a": dict(feasibility_a),
        "feasibility_b": dict(feasibility_b),
        "physics_components": {
            "a": {key: score_a.get(key) for key in ("projection_energy", "projection_points", "survival_rate", "motion_magnitude", "net_displacement", "track_coverage", "scorer_confidence")},
            "b": {key: score_b.get(key) for key in ("projection_energy", "projection_points", "survival_rate", "motion_magnitude", "net_displacement", "track_coverage", "scorer_confidence")},
        },
        "quality_components": {
            "a": dict(quality_a),
            "b": dict(quality_b),
        },
    }
    if energy_a is None or energy_b is None:
        return {**common, "global_label": "invalid", "winner_candidate_id": None, "loser_candidate_id": None,
                "preference_margin": 0.0, "pair_confidence": 0.0, "abstain_reason": "nonfinite_projection_energy"}
    if not bool(feasibility_a.get("feasible")) or not bool(feasibility_b.get("feasible")):
        return {**common, "global_label": "abstain", "winner_candidate_id": None, "loser_candidate_id": None,
                "preference_margin": 0.0, "pair_confidence": 0.0, "abstain_reason": "candidate_infeasible"}
    if math.isclose(energy_a, energy_b, rel_tol=0.0, abs_tol=float(thresholds["tie_absolute_projection_margin"])):
        return {**common, "global_label": "tie", "winner_candidate_id": None, "loser_candidate_id": None,
                "preference_margin": 0.0, "pair_confidence": 0.0, "abstain_reason": "projection_tie"}
    if energy_a < energy_b:
        label, winner_id, loser_id = "a_wins", candidate_a, candidate_b
        winner, loser, winner_quality, loser_quality = score_a, score_b, quality_a, quality_b
        raw_margin = energy_b - energy_a
    else:
        label, winner_id, loser_id = "b_wins", candidate_b, candidate_a
        winner, loser, winner_quality, loser_quality = score_b, score_a, quality_b, quality_a
        raw_margin = energy_a - energy_b
    relative_margin = raw_margin / max(abs(float(_finite(loser.get("projection_energy")) or 0.0)), 1.0e-8)
    margin_pass = (
        raw_margin >= float(thresholds["minimum_absolute_projection_margin"])
        and relative_margin >= float(thresholds["minimum_relative_projection_margin"])
    )
    noninferiority = _noninferiority(winner, loser, winner_quality, loser_quality, thresholds)
    if not margin_pass:
        return {**common, "global_label": "abstain", "winner_candidate_id": None, "loser_candidate_id": None,
                "preference_margin": raw_margin, "pair_confidence": 0.0, "abstain_reason": "projection_margin_below_threshold",
                "noninferiority": noninferiority, "relative_projection_margin": relative_margin}
    if not all(noninferiority.values()):
        return {**common, "global_label": "abstain", "winner_candidate_id": None, "loser_candidate_id": None,
                "preference_margin": raw_margin, "pair_confidence": 0.0, "abstain_reason": "noninferiority_conflict",
                "noninferiority": noninferiority, "relative_projection_margin": relative_margin}
    confidence = _pair_confidence(winner, loser, winner_quality, relative_margin, thresholds)
    if confidence < float(thresholds["minimum_pair_confidence"]):
        return {**common, "global_label": "abstain", "winner_candidate_id": None, "loser_candidate_id": None,
                "preference_margin": raw_margin, "pair_confidence": confidence, "abstain_reason": "pair_confidence_below_threshold",
                "noninferiority": noninferiority, "relative_projection_margin": relative_margin}
    return {**common, "global_label": label, "winner_candidate_id": winner_id, "loser_candidate_id": loser_id,
            "preference_margin": raw_margin, "pair_confidence": confidence, "abstain_reason": None,
            "noninferiority": noninferiority, "relative_projection_margin": relative_margin}


def decide_segments(
    *,
    pair_id: str,
    candidate_a: str,
    candidate_b: str,
    score_a: Mapping[str, Any],
    score_b: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    frame_alignment_pass: bool,
) -> list[dict[str, Any]]:
    """逐个 4-frame window 产生局部标签；不强行对齐全局 winner。"""
    frame_count = score_a.get("frame_count")
    if not isinstance(frame_count, int) or frame_count < 4 or score_b.get("frame_count") != frame_count:
        raise ValueError("segment label 需要两个相同且至少四帧的 candidate score")
    output: list[dict[str, Any]] = []
    for start in range(frame_count - 3):
        end = start + 4
        energy_a, points_a = _window_energy(score_a, start, end)
        energy_b, points_b = _window_energy(score_b, start, end)
        coverage_a = _window_mean(score_a, "track_coverage_by_frame", start, end)
        coverage_b = _window_mean(score_b, "track_coverage_by_frame", start, end)
        motion_a = _window_mean(score_a, "motion_magnitude_by_frame", start, end)
        motion_b = _window_mean(score_b, "motion_magnitude_by_frame", start, end)
        survival_a = _window_mean(score_a, "survival_by_frame", start, end)
        survival_b = _window_mean(score_b, "survival_by_frame", start, end)
        decomposition = {
            "projection_energy_a": energy_a,
            "projection_energy_b": energy_b,
            "projection_points_a": points_a,
            "projection_points_b": points_b,
            "track_coverage_a": coverage_a,
            "track_coverage_b": coverage_b,
            "motion_magnitude_a": motion_a,
            "motion_magnitude_b": motion_b,
            "survival_a": survival_a,
            "survival_b": survival_b,
        }
        base = {
            "segment_id": f"{pair_id}-segment-{start:02d}",
            "pair_id": pair_id,
            "start_frame": start,
            "end_frame": end,
            "violation_decomposition": decomposition,
            "frame_alignment_pass": bool(frame_alignment_pass),
        }
        if not frame_alignment_pass or energy_a is None or energy_b is None:
            output.append({**base, "label": "abstain", "winner_candidate_id": None, "loser_candidate_id": None,
                           "confidence": 0.0, "abstain_reason": "frame_alignment_or_projection_invalid"})
            continue
        if math.isclose(energy_a, energy_b, rel_tol=0.0, abs_tol=float(thresholds["segment_tie_absolute_projection_margin"])):
            output.append({**base, "label": "tie", "winner_candidate_id": None, "loser_candidate_id": None,
                           "confidence": 0.0, "abstain_reason": "segment_projection_tie"})
            continue
        if energy_a < energy_b:
            label, winner, loser = "a_wins", candidate_a, candidate_b
            winner_energy, loser_energy = energy_a, energy_b
            winner_coverage, loser_coverage = coverage_a, coverage_b
            winner_motion, loser_motion = motion_a, motion_b
            winner_survival, loser_survival = survival_a, survival_b
        else:
            label, winner, loser = "b_wins", candidate_b, candidate_a
            winner_energy, loser_energy = energy_b, energy_a
            winner_coverage, loser_coverage = coverage_b, coverage_a
            winner_motion, loser_motion = motion_b, motion_a
            winner_survival, loser_survival = survival_b, survival_a
        margin = loser_energy - winner_energy
        relative_margin = margin / max(abs(loser_energy), 1.0e-8)
        constraints = {
            "margin": margin >= float(thresholds["minimum_segment_absolute_projection_margin"])
            and relative_margin >= float(thresholds["minimum_segment_relative_projection_margin"]),
            "coverage": winner_coverage is not None and loser_coverage is not None
            and winner_coverage >= loser_coverage - float(thresholds["maximum_segment_coverage_drop"]),
            "motion": winner_motion is not None and loser_motion is not None
            and winner_motion >= loser_motion * (1.0 - float(thresholds["maximum_segment_motion_drop_fraction"])),
            "survival": winner_survival is not None and loser_survival is not None
            and winner_survival >= loser_survival - float(thresholds["maximum_segment_survival_drop"]),
        }
        if not all(constraints.values()):
            output.append({**base, "label": "abstain", "winner_candidate_id": None, "loser_candidate_id": None,
                           "confidence": 0.0, "abstain_reason": "segment_margin_or_noninferiority_conflict",
                           "constraints": constraints})
            continue
        confidence = _clip01(relative_margin / max(float(thresholds["segment_confidence_reference_relative_margin"]), 1.0e-8))
        output.append({**base, "label": label, "winner_candidate_id": winner, "loser_candidate_id": loser,
                       "confidence": confidence, "abstain_reason": None, "constraints": constraints})
    return output


def select_condition_pair(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """在同一 condition 的两个 antithetic group 中按预注册 confidence 规则选一条。"""
    decisive = [row for row in pair_rows if str(row.get("global_label")) in DECISIVE_LABELS]
    if not decisive:
        return {
            "selected_pair_id": None,
            "selection_status": "abstain",
            "decisive_pair_count": 0,
            "selection_rule": "max_pair_confidence_then_pair_id",
            "reason": "no_decisive_antithetic_pair",
        }
    ordered = sorted(
        decisive,
        key=lambda row: (-float(_finite(row.get("pair_confidence")) or 0.0), str(row.get("pair_id", ""))),
    )
    return {
        "selected_pair_id": str(ordered[0]["pair_id"]),
        "selection_status": "selected",
        "decisive_pair_count": len(decisive),
        "selection_rule": "max_pair_confidence_then_pair_id",
        "reason": None,
    }


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.959963984540054) -> float | None:
    """二项比例的双侧 95% Wilson 下界。"""
    if total <= 0 or successes < 0 or successes > total:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - radius)
