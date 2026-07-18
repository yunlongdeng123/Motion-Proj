"""B0 natural-rollout 的候选合法性、独立排序与 anti-collapse 聚合。"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


class NaturalRolloutRankingError(RuntimeError):
    """B0 candidate pool、ranking 或 gate 输入不合法。"""


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 1.0e-12:
        return None
    return numerator / denominator


def _median(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(np.asarray(finite, dtype=np.float64))) if finite else None


def pair_key(left: str, right: str) -> str:
    return "|".join(sorted((str(left), str(right))))


def generic_smoothness_energy(training_score: Mapping[str, Any]) -> float | None:
    """由 RAFT camera-compensated dynamics 得到无 P-UNC projector 的 generic score。"""
    diagnostics = training_score.get("provider_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    velocity = _finite(diagnostics.get("velocity_rms_px"))
    acceleration = _finite(diagnostics.get("acceleration_rms_px"))
    jerk = _finite(diagnostics.get("jerk_rms_px"))
    if velocity is None or acceleration is None or jerk is None or velocity <= 1.0e-6:
        return None
    return acceleration / velocity + 0.5 * jerk / velocity


def candidate_eligibility(
    row: Mapping[str, Any],
    base: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """相对同 condition 固定 Base 的训练侧合法性门禁。"""
    score = row.get("training_score") if isinstance(row.get("training_score"), Mapping) else {}
    base_score = base.get("training_score") if isinstance(base.get("training_score"), Mapping) else {}
    pixel = row.get("pixel_metrics") if isinstance(row.get("pixel_metrics"), Mapping) else {}
    base_pixel = base.get("pixel_metrics") if isinstance(base.get("pixel_metrics"), Mapping) else {}
    invariants = score.get("punc_invariants") if isinstance(score.get("punc_invariants"), Mapping) else {}

    coverage = _finite(score.get("track_coverage"))
    length = _finite(score.get("median_track_length_frames"))
    confidence = _finite(score.get("scorer_confidence"))
    survival = _finite(score.get("survival_rate"))
    base_survival = _finite(base_score.get("survival_rate"))
    motion = _finite(pixel.get("dynamic_degree_mean_abs_rgb"))
    base_motion = _finite(base_pixel.get("dynamic_degree_mean_abs_rgb"))
    sharpness = _finite(pixel.get("spatial_laplacian_energy"))
    base_sharpness = _finite(base_pixel.get("spatial_laplacian_energy"))
    flicker = _finite(pixel.get("global_luma_flicker_p95"))
    base_flicker = _finite(base_pixel.get("global_luma_flicker_p95"))
    first_psnr = _finite(pixel.get("first_frame_psnr_db"))
    base_first_psnr = _finite(base_pixel.get("first_frame_psnr_db"))
    saturation = _finite(row.get("saturation_fraction"))
    motion_ratio = _ratio(motion, base_motion)
    sharpness_ratio = _ratio(sharpness, base_sharpness)
    flicker_ratio = _ratio(flicker, base_flicker)
    generic = _finite(row.get("generic_smoothness_energy"))
    checks = {
        "no_future_gt": score.get("uses_future_gt") is False and row.get("uses_future_gt") is False,
        "punc_valid": bool(score.get("valid")) and _finite(score.get("projection_energy")) is not None,
        "finite_rgb": _finite(pixel.get("finite_fraction")) == 1.0,
        "projection_support": int(score.get("projection_points", 0)) >= int(thresholds["minimum_projection_points"]),
        "primary_track_support": int(score.get("primary_track_count", 0)) >= int(thresholds["minimum_primary_tracks"]),
        "track_coverage": coverage is not None and coverage >= float(thresholds["minimum_track_coverage"]),
        "track_length": length is not None and length >= float(thresholds["minimum_median_track_length"]),
        "scorer_confidence": confidence is not None and confidence >= float(thresholds["minimum_scorer_confidence"]),
        "generic_score_finite": generic is not None,
        "first_frame_absolute": first_psnr is not None and first_psnr >= float(thresholds["minimum_first_frame_psnr_db"]),
        "first_frame_relative": (
            first_psnr is not None and base_first_psnr is not None
            and first_psnr >= base_first_psnr - float(thresholds["maximum_first_frame_psnr_drop_db"])
        ),
        "sharpness": sharpness_ratio is not None and sharpness_ratio >= float(thresholds["minimum_sharpness_ratio_to_base"]),
        "flicker": (
            flicker is not None and base_flicker is not None
            and (
                flicker_ratio is not None and flicker_ratio <= float(thresholds["maximum_flicker_ratio_to_base"])
                or flicker <= base_flicker + float(thresholds["maximum_flicker_absolute_increase"])
            )
        ),
        "motion_floor_and_ceiling": (
            motion_ratio is not None
            and float(thresholds["minimum_motion_ratio_to_base"]) <= motion_ratio
            <= float(thresholds["maximum_motion_ratio_to_base"])
        ),
        "survival": (
            survival is not None and base_survival is not None
            and survival >= base_survival - float(thresholds["maximum_survival_drop_to_base"])
        ),
        "not_saturated": saturation is not None and saturation <= float(thresholds["maximum_saturation_fraction"]),
        "punc_frame0_exact": (
            _finite(invariants.get("frame0_correction_max_px")) is not None
            and float(invariants["frame0_correction_max_px"]) <= float(thresholds["maximum_punc_frame0_correction_px"])
        ),
        "punc_visibility_preserved": (
            isinstance(invariants.get("visibility_changed_count"), int)
            and int(invariants["visibility_changed_count"]) <= int(thresholds["maximum_punc_visibility_changed_count"])
        ),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "motion_ratio_to_base": motion_ratio,
        "sharpness_ratio_to_base": sharpness_ratio,
        "flicker_ratio_to_base": flicker_ratio,
        "survival_drop_to_base": (
            base_survival - survival if base_survival is not None and survival is not None else None
        ),
    }


def condition_diversity(
    rows: Sequence[Mapping[str, Any]],
    pairwise_rgb_rms: Mapping[str, float],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = sorted(
        (row for row in rows if row.get("candidate_role") != "base_fixed" and bool(row.get("eligible"))),
        key=lambda row: int(row["candidate_index"]),
    )
    pair_values = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1:]:
            value = _finite(pairwise_rgb_rms.get(pair_key(str(left["candidate_id"]), str(right["candidate_id"]))))
            if value is not None:
                pair_values.append(value)
    dynamic = [_finite(row.get("pixel_metrics", {}).get("dynamic_degree_mean_abs_rgb")) for row in eligible]
    punc = [_finite(row.get("training_score", {}).get("projection_energy")) for row in eligible]
    dynamic_finite = [value for value in dynamic if value is not None]
    punc_finite = [value for value in punc if value is not None]
    dynamic_span = (
        (max(dynamic_finite) - min(dynamic_finite)) / max(float(np.median(dynamic_finite)), 1.0e-12)
        if len(dynamic_finite) >= 2 else None
    )
    punc_span = (
        (max(punc_finite) - min(punc_finite)) / max(float(np.median(punc_finite)), 1.0e-12)
        if len(punc_finite) >= 2 else None
    )
    max_rgb_rms = max(pair_values) if pair_values else None
    checks = {
        "minimum_eligible_candidates": len(eligible) >= int(thresholds["minimum_eligible_selection_candidates"]),
        "rgb_nonduplicate": (
            max_rgb_rms is not None and max_rgb_rms >= float(thresholds["minimum_pairwise_rgb_rms"])
        ),
        "motion_or_score_spread": (
            dynamic_span is not None and dynamic_span >= float(thresholds["minimum_candidate_metric_span"])
        ) or (
            punc_span is not None and punc_span >= float(thresholds["minimum_candidate_metric_span"])
        ),
    }
    return {
        "diverse": all(checks.values()),
        "checks": checks,
        "eligible_candidate_ids": [str(row["candidate_id"]) for row in eligible],
        "eligible_count": len(eligible),
        "maximum_pairwise_rgb_rms": max_rgb_rms,
        "dynamic_relative_span": dynamic_span,
        "punc_relative_span": punc_span,
    }


def eligibility_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    pairwise_rgb_rms_by_condition: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, Any],
    *,
    ignored_checks: Sequence[str],
) -> dict[str, Any]:
    """只读移除指定 checks，给出 candidate-support 上界；不重选或重算 scorer。"""
    ignored = {str(name) for name in ignored_checks}
    adjusted = []
    for source in rows:
        row = dict(source)
        eligibility = source.get("eligibility") if isinstance(source.get("eligibility"), Mapping) else {}
        checks = eligibility.get("checks") if isinstance(eligibility.get("checks"), Mapping) else {}
        retained = {str(name): bool(value) for name, value in checks.items() if str(name) not in ignored}
        row["eligible"] = bool(retained) and all(retained.values())
        adjusted.append(row)
    natural = [row for row in adjusted if row.get("candidate_role") != "base_fixed"]
    passing = [row for row in natural if bool(row.get("eligible"))]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in adjusted:
        grouped[str(row["condition_id"])].append(row)
    eligible_by_condition = Counter(str(row["condition_id"]) for row in passing)
    diversity = [
        {
            "condition_id": condition_id,
            **condition_diversity(group, pairwise_rgb_rms_by_condition.get(condition_id, {}), thresholds),
        }
        for condition_id, group in sorted(grouped.items())
    ]
    return {
        "ignored_checks": sorted(ignored),
        "eligible_candidate_count": len(passing),
        "condition_with_at_least_one_eligible": sum(count >= 1 for count in eligible_by_condition.values()),
        "condition_with_at_least_two_eligible_upper_bound": sum(count >= 2 for count in eligible_by_condition.values()),
        "diverse_condition_count": sum(bool(row["diverse"]) for row in diversity),
        "diversity": diversity,
    }


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[cursor]]):
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = np.asarray(_rank(left), dtype=np.float64)
    right_rank = np.asarray(_rank(right), dtype=np.float64)
    if float(left_rank.std()) <= 0 or float(right_rank.std()) <= 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def select_condition_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_seed: int,
) -> dict[str, Any]:
    pool = sorted(
        (row for row in rows if row.get("candidate_role") != "base_fixed" and bool(row.get("eligible"))),
        key=lambda row: int(row["candidate_index"]),
    )
    base_rows = [row for row in rows if row.get("candidate_role") == "base_fixed"]
    if len(base_rows) != 1:
        raise NaturalRolloutRankingError("每个 condition 必须恰有一个 base_fixed")
    if not pool:
        return {
            "base_id": str(base_rows[0]["candidate_id"]), "random_id": None, "punc_best_id": None,
            "generic_best_id": None, "cotracker_oracle_id": None, "rank_spearman": None,
        }
    condition_id = str(base_rows[0]["condition_id"])
    punc_pool = [row for row in pool if _finite(row.get("training_score", {}).get("projection_energy")) is not None]
    generic_pool = [row for row in pool if _finite(row.get("generic_smoothness_energy")) is not None]
    punc_best = min(
        punc_pool,
        key=lambda row: (float(row["training_score"]["projection_energy"]), int(row["candidate_index"])),
    ) if punc_pool else None
    random_pool = [row for row in pool if punc_best is None or row["candidate_id"] != punc_best["candidate_id"]]
    if not random_pool:
        random_pool = pool
    digest = hashlib.sha256(f"{selection_seed}:{condition_id}".encode("utf-8")).digest()
    random_row = random_pool[int.from_bytes(digest[:8], "big") % len(random_pool)]
    generic_best = min(
        generic_pool,
        key=lambda row: (float(row["generic_smoothness_energy"]), int(row["candidate_index"])),
    ) if generic_pool else None
    oracle_pool = [
        row for row in pool
        if isinstance(row.get("cotracker"), Mapping)
        and bool(row["cotracker"].get("valid"))
        and _finite(row["cotracker"].get("plausibility_score")) is not None
    ]
    oracle_best = min(
        oracle_pool,
        key=lambda row: (float(row["cotracker"]["plausibility_score"]), int(row["candidate_index"])),
    ) if oracle_pool else None
    common = [
        row for row in pool
        if _finite(row.get("training_score", {}).get("projection_energy")) is not None
        and _finite(row.get("generic_smoothness_energy")) is not None
    ]
    correlation = spearman(
        [float(row["training_score"]["projection_energy"]) for row in common],
        [float(row["generic_smoothness_energy"]) for row in common],
    )
    return {
        "base_id": str(base_rows[0]["candidate_id"]),
        "random_id": str(random_row["candidate_id"]),
        "punc_best_id": str(punc_best["candidate_id"]) if punc_best else None,
        "generic_best_id": str(generic_best["candidate_id"]) if generic_best else None,
        "cotracker_oracle_id": str(oracle_best["candidate_id"]) if oracle_best else None,
        "rank_spearman": correlation,
        "eligible_pool_size": len(pool),
    }


def cotracker_plausibility_score(metrics: Mapping[str, Any]) -> float | None:
    velocity = _finite(metrics.get("image_plane_velocity_rms_px"))
    acceleration = _finite(metrics.get("image_plane_acceleration_rms_px"))
    jerk = _finite(metrics.get("image_plane_jerk_rms_px"))
    outlier = _finite(metrics.get("acceleration_outlier_fraction"))
    survival = _finite(metrics.get("survival_rate"))
    if None in (velocity, acceleration, jerk, outlier, survival) or velocity is None or velocity <= 1.0e-6:
        return None
    return acceleration / velocity + 0.5 * jerk / velocity + 2.0 * outlier + (1.0 - survival)


def pairwise_cotracker_preference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    cand_eval = candidate.get("cotracker") if isinstance(candidate.get("cotracker"), Mapping) else {}
    ref_eval = reference.get("cotracker") if isinstance(reference.get("cotracker"), Mapping) else {}
    ref_score = _finite(ref_eval.get("plausibility_score"))
    cand_score = _finite(cand_eval.get("plausibility_score"))
    if not bool(ref_eval.get("valid")) or ref_score is None:
        return {"comparable": False, "outcome": "invalid_reference", "win_credit": None, "checks": {}}
    if not bool(candidate.get("eligible")) or not bool(cand_eval.get("valid")) or cand_score is None:
        return {
            "comparable": True, "outcome": "loss", "win_credit": 0.0,
            "checks": {"candidate_valid": False}, "selected_catastrophic": True,
            "selected_low_motion": False, "relative_score_improvement": None,
        }
    cand_pixel = candidate.get("pixel_metrics") if isinstance(candidate.get("pixel_metrics"), Mapping) else {}
    ref_pixel = reference.get("pixel_metrics") if isinstance(reference.get("pixel_metrics"), Mapping) else {}
    cand_dynamic = _finite(cand_pixel.get("dynamic_degree_mean_abs_rgb"))
    ref_dynamic = _finite(ref_pixel.get("dynamic_degree_mean_abs_rgb"))
    dynamic_ratio = _ratio(cand_dynamic, ref_dynamic)
    cand_survival = _finite(cand_eval.get("survival_rate"))
    ref_survival = _finite(ref_eval.get("survival_rate"))
    cand_coverage = _finite(cand_eval.get("track_coverage"))
    ref_coverage = _finite(ref_eval.get("track_coverage"))
    cand_sharpness = _finite(cand_pixel.get("spatial_laplacian_energy"))
    ref_sharpness = _finite(ref_pixel.get("spatial_laplacian_energy"))
    sharpness_ratio = _ratio(cand_sharpness, ref_sharpness)
    cand_flicker = _finite(cand_pixel.get("global_luma_flicker_p95"))
    ref_flicker = _finite(ref_pixel.get("global_luma_flicker_p95"))
    flicker_ratio = _ratio(cand_flicker, ref_flicker)
    cand_first = _finite(cand_pixel.get("first_frame_psnr_db"))
    ref_first = _finite(ref_pixel.get("first_frame_psnr_db"))
    checks = {
        "candidate_valid": True,
        "motion_floor": dynamic_ratio is not None and dynamic_ratio >= float(thresholds["minimum_pair_dynamic_ratio"]),
        "survival": (
            cand_survival is not None and ref_survival is not None
            and cand_survival >= ref_survival - float(thresholds["maximum_pair_survival_drop"])
        ),
        "coverage": (
            cand_coverage is not None and ref_coverage is not None
            and cand_coverage >= ref_coverage - float(thresholds["maximum_pair_coverage_drop"])
        ),
        "sharpness": sharpness_ratio is not None and sharpness_ratio >= float(thresholds["minimum_pair_sharpness_ratio"]),
        "flicker": (
            cand_flicker is not None and ref_flicker is not None
            and (
                flicker_ratio is not None and flicker_ratio <= float(thresholds["maximum_pair_flicker_ratio"])
                or cand_flicker <= ref_flicker + float(thresholds["maximum_pair_flicker_absolute_increase"])
            )
        ),
        "first_frame": (
            cand_first is not None and ref_first is not None
            and cand_first >= ref_first - float(thresholds["maximum_pair_first_frame_psnr_drop_db"])
        ),
    }
    safeguards = all(checks.values())
    relative = (ref_score - cand_score) / max(ref_score, 1.0e-12)
    margin = float(thresholds["cotracker_relative_tie_margin"])
    if not safeguards:
        outcome, credit = "loss", 0.0
    elif relative >= margin:
        outcome, credit = "win", 1.0
    elif relative <= -margin:
        outcome, credit = "loss", 0.0
    else:
        outcome, credit = "tie", 0.5
    return {
        "comparable": True,
        "outcome": outcome,
        "win_credit": credit,
        "checks": checks,
        "selected_catastrophic": not safeguards,
        "selected_low_motion": not checks["motion_floor"],
        "relative_score_improvement": relative,
        "dynamic_ratio": dynamic_ratio,
        "survival_delta": (
            cand_survival - ref_survival if cand_survival is not None and ref_survival is not None else None
        ),
        "candidate_score": cand_score,
        "reference_score": ref_score,
    }


def aggregate_b0_gate(
    candidate_rows: Sequence[Mapping[str, Any]],
    pairwise_rgb_rms_by_condition: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, Any],
    *,
    selection_seed: int,
) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    for row in candidate_rows:
        by_condition[str(row["condition_id"])].append(row)
    condition_rows = []
    random_comparisons = []
    base_comparisons = []
    punc_seed_indices = []
    correlations = []
    rank_disagreement_count = 0
    diverse_count = 0
    for condition_id in sorted(by_condition):
        rows = by_condition[condition_id]
        diversity = condition_diversity(
            rows, pairwise_rgb_rms_by_condition.get(condition_id, {}), thresholds,
        )
        diverse_count += int(diversity["diverse"])
        selection = select_condition_candidates(rows, selection_seed=int(selection_seed))
        punc = by_id.get(str(selection["punc_best_id"])) if selection["punc_best_id"] else None
        random = by_id.get(str(selection["random_id"])) if selection["random_id"] else None
        base = by_id[str(selection["base_id"])]
        generic = by_id.get(str(selection["generic_best_id"])) if selection["generic_best_id"] else None
        if punc is not None:
            punc_seed_indices.append(int(punc["candidate_index"]))
        if punc is not None and generic is not None and punc["candidate_id"] != generic["candidate_id"]:
            rank_disagreement_count += 1
        if selection.get("rank_spearman") is not None:
            correlations.append(float(selection["rank_spearman"]))
        random_result = (
            pairwise_cotracker_preference(punc, random, thresholds)
            if punc is not None and random is not None else {"comparable": False, "outcome": "missing"}
        )
        base_result = (
            pairwise_cotracker_preference(punc, base, thresholds)
            if punc is not None else {"comparable": False, "outcome": "missing"}
        )
        random_result = {"condition_id": condition_id, **random_result}
        base_result = {"condition_id": condition_id, **base_result}
        random_comparisons.append(random_result)
        base_comparisons.append(base_result)
        condition_rows.append(
            {
                "condition_id": condition_id,
                "diversity": diversity,
                "selection": selection,
                "punc_vs_random": random_result,
                "punc_vs_base": base_result,
            }
        )

    valid_random = [row for row in random_comparisons if bool(row.get("comparable"))]
    valid_base = [row for row in base_comparisons if bool(row.get("comparable"))]
    random_rate = (
        sum(float(row["win_credit"]) for row in valid_random) / len(valid_random) if valid_random else None
    )
    base_rate = sum(float(row["win_credit"]) for row in valid_base) / len(valid_base) if valid_base else None
    dynamic_ratios = [row.get("dynamic_ratio") for row in valid_random + valid_base]
    survival_deltas = [row.get("survival_delta") for row in valid_random + valid_base]
    catastrophic_count = sum(bool(row.get("selected_catastrophic")) for row in valid_random + valid_base)
    low_motion_count = sum(bool(row.get("selected_low_motion")) for row in valid_random + valid_base)
    positive = [
        max(float(row.get("relative_score_improvement") or 0.0), 0.0)
        for row in valid_random if row.get("outcome") == "win"
    ]
    positive_condition_count = sum(value > 0 for value in positive)
    positive_total = sum(positive)
    maximum_positive_share = max(positive) / positive_total if positive_total > 0 and positive else None
    seed_counts = Counter(punc_seed_indices)
    maximum_seed_fraction = max(seed_counts.values()) / len(punc_seed_indices) if punc_seed_indices else None
    median_dynamic_ratio = _median(dynamic_ratios)
    median_survival_delta = _median(survival_deltas)
    checks = {
        "candidate_diversity": diverse_count >= int(thresholds["minimum_diverse_conditions"]),
        "valid_random_comparisons": len(valid_random) >= int(thresholds["minimum_valid_comparison_conditions"]),
        "valid_base_comparisons": len(valid_base) >= int(thresholds["minimum_valid_comparison_conditions"]),
        "punc_beats_random": random_rate is not None and random_rate >= float(thresholds["minimum_punc_random_win_rate"]),
        "punc_beats_base": base_rate is not None and base_rate >= float(thresholds["minimum_punc_base_win_rate"]),
        "dynamic_noninferiority": (
            median_dynamic_ratio is not None
            and median_dynamic_ratio >= float(thresholds["minimum_systematic_dynamic_ratio"])
            and low_motion_count == 0
        ),
        "survival_noninferiority": (
            median_survival_delta is not None
            and median_survival_delta >= -float(thresholds["maximum_systematic_survival_drop"])
        ),
        "no_catastrophic_selection": catastrophic_count == 0,
        "scorers_not_equivalent": rank_disagreement_count >= int(thresholds["minimum_rank_disagreement_conditions"]),
        "improvement_not_few_condition_dominated": (
            positive_condition_count >= int(thresholds["minimum_positive_improvement_conditions"])
            and maximum_positive_share is not None
            and maximum_positive_share <= float(thresholds["maximum_single_condition_positive_share"])
        ),
        "no_seed_monopoly": (
            maximum_seed_fraction is not None
            and maximum_seed_fraction <= float(thresholds["maximum_winner_seed_fraction"])
        ),
    }
    return {
        "machine_pass": all(checks.values()),
        "checks": checks,
        "condition_count": len(by_condition),
        "diverse_condition_count": diverse_count,
        "valid_random_comparison_count": len(valid_random),
        "valid_base_comparison_count": len(valid_base),
        "punc_vs_random_win_credit_rate": random_rate,
        "punc_vs_base_win_credit_rate": base_rate,
        "median_dynamic_ratio": median_dynamic_ratio,
        "median_survival_delta": median_survival_delta,
        "selected_catastrophic_count": catastrophic_count,
        "selected_low_motion_count": low_motion_count,
        "rank_disagreement_condition_count": rank_disagreement_count,
        "median_punc_generic_spearman": _median(correlations),
        "positive_improvement_condition_count": positive_condition_count,
        "maximum_single_condition_positive_share": maximum_positive_share,
        "punc_winner_seed_counts": {str(key): value for key, value in sorted(seed_counts.items())},
        "maximum_winner_seed_fraction": maximum_seed_fraction,
        "conditions": condition_rows,
    }
