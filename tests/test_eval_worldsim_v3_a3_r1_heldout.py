from __future__ import annotations

import numpy as np

from scripts.eval_worldsim_v3_a3_r1_heldout import (
    aggregate_metric_rows,
    build_variant_aggregate,
    classify_exact_pareto,
    squared_rgb_error,
)


def metric_row(variant: str, role: str, edit: str, *, better: bool) -> dict:
    return {
        "role": role,
        "edit": edit,
        "frame": 10,
        "camera": 0,
        "variant": variant,
        "affected_pixels": 2,
        "s_b_t0_pixels": 2,
        "t1_valid_pixels": 2 if better else 1,
        "depth_order_violations": 0 if better else 1,
        "common_t1_valid_pixels": 1,
        "t0_abs_error_sum_m": 0.1 if better else 0.2,
        "non_target_squared_uint8_error": 10 if better else 20,
        "non_target_channel_elements": 30,
    }


def test_uint8_rgb_error_uses_only_frozen_mask() -> None:
    target = np.zeros((2, 2, 3), dtype=np.uint8)
    predicted = target.copy()
    predicted[0, 0] = [1, 2, 3]
    predicted[1, 1] = [10, 10, 10]
    mask = np.array([[True, False], [False, False]])
    squared, elements = squared_rgb_error(predicted, target, mask)
    assert squared == 1 + 4 + 9
    assert elements == 3


def test_metric_aggregation_counts_invalid_first_hit_in_frozen_denominator() -> None:
    aggregate = aggregate_metric_rows(
        [metric_row("r0", "high-support", "lateral", better=False)]
    )
    assert aggregate["s_b_first_hit_valid_coverage"] == 0.5
    assert aggregate["s_b_depth_order_violation_rate"] == 0.5
    assert aggregate["s_b_t0_first_hit_mae_m"] == 0.2


def test_variant_aggregate_and_exact_pareto_are_result_independent() -> None:
    groups = [
        "high-support::lateral",
        "high-support::delete",
        "boundary-support::lateral",
        "boundary-support::delete",
    ]
    rows = []
    for group in groups:
        role, edit = group.split("::")
        rows.append(metric_row("r0", role, edit, better=False))
        rows.append(metric_row("r1", role, edit, better=True))
    global_rows = [
        {
            "variant": "r0",
            "frame": 10,
            "camera": 0,
            "squared_uint8_error": 20,
            "channel_elements": 30,
        },
        {
            "variant": "r1",
            "frame": 10,
            "camera": 0,
            "squared_uint8_error": 10,
            "channel_elements": 30,
        },
    ]
    r0 = build_variant_aggregate(rows, global_rows, variant="r0", group_order=groups)
    r1 = build_variant_aggregate(rows, global_rows, variant="r1", group_order=groups)
    result = classify_exact_pareto(
        r0["primary_axes"],
        r1["primary_axes"],
        {
            "s_b_first_hit_valid_coverage": "higher",
            "s_b_depth_order_violation_rate": "lower",
            "non_target_observed_rgb_mse": "lower",
            "original_global_observed_rgb_mse": "lower",
        },
    )
    assert result["classification"] == "r1_dominates_r0_pass"
    assert result["r1_non_worse"] is True


def test_exact_pareto_reports_tradeoff_and_missing_evidence() -> None:
    tradeoff = classify_exact_pareto(
        {"a": 1.0, "b": 1.0},
        {"a": 0.5, "b": 1.5},
        {"a": "lower", "b": "lower"},
    )
    assert tradeoff["classification"] == "tradeoff_non_dominated"
    missing = classify_exact_pareto(
        {"a": None}, {"a": 1.0}, {"a": "lower"}
    )
    assert missing["classification"] == "insufficient_evidence"
