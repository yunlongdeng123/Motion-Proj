from __future__ import annotations

import numpy as np

from scripts.evaluate_worldsim_v33_s3_actor_assets import (
    development_decision,
    heldout_confirmation,
    largest_component_fraction,
)


def _metric(iou: float, lpips: float, psnr: float, boundary: float) -> dict[str, float]:
    return {
        "mean_silhouette_iou": iou,
        "mean_masked_crop_lpips_alex": lpips,
        "mean_masked_rgb_psnr": psnr,
        "mean_boundary_f1_tolerance_3px": boundary,
        "max_non_target_original_delete_l1_uint8": 0.01,
        "max_lateral_fragmentation": 0.02,
    }


def _config() -> dict:
    return {
        "selection": {
            "auto_arms": ["A1_auto_1view", "A2_auto_2view", "A4_auto_4view"],
            "development_metric_order": ["mean_silhouette_iou"],
            "acceptance_vs_manual_a0": {
                "maximum_iou_degradation": 0.03,
                "maximum_boundary_f1_degradation": 0.05,
                "maximum_psnr_degradation_db": 0.5,
                "maximum_lpips_increase": 0.02,
                "maximum_outside_l1_uint8": 0.2,
                "maximum_lateral_fragmentation": 0.15,
            },
            "heldout_confirmation": {
                "maximum_iou_degradation": 0.04,
                "maximum_boundary_f1_degradation": 0.06,
                "maximum_psnr_degradation_db": 0.75,
                "maximum_lpips_increase": 0.025,
            },
        }
    }


def test_largest_component_fraction_reports_fragmentation() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[4, 6] = True
    count, fraction = largest_component_fraction(mask)
    assert count == 2
    assert fraction == 0.8


def test_development_decision_selects_best_accepted_auto_arm() -> None:
    metrics = {
        "A0_manual_2view": _metric(0.70, 0.10, 16.0, 0.50),
        "A1_auto_1view": _metric(0.69, 0.11, 15.8, 0.48),
        "A2_auto_2view": _metric(0.73, 0.09, 16.2, 0.51),
        "A4_auto_4view": _metric(0.72, 0.08, 16.3, 0.52),
    }
    decision = development_decision(metrics, _config())
    assert decision["selected_arm"] == "A2_auto_2view"
    assert decision["heldout_read"] is False


def test_development_decision_falls_back_when_all_auto_arms_fail() -> None:
    metrics = {
        "A0_manual_2view": _metric(0.70, 0.10, 16.0, 0.50),
        "A1_auto_1view": _metric(0.40, 0.30, 12.0, 0.20),
        "A2_auto_2view": _metric(0.40, 0.30, 12.0, 0.20),
        "A4_auto_4view": _metric(0.40, 0.30, 12.0, 0.20),
    }
    assert development_decision(metrics, _config())["selected_arm"] == "A0_manual_2view"


def test_heldout_confirmation_is_relative_to_manual_baseline() -> None:
    metrics = {
        "A0_manual_2view": _metric(0.70, 0.10, 16.0, 0.50),
        "A2_auto_2view": _metric(0.68, 0.11, 15.8, 0.47),
    }
    decision = heldout_confirmation(metrics, "A2_auto_2view", _config())
    assert decision["accepted"] is True
    assert decision["deltas"]["iou"] < 0


def test_decisions_support_native_checkpoint_baseline() -> None:
    config = _config()
    config["selection"].update(
        {
            "baseline_arm": "A0_native",
            "native_baseline_arm": "A0_native",
            "auto_arms": ["A4_auto_4view"],
            "acceptance_vs_baseline": config["selection"][
                "acceptance_vs_manual_a0"
            ],
        }
    )
    metrics = {
        "A0_native": _metric(0.72, 0.10, 16.0, 0.50),
        "A4_auto_4view": _metric(0.73, 0.11, 15.9, 0.52),
    }
    development = development_decision(metrics, config)
    assert development["baseline_arm"] == "A0_native"
    assert development["selected_arm"] == "A4_auto_4view"
    heldout = heldout_confirmation(metrics, "A4_auto_4view", config)
    assert heldout["baseline_arm"] == "A0_native"
    assert heldout["accepted"] is True
