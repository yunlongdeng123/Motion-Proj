from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v33.semantic_gate import (
    apply_gated_residual,
    build_semantic_gate,
    development_selection,
    evaluate_semantic_gate,
    semantic_mass,
    semantic_reintroduction_decision,
    validate_semantic_gate,
)


def _regions() -> dict[str, np.ndarray]:
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:28, 20:40] = True
    return build_semantic_gate(
        mask,
        boundary_inner_pixels=2,
        boundary_outer_pixels=3,
        contact_depth_pixels=6,
        contact_side_pixels=3,
        shadow_depth_pixels=10,
        shadow_side_pixels=8,
        boundary_weight=1.0,
        contact_weight=0.75,
        shadow_weight=0.35,
    )


def test_gate_regions_are_disjoint_from_far_and_bounded() -> None:
    regions = _regions()
    validate_semantic_gate(regions)
    assert np.all((regions["gate"] >= 0) & (regions["gate"] <= 1))
    assert not np.any(regions["allowed"] & regions["far_non_target"])
    assert regions["boundary_ring"].any()
    assert regions["ground_contact"].any()
    assert regions["shadow_support"].any()


def test_gated_residual_preserves_far_pixels_exactly_and_clamps() -> None:
    regions = _regions()
    raw = np.full((40, 60, 3), 100, dtype=np.uint8)
    candidate = np.full_like(raw, 200)
    output, audit = apply_gated_residual(
        raw, candidate, regions["gate"], residual_cap_uint8=12
    )
    np.testing.assert_array_equal(
        output[regions["far_non_target"]], raw[regions["far_non_target"]]
    )
    assert audit["maximum_applied_abs_residual_uint8"] == 12
    assert audit["changed_far_non_target_pixels"] == 0


def test_zero_gate_is_exact_identity() -> None:
    raw = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
    candidate = 255 - raw
    output, audit = apply_gated_residual(
        raw, candidate, np.zeros((6, 7), np.float32), residual_cap_uint8=8
    )
    np.testing.assert_array_equal(output, raw)
    assert audit["changed_pixels"] == 0


def test_invalid_gate_and_empty_footprint_fail_closed() -> None:
    with pytest.raises(ValueError, match="非空"):
        build_semantic_gate(
            np.zeros((4, 4), bool),
            boundary_inner_pixels=1,
            boundary_outer_pixels=1,
            contact_depth_pixels=1,
            contact_side_pixels=1,
            shadow_depth_pixels=1,
            shadow_side_pixels=1,
            boundary_weight=1,
            contact_weight=1,
            shadow_weight=1,
        )
    regions = _regions()
    regions["gate"][0, 0] = 2
    with pytest.raises(ValueError, match="数值"):
        validate_semantic_gate(regions)


def test_region_metrics_detect_improvement() -> None:
    regions = _regions()
    reference = np.full((40, 60, 3), 100, dtype=np.uint8)
    raw = reference.copy()
    raw[regions["boundary_ring"]] = 130
    gated = raw.copy()
    gated[regions["boundary_ring"]] = 110
    metrics = evaluate_semantic_gate(
        raw=raw, gated=gated, reference=reference, regions=regions
    )
    assert metrics["boundary_ring"]["l1_delta"] < 0
    assert metrics["boundary_ring"]["gated_psnr_db"] > metrics["boundary_ring"]["raw_psnr_db"]


def test_semantic_mass_and_reintroduction_detector() -> None:
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    raw = semantic_mass(np.full((8, 8), -2, np.float32), mask)
    candidate = semantic_mass(np.full((8, 8), 2, np.float32), mask)
    production = semantic_mass(np.full((8, 8), -2, np.float32), mask)
    decision = semantic_reintroduction_decision(
        raw=raw,
        unconstrained=candidate,
        production=production,
        minimum_candidate_mass_increase=0.01,
        minimum_candidate_positive_fraction_increase=0.01,
        maximum_production_mass_increase=1e-6,
        maximum_production_positive_fraction_increase=1e-6,
    )
    assert decision["unconstrained_candidate_flagged"]
    assert decision["production_safe"]


def _selection_row(delta: float, *, far_changed: int = 0) -> dict:
    return {
        "metrics": {
            "boundary_ring": {"l1_delta": delta},
            "ground_contact": {"l1_delta": delta},
            "actor_interior": {"l1_delta": 0.1},
        },
        "blend_audit": {
            "changed_far_non_target_pixels": far_changed,
            "maximum_applied_abs_residual_uint8": 8,
        },
        "delete_raw_production_exact": True,
        "semantic_reintroduction": {"production_safe": True},
    }


def _selection_gates() -> dict[str, float]:
    return {
        "maximum_mean_boundary_l1_degradation": 0,
        "maximum_mean_contact_l1_degradation": 0,
        "maximum_mean_actor_interior_l1_degradation": 0.5,
        "minimum_mean_l1_improvement": 0.05,
        "residual_cap_uint8": 12,
    }


def test_development_selects_gate_only_with_improvement_and_safeguards() -> None:
    selected = development_selection([_selection_row(-0.2)], _selection_gates())
    assert selected["selected_arm"] == "G1_semantic_gate"
    assert selected["accepted"]


def test_development_falls_back_on_no_gain_or_far_drift() -> None:
    no_gain = development_selection([_selection_row(0.0)], _selection_gates())
    assert no_gain["selected_arm"] == "G0_raw_3d"
    drift = development_selection(
        [_selection_row(-0.2, far_changed=1)], _selection_gates()
    )
    assert drift["selected_arm"] == "G0_raw_3d"
