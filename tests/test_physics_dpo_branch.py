from __future__ import annotations

import pytest
import torch

from motion_proj.diagnostics.physics_dpo_branch import (
    BranchPilotError,
    SEQUENCE_TRACE_FIELDS,
    calibrated_future_distance,
    choose_calibration_action,
    make_antithetic_perturbations,
    resolve_fork_step,
    track_correspondence,
    verify_shared_prefix_before_callback_injection,
)
from motion_proj.eval.independent_tracks import IndependentTrackState


def _state(*, shifted_query: float = 0.0, visible_last: bool = True) -> IndependentTrackState:
    points = torch.tensor(
        [
            [[2.0 + shifted_query, 3.0], [3.0, 3.0], [4.0, 3.0]],
            [[8.0 + shifted_query, 7.0], [8.0, 8.0], [8.0, 9.0]],
        ]
    )
    visibility = torch.tensor([[True, True, visible_last], [True, True, True]])
    points[~visibility] = float("nan")
    return IndependentTrackState(
        points=points,
        visibility=visibility,
        labels=["background", "dynamic_residual"],
        query_points=torch.tensor([[2.0 + shifted_query, 3.0], [8.0 + shifted_query, 7.0]]),
        affine_background=torch.zeros(2, 2, 3),
        diagnostics={},
        valid=True,
    )


def test_resolve_fork_step_matches_preregistered_25_step_fractions() -> None:
    assert resolve_fork_step(25, 0.4) == 10
    assert resolve_fork_step(25, 0.6) == 15
    assert resolve_fork_step(25, 0.8) == 20


def test_antithetic_perturbations_are_zero_mean_and_equal_norm_in_theory() -> None:
    prefix = torch.zeros(1, 14, 4, 3, 5, dtype=torch.bfloat16)
    perturbations = make_antithetic_perturbations(prefix, sigma_at_fork=3.5, strength_rho=0.01, direction_seed=7)

    assert set(perturbations) == {"g0-positive", "g0-negative", "g1-positive", "g1-negative"}
    rms = {round(float(row["theoretical_rms"]), 7) for row in perturbations.values()}
    assert len(rms) == 1
    assert all(abs(float(row["theoretical_mean"])) < 1.0e-6 for row in perturbations.values())
    assert torch.equal(
        perturbations["g0-positive"]["theoretical_delta"],
        -perturbations["g0-negative"]["theoretical_delta"],
    )


def test_calibrated_future_distance_rejects_floor_and_independent_seed_endpoints() -> None:
    passed = calibrated_future_distance(
        candidate_distance=0.2,
        rerun_floor=0.0,
        independent_distance=1.0,
        minimum_ratio=0.05,
        maximum_ratio=0.95,
    )
    assert passed["passed"] is True

    duplicate = calibrated_future_distance(
        candidate_distance=0.01,
        rerun_floor=0.0,
        independent_distance=1.0,
        minimum_ratio=0.05,
        maximum_ratio=0.95,
    )
    assert duplicate["reason"] == "candidate_indistinguishable"

    too_far = calibrated_future_distance(
        candidate_distance=0.98,
        rerun_floor=0.0,
        independent_distance=1.0,
        minimum_ratio=0.05,
        maximum_ratio=0.95,
    )
    assert too_far["reason"] == "candidate_too_independent"


def test_track_correspondence_requires_same_query_grid_and_reports_visibility_overlap() -> None:
    matched = track_correspondence(_state(), _state(visible_last=False), maximum_query_delta_px=0.01)
    assert matched["valid"] is True
    assert 0.0 < float(matched["coverage"]) < 1.0
    assert matched["label_agreement"] == 1.0

    mismatch = track_correspondence(_state(), _state(shifted_query=1.0), maximum_query_delta_px=0.01)
    assert mismatch["valid"] is False
    assert mismatch["reason"] == "query_grid_mismatch"


def test_calibration_action_is_sequential_not_grid_search() -> None:
    assert choose_calibration_action(
        current_strength_name="small",
        all_distance_indistinguishable=True,
        any_structure_mismatch=False,
        any_other_failure=True,
    ) == "increase_strength_to_medium"
    assert choose_calibration_action(
        current_strength_name="medium",
        all_distance_indistinguishable=True,
        any_structure_mismatch=False,
        any_other_failure=True,
    ) == "increase_strength_to_large"
    assert choose_calibration_action(
        current_strength_name="large",
        all_distance_indistinguishable=False,
        any_structure_mismatch=True,
        any_other_failure=True,
    ) == "adjust_fork_to_0.8"


def _trace_for_prefix_test() -> dict[str, object]:
    trace: dict[str, object] = {
        "condition_noise": torch.full((1, 2), 3.0),
        "initial_video_latents": torch.full((1, 2), 4.0),
    }
    for field_index, field in enumerate(SEQUENCE_TRACE_FIELDS):
        trace[field] = [
            torch.full((1, 2), float(field_index * 10 + step))
            for step in range(3)
        ]
    return trace


def test_common_prefix_verification_substitutes_only_callback_boundary_latent() -> None:
    official = _trace_for_prefix_test()
    sibling: dict[str, object] = {
        "condition_noise": official["condition_noise"].clone(),  # type: ignore[union-attr]
        "initial_video_latents": official["initial_video_latents"].clone(),  # type: ignore[union-attr]
    }
    for field in SEQUENCE_TRACE_FIELDS:
        sibling[field] = [item.clone() for item in official[field]]  # type: ignore[index,union-attr]
    pre = official["post_step_latents"][1].clone()  # type: ignore[index,union-attr]
    actual_delta = torch.full_like(pre, 0.25)
    post = pre + actual_delta
    sibling["post_step_latents"][1] = post  # type: ignore[index]
    injection = {
        "pre_injection_latent": pre,
        "actual_delta": actual_delta,
        "post_injection_latent": post,
    }

    result = verify_shared_prefix_before_callback_injection(
        official,
        sibling,
        injection,
        fork_step=2,
    )
    assert result["verified"] is True
    assert result["injection_semantics"] == "official_step_end_callback_after_transition"

    sibling["scheduler_inputs"][1] = sibling["scheduler_inputs"][1] + 1  # type: ignore[index,operator]
    with pytest.raises(BranchPilotError, match="scheduler_inputs"):
        verify_shared_prefix_before_callback_injection(
            official,
            sibling,
            injection,
            fork_step=2,
        )
