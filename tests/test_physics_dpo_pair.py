from __future__ import annotations

from types import SimpleNamespace

import torch

from motion_proj.auditor.state import Track
from motion_proj.diagnostics import physics_dpo_pair as pair_module
from motion_proj.diagnostics.physics_dpo_pair import (
    _constructor_coverage,
    _machine_status,
    make_renoise_delta,
)
from motion_proj.preference.pair_scoring import (
    candidate_feasibility,
    decide_global_pair,
    decide_segments,
    select_condition_pair,
    wilson_lower_bound,
)


THRESHOLDS = {
    "minimum_projection_points": 4,
    "minimum_primary_tracks": 1,
    "minimum_track_coverage": 0.5,
    "minimum_median_track_length": 4,
    "minimum_scorer_confidence": 0.5,
    "maximum_saturation_fraction": 0.99,
    "frame0_max_correction_px": 1.0e-6,
    "maximum_visibility_changed_count": 0,
    "maximum_survival_drop_to_base": 0.2,
    "minimum_motion_ratio_to_base": 0.5,
    "maximum_motion_ratio_to_base": 2.0,
    "minimum_displacement_ratio_to_base": 0.5,
    "maximum_displacement_ratio_to_base": 2.0,
    "tie_absolute_projection_margin": 1.0e-6,
    "minimum_absolute_projection_margin": 0.05,
    "minimum_relative_projection_margin": 0.05,
    "maximum_pair_survival_drop": 0.1,
    "maximum_pair_motion_drop_fraction": 0.1,
    "maximum_pair_displacement_drop_fraction": 0.1,
    "maximum_pair_coverage_drop": 0.1,
    "maximum_pair_saturation_increase": 0.01,
    "confidence_reference_relative_margin": 0.1,
    "minimum_pair_confidence": 0.1,
    "segment_tie_absolute_projection_margin": 1.0e-6,
    "minimum_segment_absolute_projection_margin": 0.02,
    "minimum_segment_relative_projection_margin": 0.02,
    "maximum_segment_coverage_drop": 0.1,
    "maximum_segment_motion_drop_fraction": 0.1,
    "maximum_segment_survival_drop": 0.1,
    "segment_confidence_reference_relative_margin": 0.05,
}


def _score(energy: float, *, motion: float = 1.0, survival: float = 0.9, coverage: float = 0.8) -> dict:
    return {
        "uses_future_gt": False,
        "projection_energy": energy,
        "projection_points": 12,
        "primary_track_count": 2,
        "track_coverage": coverage,
        "median_track_length_frames": 8,
        "scorer_confidence": 0.8,
        "punc_invariants": {"frame0_correction_max_px": 0.0, "visibility_changed_count": 0},
        "survival_rate": survival,
        "motion_magnitude": motion,
        "net_displacement": 2.0,
        "frame_count": 5,
        "projection_energy_by_frame": [None, energy, energy, energy, energy],
        "projection_points_by_frame": [0, 3, 3, 3, 3],
        "track_coverage_by_frame": [coverage] * 5,
        "motion_magnitude_by_frame": [motion] * 5,
        "survival_by_frame": [survival] * 5,
    }


def _quality() -> dict:
    return {"finite": True, "saturation_fraction": 0.1}


def test_candidate_feasibility_rejects_future_gt_and_motion_collapse() -> None:
    base = _score(1.0)
    valid = candidate_feasibility(_score(0.8), base, _quality(), THRESHOLDS)
    assert valid["feasible"]

    leaked = _score(0.8)
    leaked["uses_future_gt"] = True
    assert not candidate_feasibility(leaked, base, _quality(), THRESHOLDS)["feasible"]
    assert not candidate_feasibility(_score(0.8, motion=0.2), base, _quality(), THRESHOLDS)["feasible"]


def test_pair_requires_projection_margin_and_noninferiority() -> None:
    base = _score(1.0)
    a, b = _score(0.6), _score(1.0)
    fa = candidate_feasibility(a, base, _quality(), THRESHOLDS)
    fb = candidate_feasibility(b, base, _quality(), THRESHOLDS)
    result = decide_global_pair(
        candidate_a="a", candidate_b="b", score_a=a, score_b=b, feasibility_a=fa, feasibility_b=fb,
        quality_a=_quality(), quality_b=_quality(), thresholds=THRESHOLDS,
    )
    assert result["global_label"] == "a_wins"
    assert result["winner_candidate_id"] == "a"

    collapsed = _score(0.6, motion=0.7)
    bad = decide_global_pair(
        candidate_a="a", candidate_b="b", score_a=collapsed, score_b=b,
        feasibility_a=candidate_feasibility(collapsed, base, _quality(), THRESHOLDS), feasibility_b=fb,
        quality_a=_quality(), quality_b=_quality(), thresholds=THRESHOLDS,
    )
    assert bad["global_label"] == "abstain"
    assert bad["abstain_reason"] == "noninferiority_conflict"


def test_segment_labels_can_disagree_with_clip_label_without_invalid_schema() -> None:
    a, b = _score(0.5), _score(1.0)
    a["projection_energy_by_frame"] = [None, 0.2, 0.2, 1.2, 1.2]
    b["projection_energy_by_frame"] = [None, 1.0, 1.0, 0.2, 0.2]
    rows = decide_segments(
        pair_id="pair", candidate_a="a", candidate_b="b", score_a=a, score_b=b,
        thresholds=THRESHOLDS, frame_alignment_pass=True,
    )
    assert {row["label"] for row in rows} >= {"a_wins", "b_wins"}
    assert all(row["end_frame"] - row["start_frame"] == 4 for row in rows)


def test_condition_selection_uses_confidence_then_pair_id_without_branch_sign() -> None:
    selected = select_condition_pair([
        {"pair_id": "pair-g1", "global_label": "a_wins", "pair_confidence": 0.4},
        {"pair_id": "pair-g0", "global_label": "b_wins", "pair_confidence": 0.7},
    ])
    assert selected["selected_pair_id"] == "pair-g0"
    tie = select_condition_pair([
        {"pair_id": "pair-g1", "global_label": "a_wins", "pair_confidence": 0.7},
        {"pair_id": "pair-g0", "global_label": "b_wins", "pair_confidence": 0.7},
    ])
    assert tie["selected_pair_id"] == "pair-g0"


def test_wilson_lower_bound_is_conservative_and_finite() -> None:
    assert wilson_lower_bound(36, 48) is not None
    assert float(wilson_lower_bound(36, 48)) > 0.5
    assert wilson_lower_bound(0, 0) is None


def test_renoise_delta_reconstructs_normalized_seeded_noise() -> None:
    base_final = torch.linspace(-1.0, 1.0, 32).reshape(1, 2, 4, 4)
    prefix = torch.full_like(base_final, 0.25)
    sigma = 0.4

    delta, diagnostics = make_renoise_delta(base_final, prefix, sigma=sigma, seed=17)
    repeated, _ = make_renoise_delta(base_final, prefix, sigma=sigma, seed=17)
    different, _ = make_renoise_delta(base_final, prefix, sigma=sigma, seed=18)
    recovered_noise = (prefix + delta - base_final) / sigma

    assert torch.equal(delta, repeated)
    assert not torch.equal(delta, different)
    assert abs(float(recovered_noise.mean())) < 1.0e-6
    assert abs(float(torch.sqrt(torch.mean(recovered_noise.square()))) - 1.0) < 1.0e-6
    assert abs(diagnostics["epsilon_mean"]) < 1.0e-6
    assert abs(diagnostics["epsilon_rms"] - 1.0) < 1.0e-6


def test_punc_score_normalizes_mixed_track_devices(monkeypatch) -> None:
    original_device = "cuda" if torch.cuda.is_available() else "cpu"
    original_xyxy = torch.tensor(
        [[-1.0, -1.0, 1.0, 1.0], [0.0, -1.0, 2.0, 1.0], [3.0, -1.0, 5.0, 1.0],
         [2.0, -1.0, 4.0, 1.0], [3.0, -1.0, 5.0, 1.0]],
        device=original_device,
    )
    projected_xyxy = original_xyxy.detach().cpu().clone()
    projected_xyxy[2, [0, 2]] -= 2.0
    present = torch.ones(5, dtype=torch.bool, device=original_device)
    original = Track("track", "generated/dynamic_residual", original_xyxy, torch.ones(5, device=original_device), present)
    projected = Track("track", "generated/dynamic_residual", projected_xyxy, torch.ones(5), torch.ones(5, dtype=torch.bool))
    state = SimpleNamespace(
        tracks=[original], confidence=torch.ones(1, 5, device=original_device), diagnostics={}, uses_future_gt=False,
    )
    punc = SimpleNamespace(
        tracks=[projected], uncertainty=[torch.ones(5)], corrected=[torch.tensor([False, False, True, False, False])],
    )
    monkeypatch.setattr(pair_module, "build_candidate_tracks", lambda *args, **kwargs: {"P-UNC": punc})
    provider = SimpleNamespace(track=lambda frames: state)

    score, quality = pair_module._punc_score(
        torch.zeros(5, 3, 8, 8), provider,
        {"constrained": {"uncertainty_floor_px": 0.25, "uncertainty_confidence_scale_px": 1.5}},
    )

    assert score["projection_points"] == 1
    assert score["projection_energy"] == 4.0
    assert score["uses_future_gt"] is False
    assert quality["finite"] is True


def test_constructor_coverage_uses_two_raw_p1_pairs_per_condition() -> None:
    summary = {
        "P0-independent": {"pair_count": 3},
        "P1-common-prefix": {"pair_count": 6},
        "P2-base-renoise": {"pair_count": 3},
    }
    coverage = _constructor_coverage(summary, 3)
    assert coverage["pass"] is True
    assert coverage["expected_pair_counts"]["P1-common-prefix"] == 6

    summary["P1-common-prefix"]["pair_count"] = 3
    assert _constructor_coverage(summary, 3)["pass"] is False


def test_smoke_status_cannot_mask_failed_machine_check() -> None:
    assert _machine_status(smoke=True, checks={"schema": True, "coverage": True}) == "done"
    assert _machine_status(smoke=True, checks={"schema": True, "coverage": False}) == "blocked"
    assert _machine_status(smoke=False, checks={"schema": True}) == "awaiting_reviews"
