from __future__ import annotations

from types import SimpleNamespace

import torch

from motion_proj.preference.common_support import build_common_support
from motion_proj.preference.paired_tracks import PairedQuerySet, RawTrackObservation
from motion_proj.preference.residual_motion import compute_motion_component_evidence
from motion_proj.preference.selective_order import (
    decide_selective_relation,
    quality_comparability,
    video_quality_metrics,
)


RELATION_SETTINGS = {
    "maximum_camera_distance_px": 1.0,
    "maximum_active_fraction_drop": 0.05,
    "maximum_survival_drop": 0.05,
    "maximum_motion_drop_fraction": 0.10,
    "maximum_motion_increase_fraction": 1.0,
    "activity_equivalence_absolute": 0.05,
    "motion_equivalence_fraction": 0.10,
    "high_confidence_margin": 1.0,
}


def _manual_relation(*, winner_speed: float, camera: float = 0.0, quality: bool = True) -> dict:
    support = SimpleNamespace(valid=True, candidate_a="attack", candidate_b="reference", start_frame=0, end_frame=4)
    evidence = SimpleNamespace(
        valid=True,
        reason=None,
        camera_distance_px=camera,
        activity_a={
            "active_fraction": 1.0,
            "survival": 1.0,
            "residual_speed": winner_speed,
            "net_residual_displacement": 2.0 * winner_speed,
        },
        activity_b={
            "active_fraction": 1.0,
            "survival": 1.0,
            "residual_speed": 1.0,
            "net_residual_displacement": 2.0,
        },
    )
    intervals = {
        name: {"lower": 3.0, "upper": 4.0, "holm_significant": True}
        for name in ("punc", "acceleration", "curvature", "coherence")
    }
    return decide_selective_relation(
        condition_id="stress",
        edge_id="stress-edge",
        support=support,
        evidence=evidence,
        intervals=intervals,
        ropes={name: 1.0 for name in intervals},
        strict_threshold=2.0,
        quality={"comparable": quality},
        settings=RELATION_SETTINGS,
    )


def test_freeze_and_time_slow_attacks_cannot_be_strict_winners() -> None:
    freeze = _manual_relation(winner_speed=0.0)
    slow = _manual_relation(winner_speed=0.5)
    assert freeze["relation"] == "incomparable"
    assert slow["relation"] == "incomparable"
    assert freeze["reason"] == slow["reason"] == "activity_noninferiority"


def _query_set() -> PairedQuerySet:
    background = [[8.0 + (index % 4) * 24.0, 8.0 + (index // 4) * 24.0] for index in range(12)]
    dynamic = [[16.0 + (index % 4) * 24.0, 80.0 + (index // 4) * 20.0] for index in range(8)]
    return PairedQuerySet(
        points=torch.tensor(background + dynamic),
        query_ids=tuple([f"bg-{i:03d}" for i in range(12)] + [f"dyn-{i:03d}" for i in range(8)]),
        strata=tuple(["background"] * 12 + ["dynamic"] * 8),
        selection_scores=torch.ones(20),
        valid=True,
    )


def _observation(candidate: str, query_set: PairedQuerySet, *, jitter: bool = False) -> RawTrackObservation:
    points = query_set.points[:, None].repeat(1, 4, 1)
    time = torch.arange(4).float()
    points[:, :, 0] += time[None]
    points[12:, :, 0] += 0.5 * time[None]
    if jitter:
        points[12:, :, 1] += torch.tensor([0.0, 1.0, -1.0, 2.0])[None]
    visible = torch.ones(20, 4, dtype=torch.bool)
    return RawTrackObservation(
        candidate_id=candidate,
        query_set_hash=query_set.query_set_hash,
        raw_points=points,
        raw_visibility=visible,
        raw_confidence=torch.ones(20, 4),
        forward_backward_error=torch.zeros(20, 4),
    )


SUPPORT_SETTINGS = {
    "minimum_common_dynamic_queries": 8,
    "minimum_common_background_queries": 12,
    "minimum_common_dynamic_observations": 24,
    "minimum_spatial_clusters": 4,
    "minimum_frames_per_query": 3,
    "min_confidence": 0.5,
    "max_fb_error_px": 1.0,
    "cluster_cell_size_px": 20.0,
}


MOTION_SETTINGS = {
    "background_affine": {
        "minimum_points": 8,
        "irls_iterations": 3,
        "huber_delta": 2.5,
        "minimum_residual_px": 0.1,
        "ridge": 1.0e-4,
    },
    "curve_speed_floor_px": 0.1,
    "coherence_radius_px": 80.0,
    "coherence_minimum_neighbors": 2,
    "minimum_component_observations": 4,
    "punc": {
        "confidence_floor": 0.5,
        "uncertainty_floor_px": 0.25,
        "uncertainty_confidence_scale_px": 1.5,
        "snr_threshold": 0.1,
        "lambda_acceleration": 0.08,
        "lambda_jerk": 0.02,
        "robust_delta_px": 1.0,
        "robust_iterations": 3,
    },
}


def test_track_dropout_attack_reduces_support_instead_of_improving_denominator() -> None:
    query_set = _query_set()
    reference = _observation("reference", query_set)
    attacked = _observation("attack", query_set)
    visibility = attacked.raw_visibility.clone()
    visibility[12:, 2:] = False
    points = attacked.raw_points.clone()
    points[~visibility] = float("nan")
    attacked = RawTrackObservation(
        candidate_id="attack",
        query_set_hash=query_set.query_set_hash,
        raw_points=points,
        raw_visibility=visibility,
        raw_confidence=visibility.float(),
        forward_backward_error=torch.where(visibility, torch.zeros_like(visibility, dtype=torch.float32), torch.nan),
    )
    support = build_common_support(query_set, attacked, reference, SUPPORT_SETTINGS, window_starts=(0,))[0]
    assert not support.valid
    assert support.reason == "incomparable_support"


def test_camera_and_quality_attacks_are_incomparable() -> None:
    camera = _manual_relation(winner_speed=1.0, camera=5.0)
    assert camera["relation"] == "incomparable"
    assert camera["reason"] == "camera_mismatch"

    clean = video_quality_metrics(torch.zeros(4, 3, 16, 16))
    flicker_frames = torch.zeros(4, 3, 16, 16)
    flicker_frames[1::2] = 1.0
    attacked = video_quality_metrics(flicker_frames)
    gate = quality_comparability(clean, attacked, {
        "maximum_black_frame_fraction": 1.0,
        "maximum_saturation_fraction": 1.0,
        "maximum_temporal_l1": 0.2,
        "maximum_sharpness_jump": 1.0,
        "maximum_pair_saturation_difference": 1.0,
        "maximum_pair_temporal_l1_difference": 1.0,
        "minimum_pair_sharpness_ratio": 0.0,
        "maximum_pair_sharpness_ratio": 1.0e8,
    })
    assert not gate["comparable"]
    assert "b_temporal_jump" in gate["reasons"]


def test_common_affine_motion_transform_preserves_component_differences() -> None:
    query_set = _query_set()
    a = _observation("a", query_set)
    b = _observation("b", query_set, jitter=True)
    support = build_common_support(query_set, a, b, SUPPORT_SETTINGS, window_starts=(0,))[0]
    original = compute_motion_component_evidence(
        query_set, a, b, support, MOTION_SETTINGS, image_hw=(128, 128)
    )

    common = torch.arange(4).float()[None, :, None] * torch.tensor([2.0, 0.5])[None, None]
    a_transformed = RawTrackObservation(
        candidate_id="a",
        query_set_hash=query_set.query_set_hash,
        raw_points=a.raw_points + common,
        raw_visibility=a.raw_visibility,
        raw_confidence=a.raw_confidence,
        forward_backward_error=a.forward_backward_error,
    )
    b_transformed = RawTrackObservation(
        candidate_id="b",
        query_set_hash=query_set.query_set_hash,
        raw_points=b.raw_points + common,
        raw_visibility=b.raw_visibility,
        raw_confidence=b.raw_confidence,
        forward_backward_error=b.forward_backward_error,
    )
    transformed_support = build_common_support(
        query_set, a_transformed, b_transformed, SUPPORT_SETTINGS, window_starts=(0,)
    )[0]
    transformed = compute_motion_component_evidence(
        query_set, a_transformed, b_transformed, transformed_support, MOTION_SETTINGS, image_hw=(128, 128)
    )

    assert original.valid and transformed.valid
    for component in ("acceleration", "curvature", "coherence"):
        assert torch.allclose(
            original.differences[component],
            transformed.differences[component],
            atol=1.0e-4,
            rtol=1.0e-4,
            equal_nan=True,
        )
