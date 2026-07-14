from __future__ import annotations

import torch

from motion_proj.diagnostics.evaluator_validity import (
    _machine_decision,
    perturb_video,
    perturbation_rank_correlations,
    spearman_rank_correlation,
    synthetic_videos,
)
from motion_proj.eval.independent_tracks import (
    CoTracker3IndependentEvaluator,
    _normalise_cotracker_outputs,
    camera_compensated_velocity,
    fit_affine_background,
)


def test_affine_camera_compensation_removes_known_global_translation():
    count, time = 6, 5
    base = torch.stack(torch.meshgrid(torch.tensor([10.0, 20.0]), torch.tensor([15.0, 25.0, 35.0]), indexing="ij"), dim=-1).reshape(-1, 2)
    points = torch.stack([base + torch.tensor([2.0 * index, -1.0 * index]) for index in range(time)], dim=1)
    visibility = torch.ones(count, time, dtype=torch.bool)
    affine, valid = fit_affine_background(points, visibility)
    velocity, usable = camera_compensated_velocity(points, visibility, affine)
    assert bool(valid.all())
    assert bool(usable.all())
    assert float(torch.nan_to_num(velocity).abs().max()) < 1.0e-3


def test_perturbations_keep_video_shape_and_range():
    frames = torch.linspace(-1, 1, steps=3 * 3 * 32 * 48).reshape(3, 3, 32, 48)
    for mode in ("photometric", "codec_quantization", "resize_roundtrip"):
        result = perturb_video(frames, mode)
        assert result.shape == frames.shape
        assert float(result.min()) >= -1.0 and float(result.max()) <= 1.0


def test_rank_and_synthetic_protocol_are_deterministic():
    assert spearman_rank_correlation([1, 2, 3], [2, 4, 6]) == 1.0
    videos = synthetic_videos(7, frames=6, height=48, width=64)
    assert set(videos) == {"constant_velocity", "constant_acceleration", "smooth_turn", "occlusion"}
    assert videos["occlusion"].shape == (6, 3, 48, 64)


def test_machine_gate_uses_cross_perturbation_ranks_instead_of_self_correlation():
    fields = (
        "survival_rate",
        "camera_compensated_image_plane_velocity_rms_px",
        "camera_compensated_image_plane_acceleration_rms_px",
        "camera_compensated_image_plane_jerk_rms_px",
    )
    rows = []
    for value in (1.0, 2.0, 3.0):
        aggregate = {field: value for field in fields}
        rows.append({
            "valid": True,
            "aggregate": aggregate,
            "rerun": {"aggregate_relative_delta": {"max_relative_delta": 0.0}},
            "perturbations": {
                "photometric": {"aggregate": dict(aggregate)},
                "codec_quantization": {"aggregate": dict(aggregate)},
                "resize_roundtrip": {"aggregate": {field: 4.0 - value for field in fields}},
            },
        })
    ranks = perturbation_rank_correlations(rows, ("photometric", "codec_quantization", "resize_roundtrip"))
    assert ranks["photometric"]["camera_compensated_image_plane_acceleration_rms_px"] == 1.0
    assert ranks["resize_roundtrip"]["camera_compensated_image_plane_acceleration_rms_px"] == -1.0
    decision = _machine_decision(
        rows,
        {
            "threshold_sweep_rank_correlations": [1.0],
            "acceleration_order_correct": True,
            "jerk_order_correct": True,
            "occlusion_invalid_or_downweighted": True,
        },
        {"maximum_rerun_relative_delta": 0.02, "minimum_rank_correlation": 0.8},
        ranks,
    )
    assert not decision["checks"]["perturbation_rank_correlation"]
    assert not decision["machine_pass"]


def test_official_cotracker_output_layout_is_normalized_without_shape_guessing():
    tracks = torch.zeros(1, 3, 4, 2)
    visibility = torch.ones(1, 3, 4, dtype=torch.bool)
    points, visible = _normalise_cotracker_outputs(tracks, visibility)
    assert points.shape == (4, 3, 2)
    assert visible.shape == (4, 3)
    legacy_points, legacy_visible = _normalise_cotracker_outputs(tracks, visibility.unsqueeze(-1))
    assert torch.equal(points, legacy_points)
    assert torch.equal(visible, legacy_visible)


def test_missing_official_checkpoint_is_explicitly_unavailable(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# official entry placeholder\n", encoding="utf-8")
    evaluator = CoTracker3IndependentEvaluator({
        "repository_path": str(repo), "repository_commit": None,
        "checkpoint_path": str(tmp_path / "missing.pth"), "checkpoint_url": "https://official.invalid/weight",
        "grid_size": 8,
    })
    preflight = evaluator.preflight()
    assert not preflight["available"]
    assert any("checkpoint missing" in reason for reason in preflight["reasons"])
