from __future__ import annotations

import math

import numpy as np
import pytest

from motion_proj.worldsim_v33.view_selection import (
    atomic_save_deterministic_npz,
    circular_distance_radians,
    rank_view_candidates,
    robust_unit_interval,
    select_view_sets,
)


WEIGHTS = {
    "area": 0.2,
    "mask": 0.2,
    "sharpness": 0.15,
    "visible": 0.2,
    "occlusion": 0.15,
    "truncation": 0.1,
}


def _row(frame: int, camera: int, yaw: float, quality: float) -> dict[str, float | int]:
    return {
        "frame": frame,
        "camera_id": camera,
        "yaw_radians": yaw,
        "projected_area_fraction": quality,
        "sharpness_laplacian_variance": quality * 100.0,
        "mask_confidence": quality,
        "visible_fraction": quality,
        "occlusion_score": 1.0 - quality,
        "truncation_score": 0.0,
    }


def test_circular_distance_wraps_at_pi() -> None:
    assert circular_distance_radians(math.radians(179), math.radians(-179)) == pytest.approx(
        math.radians(2)
    )


def test_robust_unit_interval_handles_constant_input() -> None:
    np.testing.assert_array_equal(robust_unit_interval([3.0, 3.0]), [0.5, 0.5])


def test_rank_view_candidates_is_pure_and_penalizes_occlusion() -> None:
    source = [_row(1, 0, 0.0, 0.9), _row(2, 0, 0.5, 0.6)]
    source[0]["occlusion_score"] = 1.0
    original = [dict(row) for row in source]
    ranked = rank_view_candidates(source, WEIGHTS)
    assert source == original
    assert ranked[0]["view_score"] < 0.8
    assert set(ranked[0]["view_score_components"]) == set(WEIGHTS)


def test_multi_view_selection_rewards_yaw_and_rejects_near_duplicates() -> None:
    rows = rank_view_candidates(
        [
            _row(10, 0, 0.00, 1.00),
            _row(11, 0, 0.01, 0.99),
            _row(40, 1, 1.70, 0.92),
            _row(80, 2, -1.65, 0.90),
            _row(120, 0, 3.05, 0.88),
        ],
        WEIGHTS,
    )
    selected = select_view_sets(
        rows,
        view_counts=(1, 2, 4),
        yaw_weight=1.0,
        temporal_weight=0.2,
        camera_weight=0.2,
        frame_span=195,
        minimum_same_camera_frame_gap=5,
        minimum_pairwise_yaw_degrees=5.0,
        beam_width=64,
    )
    assert [(row["frame"], row["camera_id"]) for row in selected[1]["selected_views"]] == [
        (10, 0)
    ]
    four = {(row["frame"], row["camera_id"]) for row in selected[4]["selected_views"]}
    assert (11, 0) not in four
    assert len(four) == 4


def test_selection_is_deterministic_under_input_permutation() -> None:
    rows = rank_view_candidates(
        [_row(10, 0, 0.0, 0.8), _row(30, 1, 1.0, 0.8), _row(50, 2, 2.0, 0.8)],
        WEIGHTS,
    )
    kwargs = dict(
        view_counts=(2,),
        yaw_weight=0.5,
        temporal_weight=0.2,
        camera_weight=0.2,
        frame_span=195,
        minimum_same_camera_frame_gap=5,
        minimum_pairwise_yaw_degrees=1.0,
        beam_width=32,
    )
    first = select_view_sets(rows, **kwargs)
    second = select_view_sets(list(reversed(rows)), **kwargs)
    assert first == second


def test_actor_npz_writer_is_byte_exact(tmp_path) -> None:
    arrays = {
        "means": np.arange(9, dtype=np.float32).reshape(3, 3),
        "opacity": np.array([0.2, 0.8], dtype=np.float32),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    atomic_save_deterministic_npz(first, arrays)
    atomic_save_deterministic_npz(second, dict(reversed(list(arrays.items()))))
    assert first.read_bytes() == second.read_bytes()
