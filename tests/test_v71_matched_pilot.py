import numpy as np

from motion_proj.resim.matched_pilot import (
    apply_lateral_proposal,
    relative_kinematic_check,
    trajectory_bytes,
    trajectory_hash,
)


def _frames(count=40):
    output = []
    ego = {}
    for index in range(count):
        matrix = np.eye(4)
        matrix[0, 3] = 10.0 + index * 0.1
        matrix[1, 3] = 3.0
        output.append(
            {
                "frame_index": index,
                "T_world_actor": matrix.tolist(),
                "dimensions_lwh": [4.0, 1.8, 1.5],
            }
        )
        ego[index] = np.eye(4)
    return output, ego


def test_matched_requested_trajectory_is_deterministic():
    source, ego = _frames()
    first, metadata = apply_lateral_proposal(
        source, ego, peak_offset_m=0.8, timing="centered"
    )
    second, _ = apply_lateral_proposal(
        source, ego, peak_offset_m=0.8, timing="centered"
    )
    assert trajectory_bytes(first) == trajectory_bytes(second)
    assert trajectory_hash(first) == trajectory_hash(second)
    assert 0.79 <= metadata["realized_peak_offset_m"] <= 0.8


def test_scale_zero_is_exact_source_trajectory():
    source, ego = _frames()
    realized, _ = apply_lateral_proposal(
        source, ego, peak_offset_m=1.6, timing="late", scale=0.0
    )
    assert trajectory_bytes(realized) == trajectory_bytes(source)


def test_relative_kinematic_check_uses_edit_delta():
    source, ego = _frames()
    smooth, _ = apply_lateral_proposal(
        source, ego, peak_offset_m=0.4, timing="centered"
    )
    result = relative_kinematic_check(
        source,
        smooth,
        ego,
        dt_s=0.1,
        max_delta_speed_mps=3.0,
        max_delta_acceleration_mps2=2.5,
    )
    assert result["verdict"] == "PASS"

