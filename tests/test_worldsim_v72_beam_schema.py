from __future__ import annotations

import numpy as np
import pytest

from motion_proj.worldsim_v72.data.beam_schema import (
    INVALID_RETURN_COUNT,
    BeamReturnTargets,
    decode_waymo_range_images,
)


def _decode(second_without_first: bool = False):
    height, width = 2, 3
    first = np.zeros((height, width, 4), dtype=np.float32)
    second = np.zeros_like(first)
    first[..., 0] = [[5.0, -1.0, 0.0], [7.0, 8.0, -1.0]]
    first[..., 1] = 0.25
    first[..., 2] = 0.5
    second[..., 0] = -1.0
    second[1, 0, 0] = 7.5
    if second_without_first:
        second[0, 1, 0] = 4.0
    origins = np.zeros((height, width, 3), dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 0] = 1.0
    valid = np.asarray([[True, True, False], [True, True, True]])
    beam_parameters = np.zeros((height, width, 2), dtype=np.float32)
    return decode_waymo_range_images(
        range_image_return1=first,
        range_image_return2=second,
        ray_origins_world_m=origins,
        ray_directions_world=directions,
        valid_emission_mask=valid,
        beam_parameters=beam_parameters,
        query_pose=np.eye(4),
        log_id="segment-1",
        frame_id="frame-1",
        sensor_id="TOP",
        time_ns=123,
        range_min_m=0.0,
        range_max_m=75.0,
        firing_validity_source="fixed_grid_mask_v1",
    )


def test_waymo_decoder_keeps_no_return_separate_from_invalid() -> None:
    queries, targets = _decode()
    assert len(queries) == len(targets) == 6
    assert targets.return_count.tolist() == [1, 0, int(INVALID_RETURN_COUNT), 2, 1, 0]
    assert targets.no_return_mask.tolist() == [False, True, False, False, False, True]
    assert queries.valid_emission_mask.tolist() == [True, True, False, True, True, True]
    assert len(set(queries.ray_id.tolist())) == 6
    assert np.isnan(targets.range_m[1]).all()
    assert np.isnan(targets.range_m[2]).all()
    assert targets.fingerprint == targets.fingerprint


def test_waymo_decoder_rejects_second_return_without_first() -> None:
    with pytest.raises(ValueError, match="第二回波"):
        _decode(second_without_first=True)


def test_beam_targets_reject_invalid_beam_labeled_as_no_return() -> None:
    with pytest.raises(ValueError, match="必须为 -1"):
        BeamReturnTargets(
            ray_id=np.asarray(["ray"]),
            target_known_mask=np.asarray([False]),
            return_count=np.asarray([0]),
            range_m=np.full((1, 2), np.nan),
            intensity=np.zeros((1, 2)),
            intensity_valid=np.zeros((1, 2), dtype=bool),
            elongation=np.zeros((1, 2)),
            no_label_zone=np.zeros((1, 2), dtype=bool),
            provenance={},
        )
