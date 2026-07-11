import numpy as np
import pytest
import torch

from motion_proj.eval.drivinggen import (
    PROTOCOL, agent_consistency, scene_consistency, trajectory_consistency,
)
from motion_proj.eval.metrics import frechet_feature_distance


def test_drivinggen_protocol_is_explicitly_not_leaderboard_comparable():
    assert PROTOCOL["frames"] == 8
    assert PROTOCOL["nuscenes_dt_seconds"] == 0.5
    assert PROTOCOL["leaderboard_comparable"] is False


def test_scene_and_agent_consistency_are_one_for_constant_features():
    features = torch.ones(8, 4)
    assert scene_consistency(features) == pytest.approx(1.0)
    assert agent_consistency([features]) == pytest.approx(1.0)


def test_trajectory_formula_uses_half_second_interval():
    xy = np.stack([np.arange(8) * 0.5, np.zeros(8)], axis=-1)
    result = trajectory_consistency(xy, dt=0.5)
    assert result["speed_consistency"] == pytest.approx(1.0)
    assert result["acceleration_consistency"] == pytest.approx(1.0)


def test_frechet_distance_zero_for_identical_features():
    features = np.arange(24, dtype=float).reshape(6, 4)
    assert frechet_feature_distance(features, features) == pytest.approx(0.0, abs=1e-7)
