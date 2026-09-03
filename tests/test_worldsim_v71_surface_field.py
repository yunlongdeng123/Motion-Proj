import numpy as np
import pytest
import torch

from motion_proj.worldsim_v71.actor_canonical import (
    ActorCanonicalFrame,
    actor_to_world,
    assert_actor_state_immutable,
    split_frame_ranks,
    world_to_actor,
)
from motion_proj.worldsim_v71.actor_surface_field import occupied_collision_mask
from motion_proj.worldsim_v71.first_return_renderer import literal_first_return_partition
from motion_proj.worldsim_v71.ray_displacement import (
    apply_predicted_displacement,
    hard_collision_surface,
)
from motion_proj.worldsim_v71.surface_extract import extract_zero_crossing_surface


def test_coordinate_roundtrip_and_frame_roles():
    frame = ActorCanonicalFrame(
        center_world_m=np.asarray([10.0, -2.0, 1.0]),
        rotation_world_from_actor=np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    world = np.asarray([[11.0, -1.0, 1.5], [9.0, -2.5, 0.0]])
    np.testing.assert_allclose(actor_to_world(world_to_actor(world, frame), frame), world, atol=1.0e-8)
    build, heldout = split_frame_ranks(np.arange(6))
    np.testing.assert_array_equal(build, [True, True, False, True, True, False])
    np.testing.assert_array_equal(heldout, ~build)


def test_observed_anchors_do_not_move():
    anchors = torch.tensor([[0.0, 0.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0, 0.0]])
    directions = torch.tensor([[1.0, 0.0, 0.0]])
    normals = torch.tensor([[0.0, 1.0, 0.0]])
    prediction = torch.tensor([[10.0, 10.0, -10.0]])
    moved, unknown = apply_predicted_displacement(
        candidates,
        directions,
        normals,
        prediction,
        maximum_ray_displacement_m=0.3,
        maximum_normal_displacement_m=0.2,
        actor_half_size_m=torch.tensor([2.0, 1.0, 1.0]),
        cuboid_padding_m=0.1,
    )
    surface = hard_collision_surface(anchors, moved, unknown)
    torch.testing.assert_close(surface[0], anchors[0])


def test_unknown_does_not_enter_collision_surface():
    logits = torch.tensor([[0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
    torch.testing.assert_close(occupied_collision_mask(logits), torch.tensor([True, False]))


def test_literal_first_return_detects_early_surface():
    result = literal_first_return_partition(
        np.asarray([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        lateral_tolerance_m=0.2,
        depth_tolerance_m=0.2,
        device=torch.device("cpu"),
    )
    assert result["early"].tolist() == [True]
    assert result["hit"].tolist() == [False]


def test_actor_state_immutability_contract():
    before = {"track_id": "actor", "trajectory_xyz_m": [[0.0, 0.0, 0.0]], "size_lwh_m": [4.0, 2.0, 1.5], "hazardous": True}
    assert_actor_state_immutable(before, dict(before))
    changed = {**before, "hazardous": False}
    with pytest.raises(ValueError):
        assert_actor_state_immutable(before, changed)


def test_scf_zero_crossing_excludes_unknown():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    scf = torch.tensor([[0.01, 0.01], [0.01, 0.01], [0.5, 0.5]])
    logits = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    surface = extract_zero_crossing_surface(
        points,
        scf,
        logits,
        planar_band_m=0.05,
        vertical_band_m=0.05,
    )
    torch.testing.assert_close(surface, points[:1])
