"""ME-1 oracle occupancy 的决策与碰撞回归测试。"""

from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v61.me1_oracle import (
    _load_grid,
    fuse_two_factors,
    method_decision,
    me0_gate_passed,
    obb_intersects,
    occupancy_gate,
)


def test_r10_fusion_and_o2_decision_preserve_abstention() -> None:
    assert fuse_two_factors("ACCEPT", "ACCEPT") == "ACCEPT"
    assert fuse_two_factors("REJECT", "REJECT") == "REJECT"
    assert fuse_two_factors("ACCEPT", "REJECT") == "ABSTAIN"
    assert method_decision("ACCEPT", True) == "ACCEPT"
    assert method_decision("ACCEPT", False) == "ABSTAIN"
    assert method_decision("REJECT", True) == "REJECT"


def test_me0_gate_reads_nested_authority() -> None:
    assert me0_gate_passed({"checks": {"passed": True}})
    assert not me0_gate_passed({"passed": True})
    assert not me0_gate_passed({"checks": {"passed": False}})


def test_actor_zero_is_not_confused_with_empty_sentinel(tmp_path) -> None:
    path = tmp_path / "occ.npz"
    np.savez_compressed(
        path,
        static_semantics=np.zeros((2, 2, 2), dtype=np.uint8),
        actor_voxel_indices=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
        actor_instance_ids=np.asarray([0, 7], dtype=np.int32),
        grid_origin_m=np.zeros(3, dtype=np.float64),
        voxel_size_m=np.asarray(0.2, dtype=np.float64),
    )
    grid = _load_grid(path)
    assert grid["actor_grid"][0, 0, 0] == 0
    assert grid["actor_grid"][1, 1, 1] == 7
    assert grid["actor_grid"][0, 1, 0] == -1


def test_named_occupancy_gate_has_no_scalar_compensation() -> None:
    thresholds = {
        "maximum_free_space_conflict": 0.05,
        "minimum_observed_surface_support": 0.80,
        "maximum_unknown_volume_fraction": 0.20,
        "minimum_projected_surface_coverage": 0.50,
        "minimum_method_eval_depth_overlap": 0.50,
        "maximum_median_relative_depth_error": 0.20,
    }
    factors = {
        "free_space_conflict": 0.01,
        "observed_surface_support": 0.90,
        "unknown_volume_fraction": 0.10,
        "projected_surface_coverage": 0.70,
        "method_eval_depth_overlap": 0.60,
        "median_relative_depth_error": 0.10,
    }
    assert occupancy_gate(factors, thresholds)
    for name, failing in (
        ("free_space_conflict", 0.051),
        ("observed_surface_support", 0.799),
        ("unknown_volume_fraction", 0.201),
        ("projected_surface_coverage", 0.499),
        ("method_eval_depth_overlap", 0.499),
        ("median_relative_depth_error", 0.201),
    ):
        changed = dict(factors)
        changed[name] = failing
        assert not occupancy_gate(changed, thresholds)


def test_oriented_box_sat_distinguishes_contact_and_penetration() -> None:
    rotation = np.eye(3)
    size = np.asarray([2.0, 2.0, 2.0])
    assert obb_intersects(np.zeros(3), rotation, size, np.asarray([1.5, 0.0, 0.0]), rotation, size)
    assert not obb_intersects(np.zeros(3), rotation, size, np.asarray([2.0, 0.0, 0.0]), rotation, size)
    yaw = np.asarray(
        [[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
         [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
         [0.0, 0.0, 1.0]]
    )
    assert obb_intersects(np.zeros(3), rotation, size, np.asarray([1.0, 0.0, 0.0]), yaw, size)
