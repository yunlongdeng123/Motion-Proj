import json
from pathlib import Path

import numpy as np

from motion_proj.worldsim_v61.me3_predicted import (
    bind_native_actor_identity_without_geometry_fill,
    predicted_method_factors,
    resample_gaussianworld_classes,
)
from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN, VoxelGridSpec


def test_gaussianworld_class_mapping_is_fixed_and_outside_is_unknown() -> None:
    source = np.full((200, 200, 16), 17, dtype=np.uint8)
    source[0, 0, 0] = 0
    source[1, 0, 0] = 1
    source[2, 0, 0] = 16
    spec = VoxelGridSpec(
        frame="lidar_target",
        origin_m=(-50.0, -50.0, -5.0),
        voxel_size_m=0.5,
        shape=(4, 1, 18),
    )

    result = resample_gaussianworld_classes(source, spec)

    assert result["semantics"][0, 0, 0] == UNKNOWN
    assert result["semantics"][1, 0, 0] == OCCUPIED
    assert result["semantics"][2, 0, 0] == OCCUPIED
    assert result["semantics"][3, 0, 0] == FREE
    assert np.all(result["semantics"][:, :, 16:] == UNKNOWN)
    assert np.all(result["predicted_class"][:, :, 16:] == 255)
    assert not np.any(result["source_valid"][:, :, 16:])


def test_actor_binding_labels_only_existing_predicted_occupancy(tmp_path: Path) -> None:
    (tmp_path / "lidar_pose").mkdir()
    (tmp_path / "instances").mkdir()
    np.savetxt(tmp_path / "lidar_pose/000.txt", np.eye(4))
    instances = {
        "7": {
            "class_name": "vehicle",
            "frame_annotations": {
                "frame_idx": [0],
                "obj_to_world": [np.eye(4).tolist()],
                "box_size": [[2.0, 2.0, 2.0]],
            },
        }
    }
    (tmp_path / "instances/instances_info.json").write_text(
        json.dumps(instances), encoding="utf-8"
    )
    (tmp_path / "instances/frame_instances.json").write_text(
        json.dumps({"0": [7]}), encoding="utf-8"
    )
    spec = VoxelGridSpec(
        frame="lidar_target",
        origin_m=(-2.0, -2.0, -2.0),
        voxel_size_m=1.0,
        shape=(4, 4, 4),
    )
    semantics = np.full(spec.shape, UNKNOWN, dtype=np.uint8)
    semantics[2, 2, 2] = OCCUPIED
    frozen = semantics.copy()

    actor_grid, rows = bind_native_actor_identity_without_geometry_fill(
        semantics, spec, tmp_path, 0
    )

    np.testing.assert_array_equal(semantics, frozen)
    assert actor_grid[2, 2, 2] == 7
    assert np.count_nonzero(actor_grid >= 0) == 1
    assert rows[0]["predicted_occupied_inside_native_obb"] == 1
    assert rows[0]["geometry_created_by_identity_binding"] is False


def test_predicted_method_factors_keep_unknown_and_require_actor_identity() -> None:
    mask = np.ones((2, 2), dtype=bool)
    raycast = {
        "voxel_linear": np.asarray([[3, -1], [-1, -1]], dtype=np.int64),
        "actor_instance_id": np.asarray([[-1, -1], [-1, -1]], dtype=np.int32),
        "unknown_blocked": np.asarray([[False, True], [True, False]], dtype=bool),
    }
    static_case = {"hole_type": "missing_route_support"}
    actor_case = {"hole_type": "actor_removal_hole"}

    static, candidates, _ = predicted_method_factors(mask, raycast, static_case, 0.25)
    actor, _, _ = predicted_method_factors(mask, raycast, actor_case, 0.25)

    assert candidates.tolist() == [3]
    assert static["predicted_unknown_blocked_fraction"] == 0.5
    assert static["predicted_free_is_not_observed_free"] is True
    assert static["passed"] is True
    assert actor["passed"] is False
