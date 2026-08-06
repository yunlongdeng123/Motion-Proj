from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from motion_proj.worldsim_v3.lidar_provenance import (
    actor_input_mapping,
    compare_initialization_provenance,
    raw_lidar_block_contract,
    sparse_depth_residuals,
    validate_lidar_provenance_contract,
)


def test_lidar_provenance_contract_is_frozen() -> None:
    payload = yaml.safe_load(
        Path("configs/worldsim_v3/a1_lidar_provenance_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_lidar_provenance_contract(payload)
    payload["runtime_initialization"]["seed"] = 1
    with pytest.raises(ValueError, match="seed"):
        validate_lidar_provenance_contract(payload)


def test_raw_lidar_block_contract_counts_float32_xyzi(tmp_path: Path) -> None:
    scan = tmp_path / "000.bin"
    pose = tmp_path / "000.txt"
    np.arange(20, dtype=np.float32).tofile(scan)
    np.savetxt(pose, np.eye(4))
    result = raw_lidar_block_contract(scan, pose, frame=0)
    assert result["raw_point_count"] == 5
    assert len(result["scan_sha256"]) == 64
    assert len(result["pose_sha256"]) == 64


def test_actor_input_mapping_joins_true_id_and_selected_role() -> None:
    samples = {
        "13": {
            "node_type": "RigidNodes",
            "point_count": 12,
            "points_sha256": "p",
            "colors_sha256": "c",
            "visible_frame_count": 7,
        }
    }
    actors = [
        {
            "processed_true_instance_id": 13,
            "rigid_model_index": 5,
            "instance_token": "token",
            "class_name": "vehicle.car",
            "processed_frame_count": 7,
        }
    ]
    selected = {
        "high-support": {"availability": "available", "rigid_model_index": 5}
    }
    result = actor_input_mapping(samples, actors, selected)
    assert result[0]["selected_roles"] == ["high-support"]
    assert result[0]["initial_gaussian_count"] == 12


def test_sparse_depth_residual_filters_opacity_and_invalid_depth() -> None:
    lidar = np.array([[10.0, 20.0], [0.0, 5.0]])
    rendered = np.array([[12.0, 18.0], [4.0, np.nan]])
    opacity = np.array([[0.9, 0.01], [0.9, 0.9]])
    result = sparse_depth_residuals(
        rendered_depth=rendered,
        rendered_opacity=opacity,
        lidar_depth=lidar,
        lidar_valid_minimum_meters=0.001,
        rendered_valid_minimum_meters=0.001,
        minimum_rendered_opacity=0.05,
    )
    assert result["candidate_count"] == 3
    assert result["valid_count"] == 1
    np.testing.assert_allclose(result["absolute_residual_meters"], [2.0])
    np.testing.assert_allclose(result["relative_residual"], [0.2])


def test_compare_initialization_separates_recorded_inputs_from_random_count() -> None:
    source = {
        "background_lidar_sample": {
            "point_count": 2,
            "points_sha256": "p",
            "colors_sha256": "c",
            "times_sha256": None,
        },
        "instance_lidar_samples": {
            "1": {
                "node_type": "RigidNodes",
                "point_count": 1,
                "points_sha256": "ip",
                "colors_sha256": "ic",
                "visible_frame_count": 3,
            }
        },
        "initialized_gaussians": {"Background": 10, "RigidNodes": 1},
        "rng_reset": {"seed": 0},
        "limitations": [],
    }
    reproduced = {
        **source,
        "initialized_gaussians": {"Background": 12, "RigidNodes": 1},
    }
    result = compare_initialization_provenance(source, reproduced)
    assert result["recorded_lidar_actor_inputs_exact"] is True
    assert result["exact_recorded_provenance"] is False
    assert result["background_initial_gaussian_count_delta"] == 2
