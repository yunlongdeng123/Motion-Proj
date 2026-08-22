"""P6 GaussianWorld 3090 smoke 的冻结合同测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from scripts.run_worldsim_v61_p6_gaussianworld_worker import build_lidar2img


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/worldsim_v61/p6_gaussianworld_smoke_v1.yaml"


def test_p6_binds_one_official_checkpoint_without_truth_or_calibration() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["task_id"] == "WS-V61-P6-GAUSSIANWORLD-3090-SMOKE-01"
    assert config["hypothesis_id"] == "WS-V61-H-P6-GW-001"
    assert config["sources"]["official_git_commit"] == (
        "b43629eaecffd5a7cbaac1a55517766e6263e4fc"
    )
    assert config["input"]["camera_names"] == [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    assert config["input"]["drivestudio_camera_ids"] == [0, 2, 1, 5, 3, 4]
    assert config["output_contract"]["grid_shape"] == [200, 200, 16]
    assert config["output_contract"]["pc_range_m"] == [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
    assert config["output_contract"]["noise_class"] == 0
    assert config["output_contract"]["empty_class"] == 17
    assert "dummy_label_is_shape_only_and_not_truth" in config["claim_boundary"]
    assert "no_surroundocc_label_read" in config["claim_boundary"]
    assert "no_training_calibration_confirmation_or_threshold_selection" in config["claim_boundary"]


def test_lidar_to_image_uses_drivestudio_camera_to_world_contract(tmp_path: Path) -> None:
    (tmp_path / "lidar_pose").mkdir()
    (tmp_path / "extrinsics").mkdir()
    (tmp_path / "intrinsics").mkdir()
    t_global_lidar = np.eye(4)
    t_global_lidar[0, 3] = 2.0
    t_global_camera = np.eye(4)
    t_global_camera[0, 3] = 1.0
    np.savetxt(tmp_path / "lidar_pose/052.txt", t_global_lidar)
    np.savetxt(tmp_path / "extrinsics/052_0.txt", t_global_camera)
    np.savetxt(tmp_path / "intrinsics/0.txt", np.asarray([2.0, 3.0, 4.0, 5.0]))
    actual = build_lidar2img(tmp_path, 52, [0])[0]
    expected = np.asarray(
        [
            [2.0, 0.0, 4.0, 2.0],
            [0.0, 3.0, 5.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(actual, expected)
