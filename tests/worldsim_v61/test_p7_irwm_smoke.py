"""P7 IR-WM current-state capability smoke 的冻结合同测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from scripts.run_worldsim_v61_p7_irwm_worker import (
    build_frame_metas,
    build_lidar2img,
    quaternion_yaw_degrees,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/worldsim_v61/p7_irwm_smoke_v1.yaml"
WORKER_PATH = REPO_ROOT / "scripts/run_worldsim_v61_p7_irwm_worker.py"


def test_p7_binds_one_official_current_state_checkpoint_without_truth() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["task_id"] == "WS-V61-P7-IRWM-3090-SMOKE-01"
    assert config["hypothesis_id"] == "WS-V61-H-P7-IRWM-001"
    assert config["sources"]["official_git_commit"] == (
        "a83e4a24a8dbf008e5fe4e02d7efb692c1bec582"
    )
    assert config["sources"]["model_revision"] == (
        "36b16b55d21f773f080f5980b195aa2ece5b9358"
    )
    assert config["sources"]["checkpoint"] == {
        "path": "MMO_MSO_with_plan_fully_decoupled.pth",
        "bytes": 941598147,
        "sha256": "8e1816dc290df28f1e89d0b88eddb451b9ab20fc85a8ef6449d9226ceeacd1ce",
    }
    assert config["input"]["frames"] == [42, 47, 52]
    assert config["input"]["metadata_indices"] == [8, 9, 10]
    assert config["input"]["drivestudio_camera_ids"] == [0, 2, 1, 5, 3, 4]
    assert config["output_contract"]["raw_logits_shape"] == [1, 3, 1, 40000, 16, 17]
    assert config["output_contract"]["grid_shape"] == [200, 200, 16]
    assert config["output_contract"]["empty_class"] == 0
    assert "no_occupancy_ground_truth_o_method_o_eval_or_confirmation_read" in config[
        "claim_boundary"
    ]


def test_worker_does_not_enter_future_or_planning_path() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    assert "model.future_pred(" not in source
    assert "model.forward_test(" not in source
    assert '"future_decoder_started": False' in source
    assert '"planning_head_started": False' in source
    assert '"occupancy_ground_truth_read": False' in source


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


def test_can_bus_uses_first_frame_zero_then_absolute_pose_deltas(tmp_path: Path) -> None:
    (tmp_path / "lidar_pose").mkdir()
    (tmp_path / "extrinsics").mkdir()
    (tmp_path / "intrinsics").mkdir()
    for frame in (42, 47, 52):
        np.savetxt(tmp_path / f"lidar_pose/{frame:03d}.txt", np.eye(4))
        np.savetxt(tmp_path / f"extrinsics/{frame:03d}_0.txt", np.eye(4))
    np.savetxt(tmp_path / "intrinsics/0.txt", np.asarray([1.0, 1.0, 0.0, 0.0]))
    infos = []
    for index, (position, yaw) in enumerate(((1.0, 10.0), (4.0, 13.0), (9.0, 15.0))):
        can_bus = np.zeros(18, dtype=np.float64)
        can_bus[0] = position
        can_bus[-1] = yaw
        infos.append(
            {
                "can_bus_absolute": can_bus,
                "scene_token": "scene-token",
                "timestamp": index,
            }
        )
    metas = build_frame_metas(
        tmp_path, "scene-0048", [42, 47, 52], [0], infos, (928, 1600, 3)
    )
    np.testing.assert_allclose(metas[0]["can_bus"][[0, -1]], [0.0, 0.0])
    np.testing.assert_allclose(metas[1]["can_bus"][[0, -1]], [3.0, 3.0])
    np.testing.assert_allclose(metas[2]["can_bus"][[0, -1]], [5.0, 2.0])
    assert [meta["prev_bev_exists"] for meta in metas] == [False, True, True]
    assert quaternion_yaw_degrees([1.0, 0.0, 0.0, 0.0]) == 0.0
