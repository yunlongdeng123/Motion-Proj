"""唯一 ME-3 IR-WM recovery 的固定合同。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from motion_proj.worldsim_v61.me3_predicted import resample_irwm_classes
from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN, VoxelGridSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/worldsim_v61/me3r_irwm_predicted_v1.yaml"
WORKER_PATH = REPO_ROOT / "scripts/run_worldsim_v61_me3_irwm_worker.py"


def test_me3r_freezes_capability_windows_and_primary_gate() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["task_id"] == "WS-V61-ME3R-IRWM-PREDICTED-OCC-01"
    assert config["hypothesis_id"] == "WS-V61-H-ME3-IRWM-001"
    assert config["streaming"]["history_windows"] == {
        "52": {"frames": [42, 47, 52], "metadata_indices": [8, 9, 10]},
        "57": {"frames": [47, 52, 57], "metadata_indices": [9, 10, 11]},
    }
    assert config["resources"]["parallel_scene_workers"] == 2
    assert config["primary_gate"] == {
        "arm": "P1-IRWM-PREDICTED",
        "minimum_accepted_cases": 8,
        "maximum_false_safe_count": 0,
        "must_exceed_accept_count": 3,
        "minimum_oracle_yield_fraction": 0.8,
    }
    assert "this_is_the_only_ME3_recovery_after_GaussianWorld_failure" in config[
        "claim_boundary"
    ]


def test_irwm_resampling_maps_zero_to_free_and_outside_to_unknown() -> None:
    source = np.zeros((200, 200, 16), dtype=np.uint8)
    source[100, 100, 4] = 3
    spec = VoxelGridSpec(
        frame="target_lidar", origin_m=(-0.256, -0.256, -3.0),
        voxel_size_m=0.512, shape=(1, 1, 13),
    )
    result = resample_irwm_classes(source, spec)
    assert result["semantics"].shape == (1, 1, 13)
    assert result["semantics"][0, 0, 0] == OCCUPIED
    assert result["semantics"][0, 0, 1] == FREE
    assert result["semantics"][0, 0, -1] == UNKNOWN


def test_worker_never_reads_truth_or_enters_future_planning() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    assert "model.future_pred(" not in source
    assert "model.forward_test(" not in source
    assert '"occupancy_ground_truth_read": False' in source
    assert '"o_method_or_o_eval_read": False' in source
    assert '"future_decoder_started": False' in source
    assert '"planning_head_started": False' in source
