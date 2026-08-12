from __future__ import annotations

import numpy as np

from scripts.run_worldsim_v4_m2_scene import _cap_delta, _depth_mae, preflight


def test_depth_mae_is_fail_closed_without_valid_support() -> None:
    depth = np.asarray([[1.0, np.nan]], dtype=np.float32)
    assert _depth_mae(depth, depth, np.asarray([[False, True]])) == float("inf")


def test_cap_delta_is_deterministic_by_feather_then_id() -> None:
    delta = {
        "means": np.arange(12, dtype=np.float32).reshape(4, 3),
        "source_gaussian_ids": np.asarray([9, 7, 8, 6], dtype=np.int64),
        "feather_weight": np.asarray([0.4, 0.9, 0.9, 0.1], dtype=np.float32),
    }
    capped = _cap_delta(delta, 2)
    assert capped["source_gaussian_ids"].tolist() == [7, 8]


def test_preflight_abstain_retains_no_test_read() -> None:
    config = {
        "schema_version": "worldsim_v4_m2_scene_v1",
        "task_id": "WS-V4-M2-REPAIR-ROUTER-01",
        "partition": "development",
        "scene": "scene-x",
        "status": "abstain",
        "inputs": {},
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    assert preflight(config, phase="smoke") == {}
