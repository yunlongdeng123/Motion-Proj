from __future__ import annotations

import numpy as np

from scripts.run_worldsim_v4_m2_scene import (
    _atomic_abstain_metrics,
    _cap_delta,
    _depth_mae,
    _matched_arm_records,
    _snapshot_input_scene_config,
    preflight,
)


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


def test_atomic_abstain_forces_noop_coverage_and_failure() -> None:
    metrics = {
        "hole_cross_view_l1_uint8": 12.5,
        "hole_geometry_mae_m": 0.25,
        "hole_coverage": 0.9,
        "hole_effect_pixels": 90,
    }
    result = _atomic_abstain_metrics(metrics)
    assert result["hole_coverage"] == 0.0
    assert result["hole_effect_pixels"] == 0
    assert result["atomic_noop"] is True
    assert result["operation_success"] is False
    assert result["edit_error"] == 2.0 / 3.0


def test_matched_arm_records_keep_failed_and_router_rows() -> None:
    rows = _matched_arm_records(
        expected_arms=["ABSTAIN", "OBSERVED", "TELEA", "RISK_ROUTER"],
        candidate_records=[{"arm": "OBSERVED", "metrics": {"edit_error": 0.2}}],
        failures=[{"arm": "TELEA", "error": "ABSTAIN_X"}],
        abstain_metrics={"edit_error": 1.0},
    )
    assert [row["status"] for row in rows] == [
        "atomic_noop",
        "executed",
        "abstain",
        "pending_development_selection",
    ]
    assert rows[2]["reasons"] == ["ABSTAIN_X"]


def test_snapshot_input_scene_config_is_content_bound(tmp_path) -> None:
    source = tmp_path / "scene.yaml"
    source.write_text("scene: scene-a\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    binding = _snapshot_input_scene_config(source, run_dir)
    snapshot = run_dir / "source_snapshot/materialized_scene_config.yaml"
    assert snapshot.read_bytes() == source.read_bytes()
    assert binding["snapshot_path"] == str(snapshot)
    assert len(binding["sha256"]) == 64
    assert binding["bytes"] == source.stat().st_size


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
