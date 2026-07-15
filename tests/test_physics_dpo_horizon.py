from __future__ import annotations

from typing import Any

from motion_proj.data.physics_dpo_schema import build_scene_split
from motion_proj.diagnostics.physics_dpo_horizon import (
    compare_score_repeatability,
    decide_horizon,
    select_profile_conditions,
)
from motion_proj.runtime.fingerprint import sha256_json


def _source(split: str, prefix: str, scene_count: int) -> dict[str, Any]:
    scenes = []
    clips = []
    for index in range(scene_count):
        token = f"{prefix}-token-{index:02d}"
        name = f"{prefix}-scene-{index:02d}"
        scenes.append({"scene_name": name, "scene_token": token, "sample_count": 20, "clip_count": 2})
        for start in (0, 8):
            sample = f"{prefix}-sample-{index:02d}-{start:02d}"
            clips.append({
                "clip_id": f"{sample}_CAM_FRONT", "scene_name": name, "scene_token": token,
                "start_index": start, "sample_tokens": [sample],
            })
    core = {"split": split, "camera": "CAM_FRONT", "num_frames": 8, "scenes": scenes, "clips": clips}
    return {**core, "split_fingerprint": sha256_json(core)}


def _complete_profile(*, vram_gb: float, generation_seconds: float) -> dict[str, Any]:
    return {
        "status": "completed",
        "base_guard_exact": True,
        "score_valid": True,
        "score_repeatability_pass": True,
        "generation_peak_vram_bytes_max": int(vram_gb * 1024**3),
        "generation_seconds_mean": generation_seconds,
    }


def test_select_profile_conditions_uses_one_start_zero_clip_per_scene() -> None:
    split = build_scene_split(
        _source("train", "train", 5), _source("val", "val", 4), salt="horizon-test",
        preference_dev_scene_count=2, screen_eval_scene_count=1,
    )

    selected = select_profile_conditions(
        split, partition="preference_dev", condition_count=2, required_start_index=0,
    )

    assert len(selected) == 2
    assert len({row["scene_token"] for row in selected}) == 2
    assert all(row["start_index"] == 0 for row in selected)


def test_score_repeatability_rejects_missing_metric_without_zero_fill() -> None:
    aggregate = {
        "survival_rate": 0.75,
        "camera_compensated_image_plane_velocity_rms_px": 1.0,
        "camera_compensated_image_plane_acceleration_rms_px": 2.0,
        "camera_compensated_image_plane_jerk_rms_px": 3.0,
    }
    passed = compare_score_repeatability({"aggregate": aggregate}, {"aggregate": dict(aggregate)}, maximum_relative_delta=0.02)
    assert passed["valid"] is True
    assert passed["passed"] is True

    invalid = dict(aggregate)
    invalid["camera_compensated_image_plane_jerk_rms_px"] = None
    rejected = compare_score_repeatability({"aggregate": aggregate}, {"aggregate": invalid}, maximum_relative_delta=0.02)
    assert rejected["valid"] is False
    assert rejected["passed"] is False
    assert "jerk" in rejected["reason"]


def test_horizon_decision_only_uses_preregistered_resource_rule() -> None:
    profile8 = _complete_profile(vram_gb=10.0, generation_seconds=10.0)
    profile14 = _complete_profile(vram_gb=20.0, generation_seconds=21.0)
    choose14 = decide_horizon(
        {8: profile8, 14: profile14}, maximum_14_peak_vram_gb=22.0, maximum_14_generation_slowdown=2.2,
    )
    assert choose14["status"] == "done"
    assert choose14["selected_num_frames"] == 14

    profile14["generation_peak_vram_bytes_max"] = int(22.1 * 1024**3)
    choose8 = decide_horizon(
        {8: profile8, 14: profile14}, maximum_14_peak_vram_gb=22.0, maximum_14_generation_slowdown=2.2,
    )
    assert choose8["status"] == "done"
    assert choose8["selected_num_frames"] == 8


def test_horizon_decision_blocks_invalid_eight_and_accepts_14_oom_as_resource_fallback() -> None:
    invalid8 = _complete_profile(vram_gb=10.0, generation_seconds=10.0)
    invalid8["score_repeatability_pass"] = False
    blocked = decide_horizon(
        {8: invalid8, 14: _complete_profile(vram_gb=20.0, generation_seconds=20.0)},
        maximum_14_peak_vram_gb=22.0, maximum_14_generation_slowdown=2.2,
    )
    assert blocked["status"] == "blocked"

    fallback = decide_horizon(
        {8: _complete_profile(vram_gb=10.0, generation_seconds=10.0), 14: {"status": "resource_rejected"}},
        maximum_14_peak_vram_gb=22.0, maximum_14_generation_slowdown=2.2,
    )
    assert fallback["status"] == "done"
    assert fallback["selected_num_frames"] == 8
