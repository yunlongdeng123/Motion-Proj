from __future__ import annotations

from motion_proj.preference.calibration import scene_hash_split


def test_calibration_and_holdout_are_scene_disjoint_and_deterministic() -> None:
    cases = [
        {"case_id": f"case-{index:02d}", "scene_id": f"scene-{index:02d}"}
        for index in range(22)
    ]
    first = scene_hash_split(cases, calibration_count=12, salt="drivepo-v4")
    second = scene_hash_split(list(reversed(cases)), calibration_count=12, salt="drivepo-v4")

    assert first == second
    assert len(first["calibration_case_ids"]) == 12
    assert len(first["holdout_case_ids"]) == 10
    assert set(first["calibration_scene_ids"]).isdisjoint(first["holdout_scene_ids"])
