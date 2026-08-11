from scripts.finalize_dr_v2_m5 import merge_sequence


def test_merge_sequence_uses_frozen_priority() -> None:
    scene = {
        "scene": "scene-0230",
        "role": "high-support",
        "instance_token": "token",
        "edit": "delete",
        "status": "done",
        "failure_codes_pre_perception": ["TEMPORAL_FLICKER"],
        "metrics": {"x": 1},
    }
    perception = {
        "scene": "scene-0230",
        "role": "high-support",
        "edit": "delete",
        "failure_codes": ["NON_TARGET_PERCEPTION_DRIFT"],
        "metrics": {"y": 2},
    }
    priority = ["TEMPORAL_FLICKER", "NON_TARGET_PERCEPTION_DRIFT"]
    merged = merge_sequence(scene, perception, priority)
    assert merged["primary_failure"] == "TEMPORAL_FLICKER"
    assert merged["failure_codes"] == priority
