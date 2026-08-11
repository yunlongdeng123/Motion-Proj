import pytest

from scripts.run_dr_v2_m5_stress_scene import resolve_registry_actor


def test_resolve_registry_actor_preserves_boundary_abstain_coverage() -> None:
    actor = resolve_registry_actor({"actors": []}, "boundary-support", "frozen")
    assert actor == {
        "instance_token": "frozen",
        "availability": "unavailable_not_in_checkpoint_registry",
        "checkpoint_tensor_slice": {"gaussian_count": 0},
        "class_name": None,
        "rigid_model_index": None,
    }


def test_resolve_registry_actor_requires_high_support_and_unique_mapping() -> None:
    with pytest.raises(RuntimeError, match="not unique"):
        resolve_registry_actor({"actors": []}, "high-support", "required")
    duplicate = {
        "actors": [
            {"instance_token": "duplicate"},
            {"instance_token": "duplicate"},
        ]
    }
    with pytest.raises(RuntimeError, match="not unique"):
        resolve_registry_actor(duplicate, "boundary-support", "duplicate")
