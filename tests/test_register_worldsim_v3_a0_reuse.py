from copy import deepcopy

from scripts.register_worldsim_v3_a0_reuse import (
    PRESETS,
    canonical_sha256,
)


def test_reuse_presets_freeze_three_scene_cohorts() -> None:
    assert set(PRESETS) == {"scene-0230", "scene-0242"}
    assert PRESETS["scene-0230"]["expected_high_gaussians"] == 4747
    assert PRESETS["scene-0230"]["expected_boundary_gaussians"] == 1914
    assert PRESETS["scene-0242"]["expected_high_gaussians"] == 6939
    assert PRESETS["scene-0242"]["expected_boundary_gaussians"] is None


def test_canonical_hash_is_order_independent_and_value_sensitive() -> None:
    left = {"model": {"Affine": True}, "data": {"stride": 10}}
    right = {"data": {"stride": 10}, "model": {"Affine": True}}

    assert canonical_sha256(left) == canonical_sha256(right)
    changed = deepcopy(right)
    changed["data"]["stride"] = 9
    assert canonical_sha256(left) != canonical_sha256(changed)
