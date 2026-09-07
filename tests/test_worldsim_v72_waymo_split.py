from __future__ import annotations

from scripts.freeze_worldsim_v72_waymo_roles import freeze_roles


def test_waymo_split_is_disjoint_and_quality_blind() -> None:
    payload = {"schema_version": "worldsim_v72.data_roles.v1", "datasets": {}}
    train = [f"segment-{index}" for index in range(12)]
    validation = [f"segment-val-{index}" for index in range(4)]
    frozen = freeze_roles(
        payload,
        train_contexts=train,
        validation_contexts=validation,
        build_count=7,
        development_count=3,
        train_source={},
        validation_source={},
    )
    dataset = frozen["datasets"]["waymo_perception_v2"]
    roles = dataset["group_roles"]
    assert [len(roles[name]) for name in ("train", "development", "route_select", "source_test")] == [7, 3, 2, 4]
    all_ids = roles["train"] + roles["development"] + roles["route_select"] + roles["source_test"]
    assert len(all_ids) == len(set(all_ids))
    assert dataset["selection"]["quality_used_for_selection"] is False
