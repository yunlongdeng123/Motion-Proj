from pathlib import Path

import yaml

from motion_proj.worldsim_v5.datasets.nuscenes import (
    ROLES,
    SELECTION_FIELDS,
    select_fresh_scene_cohort,
)


ROOT = Path(__file__).resolve().parents[1]


def _candidate(index: int, split: str) -> dict:
    return {
        "scene": f"scene-{index:04d}",
        "scene_token": f"token-{index:04d}",
        "official_split": split,
        "location": f"location-{index % 3}",
        "time_of_day": ("day", "night", "dusk")[index % 3],
        "weather": "rain" if index % 5 == 0 else "dry_or_unspecified",
        "road_geometry": ("intersection", "turn_or_curve", "road_segment")[index % 3],
        "actor_class": ("car", "truck", "bus")[index % 3],
        "speed_regime": ("stationary", "low_speed", "normal_speed")[index % 3],
        "distance_regime": ("near", "mid", "far")[index % 3],
        "occlusion": "heavy" if index % 4 == 0 else "normal",
        "donor_support": ("strong", "medium", "weak")[index % 3],
        "eligible_actor_count": 1 + index % 12,
        "sensor_contract_complete": True,
    }


def test_fresh_selector_is_deterministic_disjoint_and_excludes_v4() -> None:
    candidates = [
        *(_candidate(index, "train") for index in range(1, 31)),
        *(_candidate(index, "val") for index in range(101, 131)),
    ]
    excluded = ["scene-0001", "scene-0002", "scene-0101", "scene-0102"]
    first = select_fresh_scene_cohort(candidates, seed=2216484596, excluded_scenes=excluded)
    second = select_fresh_scene_cohort(candidates, seed=2216484596, excluded_scenes=excluded)
    assert first == second
    assert len(first) == sum(ROLES.values()) == 36
    names = [row["scene"] for row in first]
    assert len(names) == len(set(names))
    assert not set(names) & set(excluded)
    assert {role: sum(row["role"] == role for row in first) for role in ROLES} == ROLES
    assert all(row["official_split"] == "val" for row in first if row["role"] == "test")
    assert all(row["official_split"] == "train" for row in first if row["role"] != "test")


def test_fresh_config_is_post_p0_and_result_blind() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "done"
    assert config["project"]["p0_status"] == "done"
    assert tuple(config["selection"]["allowed_metadata_fields"]) == SELECTION_FIELDS
    assert config["selection"]["selection_uses_model_results"] is False
    assert config["restrictions"]["sensor_blob_expansion_for_selection"] is False
    assert config["restrictions"]["fresh_test_quality_read"] is False
    assert len(config["v4_exclusion"]["scenes"]) == 30
    assert config["freeze"]["selection_status"] == "frozen"
    assert len(config["freeze"]["scene_records"]) == 36
    assert not set(
        scene
        for role_scenes in config["freeze"]["scene_roles"].values()
        for scene in role_scenes
    ) & set(config["v4_exclusion"]["scenes"])
