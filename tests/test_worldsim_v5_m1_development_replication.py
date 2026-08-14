from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_worldsim_v5_m1_sam_diagnostic import (
    load_config,
    validate_frame_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_replication_cohort_is_first_three_frozen_development_scenes() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v5/m1_development_replication_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["selection"]["policy"] == "first_three_scenes_without_quality_read"
    assert config["selection"]["scenes"] == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    assert config["selection"]["scene_indices"] == [382, 827, 296]
    assert config["selection"]["selection_used_quality"] is False
    assert config["replication_gate"]["denominator"] == "6_scene_by_unary_cells"
    assert config["replication_gate"]["automatic_validation_unlock"] is False
    assert config["replication_gate"]["formal_arm_selection_allowed"] is False


def test_new_sam_configs_preserve_scene0471_method_contract() -> None:
    paths = [
        ROOT / "configs/worldsim_v5/m1_sam_diagnostic_scene0471_v1.yaml",
        ROOT / "configs/worldsim_v5/m1_sam_diagnostic_scene1087_v1.yaml",
        ROOT / "configs/worldsim_v5/m1_sam_diagnostic_scene0379_v1.yaml",
    ]
    configs = [load_config(path) for path in paths]
    assert [config["scene"]["name"] for config in configs] == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    for config in configs:
        evidence, evaluation = validate_frame_contract(config)
        assert evidence == [0, 40, 80, 120, 160]
        assert evaluation == [2, 42, 82, 122, 162]
        assert config["prompts"] == configs[0]["prompts"]
        assert config["sam2"] == configs[0]["sam2"]
        assert config["outputs"] == configs[0]["outputs"]


def test_replication_inputs_bind_distinct_formal_checkpoints() -> None:
    expected = {
        "scene-1087": "84c34b837e8083a84a77de0662c6c702ce978a2e8e11a4038d487e1f6c9754be",
        "scene-0379": "d77fa13f5ebe1d469222c3df78eb316d836178dca608d00c17f0af3a56d3042d",
    }
    for scene, checkpoint_sha in expected.items():
        config = yaml.safe_load(
            (
                ROOT
                / f"configs/worldsim_v5/m1_sam_diagnostic_{scene.replace('-', '')}_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        assert config["inputs"]["formal_checkpoint"]["sha256"] == checkpoint_sha
        assert config["scene"]["name"] == scene
