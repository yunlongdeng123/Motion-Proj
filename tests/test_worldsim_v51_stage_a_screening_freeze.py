from __future__ import annotations

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v5_m1_sam_diagnostic import (
    load_config as load_sam_config,
    validate_frame_contract,
)


def test_screening_candidates_scenes_and_gate_are_frozen() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/stage_a_screening_freeze_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert list(config["retained_candidates"]) == ["A1", "A2"]
    assert config["screening_scenes"] == ["scene-0998", "scene-0359"]
    assert config["screening_gate"][
        "boundary_f1_clearly_positive_scene_count_minimum"
    ] == 1
    assert config["screening_gate"][
        "boundary_f1_clear_delta_minimum_inclusive"
    ] == 0.001
    assert config["selection_policy"]["maximum_survivors_after_s"] == 1
    assert config["locks"]["validation_quality_read"] is False


def test_sam_screening_configs_inherit_frozen_split() -> None:
    for scene, index in (("scene0998", 756), ("scene0359", 276)):
        config = load_sam_config(
            ROOT / f"configs/worldsim_v51/m1_sam_screening_{scene}_v1.yaml"
        )
        evidence, evaluation = validate_frame_contract(config)
        assert config["scene"]["index"] == index
        assert evidence == [0, 40, 80, 120, 160]
        assert evaluation == [2, 42, 82, 122, 162]
        assert config["split"]["heldout_remainder"] == 4
        assert config["sam2"]["quality_gate"]["minimum_prompt_bbox_iou"] == 0.02
