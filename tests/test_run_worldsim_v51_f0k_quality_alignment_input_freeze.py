from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0k_quality_alignment_input_freeze import _validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0k_quality_alignment_input_freeze_v1.yaml"


def test_f0k_preregisters_weak_reference_semantics_and_thresholds_before_pixel_read():
    config = _validate_config(CONFIG)
    gate = config["quality_gate_preregistration"]
    assert "not_instance_ground_truth" in gate["reference_semantics"]
    assert gate["eligible_actor_view_minimum_support_pixels"] == 64
    assert gate["eligible_track_minimum_views"] == 2
    assert gate["per_scene_thresholds"] == {
        "minimum_eligible_tracks": 1,
        "minimum_eligible_actor_views": 2,
        "minimum_foreground_coverage": 0.70,
        "minimum_one_to_one_assignment_recall": 0.35,
        "minimum_assignment_efficiency": 0.75,
        "minimum_persistent_track_fraction": 0.50,
    }
    assert config["locks"]["candidate_mask_pixels_read"] is False
    assert config["locks"]["dynamic_mask_pixels_read"] is False
    assert gate["pass_requires"] == "all_three_scenes_pass_all_thresholds"
