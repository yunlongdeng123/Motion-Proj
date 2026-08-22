from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/worldsim_v61/me3_gaussianworld_predicted_v1.yaml"


def test_me3_freezes_stream_mapping_unknown_and_parallelism() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["task_id"] == "WS-V61-ME3-PREDICTED-OCC-01"
    assert config["hypothesis_id"] == "WS-V61-H-ME3-GW-002"
    assert config["streaming"]["frames"] == list(range(2, 58, 5))
    assert config["streaming"]["target_frames"] == [52, 57]
    assert config["streaming"]["camera_ids"] == [0, 2, 1, 5, 3, 4]
    assert config["target_grid"]["class_mapping"] == {
        "0": "UNKNOWN",
        "1-16": "OCCUPIED",
        "17": "FREE",
        "outside_source_extent": "UNKNOWN",
    }
    assert config["method_gate"]["unknown_policy"] == "blocks_ray_and_abstains"
    assert config["method_gate"]["predicted_free_is_not_observed_truth"] is True
    assert config["resources"]["parallel_scene_workers"] == 2
    assert config["primary_gate"]["minimum_accepted_cases"] == 8
    assert config["primary_gate"]["maximum_false_safe_count"] == 0
    assert config["primary_gate"]["minimum_oracle_yield_fraction"] == 0.8


def test_me3_claim_boundary_keeps_prediction_separate_from_truth() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    claims = set(config["claim_boundary"])
    assert "GaussianWorld_is_predicted_evidence_not_truth" in claims
    assert "predicted_UNKNOWN_blocks_rays_and_predicted_FREE_is_not_observed_FREE" in claims
    assert "native_boxes_bind_identity_only_and_never_create_occupancy" in claims
    assert "method_decisions_freeze_before_O_eval_is_loaded" in claims
