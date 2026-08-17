from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0l_train_only_quality_identity_alignment import _assignment_metrics, _validate_config
from scripts.audit_worldsim_v51_f0l_train_only_quality_identity_alignment import audit


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0l_train_only_quality_identity_alignment_v1.yaml"


def test_f0l_config_binds_frozen_thresholds_and_exact_read_denominator():
    config, manifest = _validate_config(CONFIG)
    assert manifest["view_count"] == 45
    assert config["locks"]["candidate_mask_pixel_reads"] == 45
    assert config["locks"]["dynamic_mask_pixel_reads"] == 45
    assert config["locks"]["threshold_search"] is False
    assert config["evaluation"]["pass_requires"] == "all_three_scenes_pass_all_thresholds"


def test_assignment_metrics_penalize_short_id_collision():
    rows = [
        {"instance_token": "a", "support_pixels": 100, "positive_pixels": 90, "label_counts": {1: 90}},
        {"instance_token": "a", "support_pixels": 100, "positive_pixels": 90, "label_counts": {1: 90}},
        {"instance_token": "b", "support_pixels": 100, "positive_pixels": 90, "label_counts": {1: 90}},
        {"instance_token": "b", "support_pixels": 100, "positive_pixels": 90, "label_counts": {1: 90}},
    ]
    thresholds = {
        "minimum_eligible_tracks": 1, "minimum_eligible_actor_views": 2, "minimum_foreground_coverage": 0.70,
        "minimum_one_to_one_assignment_recall": 0.35, "minimum_assignment_efficiency": 0.75,
        "minimum_persistent_track_fraction": 0.50,
    }
    result = _assignment_metrics(rows, thresholds)
    assert result["metrics"]["foreground_coverage"] == 0.9
    assert result["metrics"]["independent_best_identity_recall"] == 0.9
    assert result["metrics"]["one_to_one_assignment_recall"] == 0.45
    assert result["metrics"]["assignment_efficiency"] == 0.5
    assert result["checks"]["assignment_efficiency"] is False


def test_f0l_r043_independent_audit_confirms_algorithm_rejection():
    result = audit(
        CONFIG,
        Path(
            "/root/autodl-tmp/runs/worldsim_v51/"
            "WS-V51-M1-F-IDENTITY-EMBEDDING-01/"
            "20260818T200000Z__m1-stage-f-f0l-quality-alignment-s20260814-r043"
        ),
    )
    assert result["status"] == "pass"
    assert result["audited_outcome"] == "rejected"
    assert result["scene_pass_vector"] == [False, True, False]
    assert result["threshold_search"] is False
    assert result["identity_training_authorized"] is False
