import json

import pytest
import yaml

from scripts.build_n1_cutin_audit import build_audit_pack
from scripts.validate_n1_cutin_review import validate


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _record(event_id, scene_id, status="PASS"):
    primary_reason = None if status == "PASS" else "INSUFFICIENT_RAW_SUPPORT"
    return {
        "event_id": event_id,
        "event_record_sha256": f"{event_id}-hash",
        "scene_id": scene_id,
        "subject_instance_token": "subject-token",
        "receiver_instance_token": "receiver-token",
        "strict": {
            "status": status,
            "primary_reason": primary_reason,
            "machine_positive": status == "PASS",
            "subject": {
                "per_frame": [
                    {
                        "frame": 0,
                        "world_xy": [0.0, 3.0],
                        "yaw_rad": 0.0,
                        "dimensions_lwh": [4.0, 1.8, 1.5],
                        "target_d_m": 3.0,
                        "target_heading_error_deg": 1.0,
                        "box_inside_target_band": False,
                    },
                    {
                        "frame": 5,
                        "world_xy": [4.0, 0.2],
                        "yaw_rad": 0.0,
                        "dimensions_lwh": [4.0, 1.8, 1.5],
                        "target_d_m": 0.2,
                        "target_heading_error_deg": 1.0,
                        "box_inside_target_band": True,
                    },
                ]
            },
            "receiver": {
                "gap_m_by_frame": [8.0, 8.2],
                "longitudinal_speed_mps_by_frame": [5.0, 5.0],
                "actor_id_by_frame": [2, 2],
                "nearest_rear_rank_by_frame": [1, 1],
            },
            "receiver_per_frame": [
                {
                    "frame": 0,
                    "nearest_rear": {
                        "actor_id": 2,
                        "nearest_rear_rank": 1,
                        "bumper_gap_m": 8.0,
                        "world_xy": [-8.0, 0.0],
                        "yaw_rad": 0.0,
                        "dimensions_lwh": [4.0, 1.8, 1.5],
                    },
                },
                {
                    "frame": 5,
                    "nearest_rear": {
                        "actor_id": 2,
                        "nearest_rear_rank": 1,
                        "bumper_gap_m": 8.2,
                        "world_xy": [-4.0, 0.0],
                        "yaw_rad": 0.0,
                        "dimensions_lwh": [4.0, 1.8, 1.5],
                    },
                },
            ],
            "corridor": {"centerline_xy": [[-10.0, 0.0], [10.0, 0.0]]},
        },
    }


def _fixture(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "AWAITING_HUMAN_REVIEW").write_text("await\n", encoding="utf-8")
    config = {
        "schema_version": "receiver-centric-cutin-final-v1",
        "audit": {
            "primary_target_count": 30,
            "primary_max_count": 40,
            "abstain_diagnostic_max_count": 10,
        },
        "human_gates": {
            "pass_min_reviewed_determinate": 1,
            "pass_min_true_positive": 1,
            "pass_min_positive_scenes": 1,
            "pass_min_precision": 1.0,
            "pass_min_wilson_lower_bound": 0.0,
            "pass_max_uncertain_fraction": 0.0,
            "sparse_min_true_positive": 1,
            "sparse_min_positive_scenes": 1,
            "sparse_min_precision": 1.0,
            "sparse_max_uncertain_fraction": 0.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_json(run_dir / "strict_event_pool.json", {"strict_event_pool_sha256": "p" * 64})
    (run_dir / "strict_candidates.jsonl").write_text(
        json.dumps(_record("event-pass", "scene-a")) + "\n", encoding="utf-8"
    )
    (run_dir / "diagnostic_abstain_candidates.jsonl").write_text(
        json.dumps(_record("event-abstain", "scene-b", status="ABSTAIN")) + "\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "summary.json",
        {
            "task_id": "N1-EVENT-CUTIN-FINAL-01",
            "strict_event_pool_sha256": "p" * 64,
            "n2_authorized": False,
        },
    )
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    build_audit_pack(run_dir, config_path)
    return run_dir


def test_v2_audit_blind_page_and_validator_preserve_order(tmp_path):
    run_dir = _fixture(tmp_path)
    blind = (run_dir / "audit" / "index.html").read_text(encoding="utf-8")
    assert "machine status" not in blind
    assert "primary_pass" not in blind
    assert "diagnostic_abstain" not in blind
    template = [
        json.loads(line)
        for line in (run_dir / "audit" / "review_template.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reviewed = []
    for row in template:
        if row["review_tier"] == "primary_pass":
            reviewed.append(
                {
                    **row,
                    "subject_maneuver_verdict": "VALID",
                    "receiver_corridor_verdict": "VALID",
                    "receiver_relation_verdict": "VALID",
                    "temporal_persistence_verdict": "VALID",
                    "overall_verdict": "TRUE_POSITIVE",
                    "reviewer": "human",
                    "notes": "原始时序和角色均可辨认。",
                }
            )
        else:
            reviewed.append(
                {
                    **row,
                    "subject_maneuver_verdict": "UNCERTAIN",
                    "receiver_corridor_verdict": "VALID",
                    "receiver_relation_verdict": "VALID",
                    "temporal_persistence_verdict": "VALID",
                    "overall_verdict": "UNCERTAIN",
                    "reviewer": "human",
                    "notes": "仅作 diagnostic，不进入 primary precision。",
                }
            )
    review_file = run_dir / "audit" / "review_working.jsonl"
    review_file.write_text("".join(json.dumps(row) + "\n" for row in reviewed), encoding="utf-8")
    result = validate(run_dir, review_file)
    assert result["all_human_gates_passed"]
    assert result["primary_overall_counts"] == {"TRUE_POSITIVE": 1}
    review_file.write_text("".join(json.dumps(row) + "\n" for row in reversed(reviewed)), encoding="utf-8")
    with pytest.raises(ValueError, match="顺序"):
        validate(run_dir, review_file)
