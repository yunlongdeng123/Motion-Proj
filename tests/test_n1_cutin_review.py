import json

import pytest
import yaml

from motion_proj.resim.canonical_hash import canonical_sha256
from motion_proj.runtime.fingerprint import file_fingerprint
from scripts.validate_n1_cutin_review import validate, wilson_lower_bound


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path):
    run_dir = tmp_path / "run"
    audit = run_dir / "audit"
    evidence = audit / "evidence"
    evidence.mkdir(parents=True)
    (run_dir / "AWAITING_HUMAN_REVIEW").write_text("await\n", encoding="utf-8")
    config = {
        "human_audit": {
            "review_schema": "receiver_cutin",
            "min_reviewed_items": 1,
            "min_true_positive_count": 1,
            "min_true_positive_scenes": 1,
            "min_precision": 1.0,
            "min_wilson_95_lower_bound": 0.0,
            "max_uncertain_fraction": 0.0,
        }
    }
    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "summary.json",
        {
            "task_id": "N1-EVENT-CUTIN-01",
            "formal": True,
            "audit_ready": True,
            "machine_gate_passed": True,
            "n2_authorized": False,
        },
    )
    _write_json(
        run_dir / "event_pool.json",
        {"event_pool_sha256": "a" * 64},
    )
    _write_json(evidence / "K4-001.json", {"scene_id": "scene-1"})
    template = {
        "audit_id": "K4-001",
        "evidence_sha256": "b" * 64,
        "panel_sha256": "c" * 64,
        "subject_maneuver_verdict": "",
        "receiver_corridor_verdict": "",
        "receiver_relation_verdict": "",
        "temporal_persistence_verdict": "",
        "overall_verdict": "",
        "failure_codes": [],
        "reviewer": "",
        "notes": "",
    }
    _write_json(audit / "review_template.jsonl", template)
    hashes = {
        "evidence/K4-001.json": file_fingerprint(
            str(evidence / "K4-001.json")
        ),
        "review_template.jsonl": file_fingerprint(
            str(audit / "review_template.jsonl")
        ),
    }
    _write_json(
        audit / "audit_manifest.json",
        {
            "schema_version": "n1-receiver-cutin-human-audit-pack-v1",
            "event_pool_sha256": "a" * 64,
            "immutable_file_hashes": hashes,
            "immutable_artifact_set_sha256": canonical_sha256(hashes),
        },
    )
    reviewed = {
        **template,
        "subject_maneuver_verdict": "VALID",
        "receiver_corridor_verdict": "VALID",
        "receiver_relation_verdict": "VALID",
        "temporal_persistence_verdict": "VALID",
        "overall_verdict": "TRUE_POSITIVE",
        "reviewer": "human",
        "notes": "轨迹、接收车身份和时序均已人工确认。",
    }
    review_file = audit / "review_working.jsonl"
    _write_json(review_file, reviewed)
    return run_dir, review_file


def test_wilson_lower_bound_known_values():
    assert wilson_lower_bound(0, 0) is None
    assert wilson_lower_bound(20, 20) == pytest.approx(0.8388748419)
    assert wilson_lower_bound(16, 20) == pytest.approx(0.5839825677)


def test_validate_receiver_cutin_review(tmp_path):
    run_dir, review_file = _fixture(tmp_path)
    result = validate(run_dir, review_file)
    assert result["all_human_gates_passed"]
    assert result["overall_counts"] == {"TRUE_POSITIVE": 1}
    assert result["n2_authorized"] is False


def test_false_positive_requires_invalid_component(tmp_path):
    run_dir, review_file = _fixture(tmp_path)
    row = json.loads(review_file.read_text(encoding="utf-8"))
    row["overall_verdict"] = "FALSE_POSITIVE"
    row["failure_codes"] = ["ROUTE_CONTINUATION"]
    _write_json(review_file, row)
    with pytest.raises(ValueError, match="必须含 INVALID"):
        validate(run_dir, review_file)
