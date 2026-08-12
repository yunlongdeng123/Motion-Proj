from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_worldsim_v4_m1_freeze import M1FreezeAuditError, audit


def _json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> dict:
    thresholds = {"v33_o1": 0.5, "raw": 0.5, "temperature": 0.5, "beta": 0.15}
    scene_records = [
        {"scene": "scene-a", "status": "abstain", "reason": "ABSTAIN_LEGACY_SPLIT_LEAK"},
        {"scene": "scene-b", "status": "abstain", "reason": "ABSTAIN_NO_ACTOR"},
        {"scene": "scene-c", "status": "done"},
        {"scene": "scene-d", "status": "done"},
    ]
    accounting = {
        "required_scene_count": 4,
        "evaluable_scene_count": 2,
        "abstain_scene_count": 2,
        "coverage_denominator": "all_required_scenes",
        "quality_metric_denominator": "evaluable_scenes_only",
    }
    status = {
        "status": "done",
        "phase": "six_scene_development",
        "development_content_read": True,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    summary = {
        "scenes": ["scene-a", "scene-b", "scene-c", "scene-d"],
        "evaluable_scenes": ["scene-c", "scene-d"],
        "abstain_scenes": ["scene-a", "scene-b"],
        "cohort_accounting": accounting,
        "scene_records": scene_records,
        "selected_evidence_arm": "raw__risk_100",
        "selected_calibration": "raw",
        "selected_mask_threshold": 0.5,
        "base_rgb_render_checks": [{"exact": True}],
        "checkpoint_checks": [{"exact": True}],
    }
    metrics = {
        "gate_preview": {
            "status": "pass",
            "checks": {
                "boundary_f1": True,
                "false_negative_semantic_mass": True,
                "calibration": True,
                "base_rgb_exact": True,
            },
        }
    }
    payloads = {
        "status.json": status,
        "summary.json": summary,
        "metrics.json": metrics,
        "calibration.json": {"selected": "raw"},
        "manifest.json": {"status": "done"},
    }
    files = {
        name: {"sha256": _json(tmp_path / name, payload)}
        for name, payload in payloads.items()
    }
    return {
        "schema_version": "worldsim_v4_m1_evidence_v1",
        "status": "development_candidate",
        "protocol": {
            "development_scenes": ["scene-a", "scene-b", "scene-c", "scene-d"],
            "abstain_no_actor_scenes": ["scene-b"],
            "development_abstentions": {
                "scene-a": {"reason": "ABSTAIN_LEGACY_SPLIT_LEAK"}
            },
        },
        "development_result": {
            "status": "done",
            "gate_status": "pass",
            "run": str(tmp_path),
            "project_git_head": "abc",
            "scenes": ["scene-a", "scene-b", "scene-c", "scene-d"],
            "evaluable_scenes": ["scene-c", "scene-d"],
            "abstain_scenes": ["scene-a", "scene-b"],
            **{key: accounting[key] for key in ("required_scene_count", "evaluable_scene_count", "abstain_scene_count")},
            "files": files,
        },
        "frozen_selection": {
            "evidence_arm": "raw__risk_100",
            "calibration": "raw",
            "mask_threshold": 0.5,
            "temporal_retention": 0.75,
        },
        "evidence": {"temporal": {"retention": 0.75}},
    }


def test_freeze_audit_accepts_content_addressed_development_result(tmp_path: Path) -> None:
    summary = audit(_config(tmp_path))
    assert summary["status"] == "done"
    assert summary["selected"]["evidence_arm"] == "raw__risk_100"


def test_freeze_audit_rejects_abstention_reason_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary_path = tmp_path / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["scene_records"][0]["reason"] = "OTHER"
    config["development_result"]["files"]["summary.json"]["sha256"] = _json(
        summary_path, payload
    )
    with pytest.raises(M1FreezeAuditError, match="abstention reason"):
        audit(config)
