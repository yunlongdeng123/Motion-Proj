from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_worldsim_v4_m1_validation import M1ValidationAuditError, audit


def _write(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> dict:
    scenes = [f"scene-{index}" for index in range(6)]
    evaluable = [scenes[index] for index in (0, 2, 5)]
    abstain = [scenes[index] for index in (1, 3, 4)]
    selection = {
        "evidence_arm": "raw__risk_100",
        "calibration": "raw",
        "mask_threshold": 0.5,
        "temporal_retention": 0.75,
    }
    gate = {
        "status": "reject",
        "required_directional_support_scene_count": 4,
        "directional_support_scene_count": 0,
        "arm_search_performed": False,
        "calibration_fit_performed": False,
        "threshold_search_performed": False,
    }
    accounting = {
        "required_scene_count": 6,
        "evaluable_scene_count": 3,
        "abstain_scene_count": 3,
        "coverage_denominator": "all_required_scenes",
        "quality_metric_denominator": "evaluable_scenes_only",
    }
    summary = {
        "scenes": scenes,
        "evaluable_scenes": evaluable,
        "abstain_scenes": abstain,
        "cohort_accounting": accounting,
        "confirmation_gate": gate,
        "frozen_selection": selection,
        "base_rgb_render_checks": [{"exact": True}],
        "checkpoint_checks": [{"exact": True}],
    }
    summary_sha = _write(tmp_path / "summary.json", summary)
    manifest = {"status": "done", "task_id": "WS-V4-M1-EVIDENCE-FIELD-01"}
    manifest_sha = _write(tmp_path / "manifest.json", manifest)
    status = {
        "status": "done",
        "confirmation_status": "reject",
        "phase": "six_scene_validation_confirmation",
        "scenes": scenes,
        "summary_sha256": summary_sha,
        "manifest_sha256": manifest_sha,
        "validation_content_read": True,
        "validation_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    metrics = {
        "confirmation_gate": gate,
        "arm_search_performed": False,
        "calibration_fit_performed": False,
        "threshold_search_performed": False,
    }
    payloads = {"status.json": status, "metrics.json": metrics}
    hashes = {name: _write(tmp_path / name, payload) for name, payload in payloads.items()}
    hashes.update({"summary.json": summary_sha, "manifest.json": manifest_sha})
    return {
        "schema_version": "worldsim_v4_m1_evidence_v1",
        "status": "rejected",
        "protocol": {"validation_scenes": scenes},
        "frozen_selection": selection,
        "validation_result": {
            "status": "done",
            "gate_status": "reject",
            "run": str(tmp_path),
            "project_git_head": "abc",
            "evaluable_scenes": evaluable,
            "abstain_scenes": abstain,
            "files": {name: {"sha256": digest} for name, digest in hashes.items()},
        },
    }


def test_validation_audit_accepts_frozen_rejection(tmp_path: Path) -> None:
    result = audit(_config(tmp_path))
    assert result["task_status"] == "rejected"
    assert result["m2_fallback_authorized"] is True
    assert result["m1_feature_expansion_authorized"] is False


def test_validation_audit_rejects_any_validation_search(tmp_path: Path) -> None:
    config = _config(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["threshold_search_performed"] = True
    config["validation_result"]["files"]["metrics.json"]["sha256"] = _write(
        metrics_path, metrics
    )
    with pytest.raises(M1ValidationAuditError, match="optimization was performed"):
        audit(config)


def test_validation_audit_rejects_immutability_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_checks"][0]["exact"] = False
    new_sha = _write(summary_path, summary)
    config["validation_result"]["files"]["summary.json"]["sha256"] = new_sha
    status_path = tmp_path / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["summary_sha256"] = new_sha
    config["validation_result"]["files"]["status.json"]["sha256"] = _write(
        status_path, status
    )
    with pytest.raises(M1ValidationAuditError, match="checkpoint immutability"):
        audit(config)
