from __future__ import annotations

import json

import pytest

from motion_proj.diagnostics.physics_preference_reaudit import (
    PreferenceReauditError,
    _assert_no_future_gt,
    _core_candidates,
    _ensure_source_immutable,
    _source_fingerprints,
)


def test_future_gt_and_debug_ego_fail_closed() -> None:
    _assert_no_future_gt({"uses_future_gt": False}, label="safe")
    with pytest.raises(PreferenceReauditError, match="future"):
        _assert_no_future_gt({"uses_future_gt": True}, label="leak")
    with pytest.raises(PreferenceReauditError, match="gt_ego_debug"):
        _assert_no_future_gt({"generated_geometry_mode": "gt_ego_debug"}, label="leak")


def test_core_candidate_contract_requires_one_base_and_four_siblings() -> None:
    condition = {"condition_id": "condition"}
    candidates = [{
        "condition_id": "condition",
        "candidate_id": "base",
        "candidate_role": "base_guard",
        "branch_family": "base_guard",
    }]
    candidates.extend({
        "condition_id": "condition",
        "candidate_id": f"sibling-{index}",
        "candidate_role": "sibling",
        "branch_family": "common_prefix",
        "prefix_trace_hash": "same",
    } for index in range(4))
    grouped = _core_candidates([condition], candidates)
    assert len(grouped["condition"]["siblings"]) == 4

    with pytest.raises(PreferenceReauditError, match="1 Base"):
        _core_candidates([condition], candidates[:-1])


def test_historical_artifact_fingerprint_change_is_rejected(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"status": "frozen"}), encoding="utf-8")
    expected = _source_fingerprints(tmp_path)
    _ensure_source_immutable(tmp_path, expected)
    (tmp_path / "summary.json").write_text(json.dumps({"status": "changed"}), encoding="utf-8")
    with pytest.raises(PreferenceReauditError, match="被修改"):
        _ensure_source_immutable(tmp_path, expected)
