from __future__ import annotations

import json

import pytest

from motion_proj.diagnostics.physics_preference_reaudit import (
    PreferenceReauditError,
    _assert_no_future_gt,
    _core_candidates,
    _ensure_source_immutable,
    _measurement_window_eligible,
    _observation_exactness,
    _source_fingerprints,
)
from motion_proj.preference.paired_tracks import RawTrackObservation


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


def test_measurement_rope_rejects_invalid_windows_and_requires_exact_rerun() -> None:
    import torch
    from types import SimpleNamespace

    points = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]])
    reference = RawTrackObservation(
        candidate_id="reference", query_set_hash="hash", raw_points=points,
        raw_visibility=torch.ones(1, 2, dtype=torch.bool), raw_confidence=torch.ones(1, 2),
        forward_backward_error=torch.zeros(1, 2),
    )
    repeated = RawTrackObservation(
        candidate_id="repeated", query_set_hash="hash", raw_points=points.clone(),
        raw_visibility=torch.ones(1, 2, dtype=torch.bool), raw_confidence=torch.ones(1, 2),
        forward_backward_error=torch.zeros(1, 2),
    )
    assert _observation_exactness(reference, repeated)["exact"]
    changed = RawTrackObservation(
        candidate_id="changed", query_set_hash="hash", raw_points=points + 0.1,
        raw_visibility=torch.ones(1, 2, dtype=torch.bool), raw_confidence=torch.ones(1, 2),
        forward_backward_error=torch.zeros(1, 2),
    )
    assert not _observation_exactness(reference, changed)["exact"]

    cfg = SimpleNamespace(upo=SimpleNamespace(relation=SimpleNamespace(maximum_camera_distance_px=1.0)))
    eligible, reasons = _measurement_window_eligible(
        SimpleNamespace(valid=False),
        SimpleNamespace(valid=True, reason=None, camera_distance_px=0.0),
        {"comparable": True},
        cfg,
    )
    assert not eligible
    assert reasons == ["support_invalid"]
