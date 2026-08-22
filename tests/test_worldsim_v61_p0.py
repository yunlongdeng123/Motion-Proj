from pathlib import Path

import pytest

from motion_proj.worldsim_v61.p0_scope_freeze import (
    P0ContractError,
    _evaluate_r10_baseline,
    _prepare_run_root,
    _resolve_uri,
)


def test_resolve_uri_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(P0ContractError):
        _resolve_uri(tmp_path, "repo://../secret")


def test_r10_baseline_requires_exact_accepted_ids() -> None:
    metrics = {
        "case_count": 2,
        "accept_count": 1,
        "abstain_count": 0,
        "reject_count": 1,
        "false_safe_count": 0,
        "accepted_mask_pixels": 12,
    }
    decisions = [
        {"case_id": "a", "overall_decision": "ACCEPT", "false_safe": False},
        {"case_id": "b", "overall_decision": "REJECT", "false_safe": False},
    ]
    expected = {
        "case_count": 2,
        "accept_count": 1,
        "abstain_count": 0,
        "reject_count": 1,
        "accepted_mask_pixels": 12,
        "accepted_case_ids": ["a"],
    }
    assert _evaluate_r10_baseline(metrics, decisions, expected)["checks"]["passed"]
    expected["accepted_case_ids"] = ["b"]
    assert not _evaluate_r10_baseline(metrics, decisions, expected)["checks"]["passed"]


def test_prepare_run_root_creates_missing_namespace(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/worldsim_v61"
    assert not run_root.exists()
    assert _prepare_run_root(run_root) > 0.0
    assert run_root.is_dir()
