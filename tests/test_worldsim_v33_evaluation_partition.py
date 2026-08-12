from __future__ import annotations

import pytest

from motion_proj.worldsim_v33.evaluation_partition import (
    manifest_evaluation_partition,
    resolve_evaluation_frames,
    resolve_forbidden_optimization_frames,
)


def config() -> dict:
    return {
        "split": {
            "development_frames": [2, 7],
            "heldout_frames": [4, 9],
        }
    }


def test_development_formal_forbids_development_and_heldout() -> None:
    assert resolve_forbidden_optimization_frames(
        config(), phase="formal", evaluation_partition="development"
    ) == {2, 4, 7, 9}


def test_legacy_heldout_manifest_and_formal_contract_remain_compatible() -> None:
    assert manifest_evaluation_partition({}) == "heldout"
    assert resolve_forbidden_optimization_frames(
        config(), phase="formal", evaluation_partition="heldout"
    ) == {4, 9}


def test_partition_frames_fail_closed_on_overlap() -> None:
    value = config()
    value["split"]["heldout_frames"] = [4, 7]
    with pytest.raises(ValueError, match="重叠"):
        resolve_evaluation_frames(value, "development")


def test_manifest_partition_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="非法"):
        manifest_evaluation_partition({"evaluation_partition": "test"})
