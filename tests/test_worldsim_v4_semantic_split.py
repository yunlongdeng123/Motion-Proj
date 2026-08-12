from __future__ import annotations

import pytest

from motion_proj.worldsim_v4.semantic_split import (
    SemanticSplitError,
    forbidden_frames_from_mask_manifest,
    validate_prompt_optimization_split,
)


def config() -> dict:
    return {
        "split": {
            "development_frames": [2, 7],
            "heldout_frames": [4, 9],
        }
    }


def prompt() -> dict:
    return {
        "train_frames": [0, 1, 3, 5, 6, 8],
        "development_frames": [2, 7],
        "heldout_frames": [4, 9],
        "development_excluded": True,
        "heldout_excluded": True,
    }


def test_prompt_and_mask_exclude_both_frozen_partitions() -> None:
    assert validate_prompt_optimization_split(config(), prompt()) == {2, 4, 7, 9}
    mask = {
        **prompt(),
        "masks": [{"frame": 0}, {"frame": 3}],
    }
    assert forbidden_frames_from_mask_manifest(config(), mask) == {2, 4, 7, 9}


def test_prompt_rejects_development_optimization_leak() -> None:
    value = prompt()
    value["train_frames"].append(2)
    with pytest.raises(SemanticSplitError, match="冻结分区"):
        validate_prompt_optimization_split(config(), value)


def test_mask_rejects_missing_development_proof() -> None:
    value = {
        **prompt(),
        "development_excluded": False,
        "masks": [{"frame": 0}],
    }
    with pytest.raises(SemanticSplitError, match="development_excluded"):
        forbidden_frames_from_mask_manifest(config(), value)


def test_legacy_config_without_development_stays_compatible() -> None:
    legacy_config = {"split": {"heldout_frames": [4, 9]}}
    legacy_prompt = {
        "train_frames": [0, 1, 2, 3],
        "heldout_frames": [4, 9],
        "heldout_excluded": True,
    }
    assert validate_prompt_optimization_split(legacy_config, legacy_prompt) == {4, 9}
