from __future__ import annotations

import pytest

from scripts.prepare_worldsim_v32_s1_prompts import resolve_optimization_frames


def test_resolve_optimization_frames_excludes_development_and_heldout() -> None:
    config = {
        "split": {
            "development_frames": [2, 7],
            "heldout_frames": [4, 9],
        }
    }

    train, development, heldout = resolve_optimization_frames(config, 10)

    assert train == [0, 1, 3, 5, 6, 8]
    assert development == [2, 7]
    assert heldout == [4, 9]


def test_resolve_optimization_frames_keeps_v32_backward_compatibility() -> None:
    train, development, heldout = resolve_optimization_frames(
        {"split": {"heldout_frames": [4]}}, 5
    )

    assert train == [0, 1, 2, 3]
    assert development == []
    assert heldout == [4]


def test_resolve_optimization_frames_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        resolve_optimization_frames(
            {"split": {"development_frames": [2], "heldout_frames": [2]}}, 5
        )
