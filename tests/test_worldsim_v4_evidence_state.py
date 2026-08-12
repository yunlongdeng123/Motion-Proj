from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from motion_proj.worldsim_v4.evidence_state import (
    atomic_save_evidence_state,
    build_evidence_state,
    load_evidence_state,
)


def _inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    instance = {
        "gaussian_id": np.arange(3),
        "base_model": np.asarray([0, 0, 1]),
        "base_index": np.asarray([0, 1, 0]),
        "hard_instance_id": np.asarray([7, -1, 7]),
        "instance_opacity": np.asarray([0.8, 0.1, 0.98]),
    }
    semantic = {
        "semantic_score": np.asarray([0.9, 0.1, 1.0]),
        "visible_mass": np.asarray([2.0, 2.0, 1.0]),
        "depth_consistency_rate": np.asarray([1.0, 0.5, 1.0]),
        "num_positive_views": np.asarray([3, 0, 4]),
        "num_negative_views": np.asarray([1, 3, 0]),
        "labels": np.asarray([2, 0, 1]),
    }
    return instance, semantic


def test_state_initializes_from_o1_and_saves_required_posterior(tmp_path: Path) -> None:
    instance, semantic = _inputs()
    state = build_evidence_state(
        instance_field=instance,
        semantic_sidecar=semantic,
        actor_instance_id=7,
        actor_token="actor-token",
        prior_strength=2.0,
        unassigned_probability=0.0001,
        visibility_saturation_mass=1.0,
        mask_confidence_floor=0.25,
        depth_confidence_floor=0.25,
        lidar_confidence_floor=0.5,
        observed_authenticity=0.8,
    )
    assert set(("alpha", "beta", "posterior", "uncertainty")) <= set(state)
    assert 0.75 < state["posterior"][0] < instance["instance_opacity"][0]
    assert state["posterior"][1] < instance["instance_opacity"][1]
    assert state["lidar_weight"].tolist() == [0.5, 0.5, 1.0]

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    atomic_save_evidence_state(first, state)
    atomic_save_evidence_state(second, state)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    loaded = load_evidence_state(first)
    np.testing.assert_array_equal(loaded["gaussian_id"], state["gaussian_id"])
