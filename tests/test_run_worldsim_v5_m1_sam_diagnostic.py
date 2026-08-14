from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_worldsim_v5_m1_sam_diagnostic import (
    SamDiagnosticError,
    load_config,
    normalize_box_logits,
    select_prompt_actors,
    validate_frame_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _actor(class_name: str, translations: list[float]) -> dict:
    poses = []
    for x in translations:
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose.tolist())
    return {
        "class_name": class_name,
        "frame_annotations": {
            "frame_idx": list(range(len(poses))),
            "obj_to_world": poses,
            "box_size": [[4.0, 2.0, 1.5] for _ in poses],
        },
    }


def test_prompt_actor_selection_is_result_blind_and_strict() -> None:
    instances = {
        "2": _actor("vehicle.car", [0.0, 0.6, 1.2]),
        "0": _actor("vehicle.car", [0.0, 0.5, 1.0]),
        "1": _actor("human.pedestrian.adult", [0.0, 2.0]),
        "3": _actor("vehicle.bicycle", [0.0, 2.0]),
    }
    selected = select_prompt_actors(instances, minimum_trajectory_m=1.0)
    assert list(selected) == [2]
    assert np.isclose(selected[2]["trajectory_distance_m"], 1.2)


def test_normalize_box_logits_accepts_single_and_batch_shapes() -> None:
    single = normalize_box_logits(np.zeros((1, 8, 9), dtype=np.float32), 1)
    batch = normalize_box_logits(np.zeros((2, 1, 8, 9), dtype=np.float32), 2)
    assert single.shape == (1, 8, 9)
    assert batch.shape == (2, 8, 9)
    with pytest.raises(SamDiagnosticError, match="shape"):
        normalize_box_logits(np.zeros((3, 8, 9), dtype=np.float32), 2)


def test_sparse_frame_contract_keeps_heldout_unread() -> None:
    config = {
        "scene": {"frame_count": 196},
        "split": {
            "modulus": 5,
            "train_remainders": [0, 1, 3],
            "development_remainder": 2,
            "heldout_remainder": 4,
            "evidence_frames": [0, 40, 80, 120, 160],
            "evaluation_frames": [2, 42, 82, 122, 162],
        },
    }
    evidence, evaluation = validate_frame_contract(config)
    assert evidence == [0, 40, 80, 120, 160]
    assert evaluation == [2, 42, 82, 122, 162]
    config["split"]["evaluation_frames"] = [4]
    with pytest.raises(SamDiagnosticError, match="development remainder"):
        validate_frame_contract(config)


def test_scene0471_config_freezes_sparse_train_development_split() -> None:
    config = load_config(
        ROOT / "configs/worldsim_v5/m1_sam_diagnostic_scene0471_v1.yaml"
    )
    evidence, evaluation = validate_frame_contract(config)
    assert evidence == [0, 40, 80, 120, 160]
    assert evaluation == [2, 42, 82, 122, 162]
    assert config["split"]["train_remainders"] == [0, 1, 3]
    assert config["split"]["heldout_remainder"] == 4
