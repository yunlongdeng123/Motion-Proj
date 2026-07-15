from __future__ import annotations

import copy
from typing import Any

import pytest

from motion_proj.data.physics_dpo_schema import (
    PREFERENCE_SCHEMA_VERSION,
    PhysicsDpoSchemaError,
    build_scene_split,
    make_condition_id,
    validate_scene_split,
    validate_training_dataset,
)
from motion_proj.runtime.fingerprint import sha256_json


def _source(split: str, prefix: str, scenes: int) -> dict[str, Any]:
    scene_rows = []
    clip_rows = []
    for index in range(scenes):
        token = f"{prefix}-scene-token-{index:02d}"
        name = f"{prefix}-scene-{index:02d}"
        scene_rows.append({"scene_name": name, "scene_token": token, "sample_count": 16, "clip_count": 2})
        for clip_index in range(2):
            clip_rows.append({"clip_id": f"{prefix}-clip-{index:02d}-{clip_index}", "scene_name": name, "scene_token": token})
    core = {"split": split, "camera": "CAM_FRONT", "num_frames": 8, "scenes": scene_rows, "clips": clip_rows}
    return {**core, "split_fingerprint": sha256_json(core)}


def _split() -> dict[str, Any]:
    return build_scene_split(_source("train", "train", 5), _source("val", "val", 4), salt="fixed", preference_dev_scene_count=1, screen_eval_scene_count=1)


def _condition(split: dict[str, Any]) -> dict[str, Any]:
    partition = split["partitions"]["preference_train"]
    scene, clip = partition["scenes"][0], partition["clips"][0]
    record = {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "scene_id": scene["scene_name"], "scene_token": scene["scene_token"], "clip_id": clip["clip_id"],
        "split": "preference_train", "camera": "CAM_FRONT", "conditioning_frame": 0,
        "condition_frame_sha256": "condition-frame", "num_frames": 8, "fps": 7,
        "generation_protocol": "svd_official_v1", "scheduler_fingerprint": "scheduler",
        "base_model_fingerprint": "base", "uses_future_gt": False, "git_commit": "commit", "config_fingerprint": "config",
    }
    record["condition_id"] = make_condition_id(record)
    return record


def _candidate(condition: dict[str, Any], suffix: str, *, role: str, direction: str, group: str, strength: float, rms: float, family: str) -> dict[str, Any]:
    return {
        "candidate_id": f"candidate-{suffix}", "condition_id": condition["condition_id"], "scene_id": condition["scene_id"],
        "split": condition["split"], "candidate_role": role, "rgb_video_path": f"{suffix}.mp4",
        "vae_latent_path": f"{suffix}.pt", "diagnostics_path": f"{suffix}.json", "generation_protocol": "svd_official_v1",
        "base_model_fingerprint": "base", "scheduler_fingerprint": "scheduler", "initial_latent_hash": "initial",
        "prefix_latent_hash": "prefix", "prefix_trace_hash": "trace", "fork_step": 15, "branch_family": family,
        "branch_direction": direction, "branch_strength": strength, "perturbation_rms": rms, "antithetic_group_id": group,
        "generation_seed": 7, "guidance_schedule": [1.0, 3.0], "num_frames": 8, "fps": 7,
        "uses_future_gt": False, "git_commit": "commit", "config_fingerprint": "config",
    }


def _dataset() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split = _split()
    condition = _condition(split)
    candidates = [
        _candidate(condition, "base", role="base_guard", direction="base", group="base", strength=0.0, rms=0.0, family="base_guard"),
        _candidate(condition, "p1", role="sibling", direction="positive", group="g1", strength=0.1, rms=0.2, family="common_prefix"),
        _candidate(condition, "n1", role="sibling", direction="negative", group="g1", strength=0.1, rms=0.2, family="common_prefix"),
        _candidate(condition, "p2", role="sibling", direction="positive", group="g2", strength=0.1, rms=0.2, family="common_prefix"),
        _candidate(condition, "n2", role="sibling", direction="negative", group="g2", strength=0.1, rms=0.2, family="common_prefix"),
    ]
    preference = {
        "pair_id": "pair-1", "condition_id": condition["condition_id"], "candidate_a": "candidate-p1", "candidate_b": "candidate-n1",
        "split": "preference_train", "global_label": "a_wins", "winner_candidate_id": "candidate-p1", "loser_candidate_id": "candidate-n1",
        "feasibility_a": "feasible", "feasibility_b": "feasible", "physics_components": {"projection": 1.0},
        "quality_components": {"quality": 1.0}, "preference_margin": 0.1, "pair_confidence": 0.8,
        "scorer_fingerprint": "scorer", "human_review_id": None, "abstain_reason": None, "uses_future_gt": False,
    }
    segment = {
        "segment_id": "segment-1", "pair_id": "pair-1", "start_frame": 0, "end_frame": 4, "label": "a_wins",
        "winner_candidate_id": "candidate-p1", "loser_candidate_id": "candidate-n1", "confidence": 0.7,
        "violation_decomposition": {"projection": 1.0}, "frame_alignment_pass": True, "abstain_reason": None,
    }
    return split, [condition], candidates, [preference], [segment]


def test_scene_split_is_deterministic_and_scene_disjoint() -> None:
    first = _split()
    second = _split()

    assert first["split_fingerprint"] == second["split_fingerprint"]
    owners = validate_scene_split(first)
    assert len(owners["scene_owner"]) == 9
    assert first["partitions"]["preference_dev"]["scene_count"] == 1
    assert first["partitions"]["screen_eval"]["scene_count"] == 1
    assert first["partitions"]["preference_train"]["clip_count"] == 8
    assert first["partitions"]["formal_test"]["clip_count"] == 6


def test_training_schema_accepts_aligned_four_sibling_group() -> None:
    split, conditions, candidates, preferences, segments = _dataset()

    validated = validate_training_dataset(split, conditions, candidates, preferences, segments, require_segments=True)

    assert validated["condition_count"] == 1
    assert validated["candidate_count"] == 5
    assert validated["preference_count"] == 1
    assert validated["segment_count"] == 1


def test_training_schema_accepts_permutation_rms_roundoff_but_rejects_real_strength_gap() -> None:
    split, conditions, candidates, preferences, segments = _dataset()
    near_equal = copy.deepcopy(candidates)
    near_equal[3]["perturbation_rms"] = 0.20000001
    near_equal[4]["perturbation_rms"] = 0.20000001

    validated = validate_training_dataset(split, conditions, near_equal, preferences, segments, require_segments=True)
    assert validated["candidate_count"] == 5

    mismatched = copy.deepcopy(candidates)
    mismatched[3]["perturbation_rms"] = 0.21
    mismatched[4]["perturbation_rms"] = 0.21
    with pytest.raises(PhysicsDpoSchemaError, match="等范数"):
        validate_training_dataset(split, conditions, mismatched, preferences, segments, require_segments=True)


def test_training_schema_rejects_future_gt_and_all_tie_pair() -> None:
    split, conditions, candidates, preferences, segments = _dataset()
    bad_conditions = copy.deepcopy(conditions)
    bad_conditions[0]["uses_future_gt"] = True
    with pytest.raises(PhysicsDpoSchemaError, match="future_gt"):
        validate_training_dataset(split, bad_conditions, candidates, preferences, segments, require_segments=True)

    tie_segments = copy.deepcopy(segments)
    tie_segments[0].update({"label": "tie", "winner_candidate_id": None, "loser_candidate_id": None, "abstain_reason": "no margin"})
    with pytest.raises(PhysicsDpoSchemaError, match="all-tie"):
        validate_training_dataset(split, conditions, candidates, preferences, tie_segments, require_segments=True)


def test_training_schema_rejects_cross_condition_pair_and_unpaired_sibling() -> None:
    split, conditions, candidates, preferences, segments = _dataset()
    bad_candidates = candidates[:-1]
    with pytest.raises(PhysicsDpoSchemaError, match="sibling 数"):
        validate_training_dataset(split, conditions, bad_candidates, preferences, segments, require_segments=True)

    second = _condition(split)
    second["clip_id"] = split["partitions"]["preference_train"]["clips"][1]["clip_id"]
    second["condition_frame_sha256"] = "different-frame"
    second["condition_id"] = make_condition_id(second)
    other = _candidate(second, "other", role="base_guard", direction="base", group="base2", strength=0.0, rms=0.0, family="base_guard")
    cross_preferences = copy.deepcopy(preferences)
    cross_preferences[0]["candidate_b"] = other["candidate_id"]
    cross_preferences[0]["loser_candidate_id"] = other["candidate_id"]
    with pytest.raises(PhysicsDpoSchemaError, match="跨 condition"):
        validate_training_dataset(split, conditions + [second], candidates + [other], cross_preferences, segments, require_segments=True, exact_sibling_count=None)
