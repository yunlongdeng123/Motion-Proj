from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from motion_proj.cfbench.evaluate import aggregate_results
from motion_proj.cfbench.geometry import footprint, footprints_overlap, shifted_pose
from motion_proj.cfbench.registry import support_for_case, validate_registry
from motion_proj.cfbench.schema import (
    DIMENSIONS,
    INTERVENTION_FAMILIES,
    ContractError,
    validate_manifest,
    validate_result,
)
from scripts.build_worldsim_v75_downstream_bench import build_manifest
from scripts.materialize_worldsim_v75_downstream_inputs import _trajectory
from scripts.preflight_worldsim_v75_downstream_bench import _path_group


def _write_scene(root: Path, scene_id: str, actor_offset: int) -> None:
    instance_dir = root / scene_id / "instances"
    instance_dir.mkdir(parents=True)
    info = {}
    for actor_index in range(8):
        actor_key = str(actor_offset + actor_index)
        transforms = []
        for frame in range(40):
            transforms.append(
                [
                    [1.0, 0.0, 0.0, float(frame)],
                    [0.0, 1.0, 0.0, float(actor_index * 10)],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        info[actor_key] = {
            "id": f"actor-{scene_id}-{actor_index}",
            "class_name": "vehicle.car",
            "frame_annotations": {
                "frame_idx": list(range(40)),
                "obj_to_world": transforms,
                "box_size": [[4.0, 1.8, 1.5]] * 40,
            },
        }
    (instance_dir / "instances_info.json").write_text(json.dumps(info), encoding="utf-8")
    (instance_dir / "frame_instances.json").write_text(
        json.dumps(
            {
                str(frame): list(range(actor_offset, actor_offset + 8))
                for frame in range(196)
            }
        ),
        encoding="utf-8",
    )


def test_builder_creates_balanced_24_case_pilot(tmp_path: Path) -> None:
    for index, scene_id in enumerate(("179", "191", "204")):
        _write_scene(tmp_path, scene_id, index * 8)
    manifest = build_manifest(tmp_path)
    validate_manifest(manifest)
    assert len(manifest["cases"]) == 24
    counts = {family: 0 for family in INTERVENTION_FAMILIES}
    for case in manifest["cases"]:
        counts[case["intervention"]["family"]] += 1
    assert set(counts.values()) == {6}
    assert sum(case["target"]["role"] == "ego" for case in manifest["cases"]) == 6
    assert all(case["qualification"]["human_verdict"] is None for case in manifest["cases"])


def test_manifest_rejects_composite_score(tmp_path: Path) -> None:
    for index, scene_id in enumerate(("179", "191", "204")):
        _write_scene(tmp_path, scene_id, index * 8)
    manifest = build_manifest(tmp_path)
    manifest["benchmark"]["no_composite_score"] = False
    with pytest.raises(ContractError, match="no_composite_score"):
        validate_manifest(manifest)


def test_manifest_rejects_more_than_one_changed_variable(tmp_path: Path) -> None:
    for index, scene_id in enumerate(("179", "191", "204")):
        _write_scene(tmp_path, scene_id, index * 8)
    manifest = build_manifest(tmp_path)
    intervention = manifest["cases"][0]["intervention"]
    intervention["factual"]["yaw_delta_deg"] = 0.0
    intervention["counterfactual"]["yaw_delta_deg"] = 15.0
    with pytest.raises(ContractError, match="exactly intervention.variable"):
        validate_manifest(manifest)


def test_registry_distinguishes_native_consumer_and_unsupported() -> None:
    registry = {
        "schema_version": "worldsim_v75_cfbench_model_registry_v1",
        "models": [
            {
                "model_id": "m",
                "capabilities": {
                    "ego:actor_speed_change": "native",
                    "non_ego:actor_removal": "consumer_only",
                },
            }
        ],
    }
    validate_registry(registry)
    ego_case = {"target": {"role": "ego"}, "intervention": {"family": "actor_speed_change"}}
    remove_case = {"target": {"role": "non_ego"}, "intervention": {"family": "actor_removal"}}
    insert_case = {"target": {"role": "non_ego"}, "intervention": {"family": "actor_insertion"}}
    assert support_for_case(registry["models"][0], ego_case) == "native"
    assert support_for_case(registry["models"][0], remove_case) == "consumer_only"
    assert support_for_case(registry["models"][0], insert_case) == "unsupported"


def _result(model_id: str, value: float) -> dict:
    return {
        "schema_version": "worldsim_v75_cfbench_result_v1",
        "model_id": model_id,
        "case_id": "c1",
        "status": "completed",
        "metrics": {
            dimension: {"status": "scored", "value": value, "evidence": "unit-test"}
            for dimension in DIMENSIONS
        },
    }


def test_aggregate_keeps_dimensions_separate() -> None:
    result = _result("model-a", 0.5)
    validate_result(result)
    aggregate = aggregate_results([result])
    assert aggregate["no_composite_score"] is True
    assert "composite" not in aggregate["models"]["model-a"]
    assert tuple(aggregate["models"]["model-a"]["dimensions"]) == DIMENSIONS


def test_result_rejects_overall_score() -> None:
    result = _result("model-a", 1.0)
    result["overall_score"] = 1.0
    with pytest.raises(ContractError, match="composite"):
        validate_result(result)


def test_counterfactual_footprint_collision_gate() -> None:
    pose = np.eye(4)
    moved = shifted_pose(pose, [8.0, 0.0, 0.0])
    near = shifted_pose(pose, [1.0, 0.0, 0.0])
    base_footprint = footprint(pose, [4.0, 2.0, 1.5])
    assert not footprints_overlap(base_footprint, footprint(moved, [4.0, 2.0, 1.5]))
    assert footprints_overlap(base_footprint, footprint(near, [4.0, 2.0, 1.5]))


def test_resim_adapter_keeps_branch_origin_fixed() -> None:
    poses = {}
    for frame in range(64):
        pose = np.eye(4)
        pose[0, 3] = float(frame)
        poses[frame] = pose
    case = {
        "case_id": "ego-speed",
        "anchor": {"event_frame": 5, "rollout_frames": 19},
        "intervention": {
            "family": "actor_speed_change",
            "factual": {"speed_scale": 1.0},
            "counterfactual": {"speed_scale": 0.5},
        },
    }
    factual = _trajectory(case, poses, "factual")
    counterfactual = _trajectory(case, poses, "counterfactual")
    assert factual[0] == counterfactual[0] == [0.0, 0.0, 0.0]
    assert factual[-1][0] == pytest.approx(18.0)
    assert counterfactual[-1][0] == pytest.approx(9.0)


def test_preflight_rejects_aria2_partial(tmp_path: Path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"partial")
    Path(f"{weight}.aria2").write_bytes(b"state")
    row = {"weights": {"all_of": [str(weight)]}}
    result = _path_group(row, "weights")
    assert result["passed"] is False
    assert result["incomplete_matches"][str(weight)][str(weight)] == "aria2_in_progress"


def test_preflight_enforces_minimum_size(tmp_path: Path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"short")
    pattern = str(weight)
    row = {
        "weights": {
            "all_of": [pattern],
            "min_size_bytes": {pattern: 10},
        }
    }
    result = _path_group(row, "weights")
    assert result["passed"] is False
    assert result["incomplete_matches"][pattern][pattern] == "size_below_minimum"
