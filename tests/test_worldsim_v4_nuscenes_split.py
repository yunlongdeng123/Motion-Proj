from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from motion_proj.worldsim_v4.datasets.nuscenes import (
    REQUIRED_CHANNELS,
    ROLES,
    CohortError,
    build_frame_partitions,
    canonical_json_sha256,
    classify_scene_context,
    select_scene_cohort,
    validate_cohort,
)
from scripts.build_worldsim_v4_nuscenes_cohort import _record_blocked_terminal


def _frames(count: int = 40) -> list[dict]:
    return [
        {"token": f"sample-{index:02d}", "timestamp": index * 500_000}
        for index in range(count)
    ]


def _scene(index: int, split: str) -> dict:
    partitions = build_frame_partitions(_frames())
    return {
        "scene": f"scene-{index:04d}",
        "scene_token": f"token-{index:04d}",
        "official_split": split,
        "description": "intersection with moving car",
        "location": ["boston-seaport", "singapore-onenorth", "singapore-queenstown"][index % 3],
        "time_of_day": ["day", "dusk", "night"][index % 3],
        "weather": ["dry_or_unspecified", "rain"][index % 2],
        "road_geometry": ["intersection", "turn_or_curve", "road_segment"][index % 3],
        "actor_class": ["car", "truck", "bus"][index % 3],
        "speed_regime": ["stationary", "low_speed", "normal_speed"][index % 3],
        "distance_regime": ["near", "mid", "far"][index % 3],
        "occlusion": ["normal", "heavy"][index % 2],
        "donor_support": ["strong", "medium", "weak"][index % 3],
        "donor_support_is_metadata_proxy": True,
        "eligible_actor_count": index % 7,
        "actors": {
            "high_support": {"instance_token": f"actor-{index}"},
            "difficult": {"status": "ABSTAIN_NO_DIFFICULT_ACTOR"},
        },
        "edits": {"remove": f"actor-{index}", "lateral": f"actor-{index}", "insert": f"actor-{index}"},
        "continuous_clip": {
            "status": "ready",
            "actor_instance_token": f"actor-{index}",
            "sample_tokens": [f"sample-{i:02d}" for i in range(7)],
            "duration_s": 3.0,
        },
        "camera_set": list(REQUIRED_CHANNELS[:3]),
        "lidar": "LIDAR_TOP",
        "sensor_contract_complete": True,
        "sample_count": 40,
        **partitions,
    }


def _manifest(selected: list[dict]) -> dict:
    return {
        "schema_version": "worldsim_v4_nuscenes_cohort_v1",
        "task_id": "WS-V4-D0-NUSCENES-COHORT-01",
        "status": "done",
        "selection_uses_model_results": False,
        "selection_fields": ["official_split", "location", "weather"],
        "scenes": selected,
    }


def test_context_classification_is_result_free() -> None:
    result = classify_scene_context(
        "Rain, arrive at intersection and turn left",
        "n015-2018-11-14-19-09-14+0800",
    )
    assert result == {
        "time_of_day": "night",
        "weather": "rain",
        "road_geometry": "intersection",
    }


def test_frame_partitions_are_complete_disjoint_and_stable() -> None:
    result = build_frame_partitions(_frames())
    assert [len(result[key]) for key in ("train_frames", "development_frames", "heldout_frames")] == [24, 8, 8]
    tokens = [row["sample_token"] for rows in result.values() for row in rows]
    assert len(tokens) == len(set(tokens)) == 40
    assert result == build_frame_partitions(list(reversed(_frames())))


def test_selection_is_deterministic_scene_disjoint_and_split_safe() -> None:
    candidates = [_scene(index, "train" if index < 50 else "val") for index in range(80)]
    anchors = ["scene-0001", "scene-0002"]
    first = select_scene_cohort(candidates, seed=40117, development_anchors=anchors)
    second = select_scene_cohort(list(reversed(candidates)), seed=40117, development_anchors=anchors)
    assert first == second
    assert len(first) == 30
    assert len({row["scene"] for row in first}) == 30
    assert {role: sum(row["role"] == role for row in first) for role in ROLES} == ROLES
    assert all(row["official_split"] == "val" for row in first if row["role"] == "test")
    assert all(row["official_split"] == "train" for row in first if row["role"] != "test")
    assert {row["scene"] for row in first if row["role"] == "development"}.issuperset(anchors)


def test_validate_cohort_fails_on_leak_and_model_result_selection() -> None:
    candidates = [_scene(index, "train" if index < 50 else "val") for index in range(80)]
    selected = select_scene_cohort(candidates, seed=7, development_anchors=[])
    manifest = _manifest(selected)
    validate_cohort(manifest)
    leaked = copy.deepcopy(manifest)
    next(row for row in leaked["scenes"] if row["role"] == "test")["official_split"] = "train"
    with pytest.raises(CohortError, match="泄漏"):
        validate_cohort(leaked)
    quality = copy.deepcopy(manifest)
    quality["selection_fields"].append("psnr")
    with pytest.raises(CohortError, match="模型质量"):
        validate_cohort(quality)


def test_manifest_hash_is_canonical() -> None:
    left = {"b": [2, 1], "a": {"x": True}}
    right = {"a": {"x": True}, "b": [2, 1]}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_validate_cohort_rejects_out_of_contract_continuous_clip() -> None:
    candidates = [_scene(index, "train" if index < 50 else "val") for index in range(80)]
    manifest = _manifest(select_scene_cohort(candidates, seed=13, development_anchors=[]))
    manifest["scenes"][0]["continuous_clip"]["duration_s"] = 1.5
    with pytest.raises(CohortError):
        validate_cohort(manifest)


def test_selection_is_stable_across_python_hash_seeds() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = """
import runpy
from motion_proj.worldsim_v4.datasets.nuscenes import canonical_json_sha256, select_scene_cohort
namespace = runpy.run_path('tests/test_worldsim_v4_nuscenes_split.py')
scene = namespace['_scene']
candidates = [scene(index, 'train' if index < 50 else 'val') for index in range(80)]
selected = select_scene_cohort(candidates, seed=40117, development_anchors=['scene-0001'])
print(canonical_json_sha256(selected))
"""
    fingerprints = set()
    for hash_seed in ("1", "17", "40117"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(project_root)}
        fingerprints.add(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=project_root,
                env=env,
                text=True,
            ).strip()
        )
    assert len(fingerprints) == 1


def test_preflight_failure_records_immutable_blocked_terminal(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("task_id: WS-V4-D0-NUSCENES-COHORT-01\n", encoding="utf-8")
    run_dir = tmp_path / "blocked-run"
    _record_blocked_terminal(config, run_dir, tmp_path, CohortError("frozen drift"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert status["status"] == "blocked"
    assert manifest["status"] == "blocked"
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "fingerprint.json").is_file()
