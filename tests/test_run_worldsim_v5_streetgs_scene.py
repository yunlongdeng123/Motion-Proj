from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.run_worldsim_v5_streetgs_scene import (
    COHORT_SHA256,
    FRAME_CONTRACT,
    SCENE_CONTRACT,
    V5StreetGSTrainingError,
    build_train_command,
    load_training_config,
    sha256_file,
    validate_config,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict:
    patch = tmp_path / "compatibility.patch"
    patch.write_text("patch\n", encoding="utf-8")
    cohort_path = tmp_path / "cohort.yaml"
    cohort = {
        "schema_version": "worldsim_v5_nuscenes_fresh_cohort_v1",
        "status": "done",
        "freeze": {
            "cohort_sha256": COHORT_SHA256,
            "scene_roles": {"development": list(SCENE_CONTRACT)},
        },
    }
    cohort_path.write_text(yaml.safe_dump(cohort, sort_keys=False), encoding="utf-8")
    preprocess_run = tmp_path / "preprocess_run"
    processed_root = Path(
        "/root/autodl-tmp/data/worldsim_v5/drivestudio_processed_10Hz/trainval"
    )
    _write_json(
        preprocess_run / "summary.json",
        {
            "schema_version": "worldsim_v5_m1_preprocess_summary_v1",
            "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
            "stage": "development_preprocess",
            "status": "done",
            "scene_count": 8,
            "processed_root": str(processed_root),
            "quality_read": False,
            "training_started": False,
            "model_inference_started": False,
            "project_git": {"head": "preprocess-commit", "dirty": False},
        },
    )
    artifacts = {}
    sky_runs = {}
    for scene, index in SCENE_CONTRACT.items():
        artifact = preprocess_run / "artifacts" / f"{scene}.json"
        inventory_sha = f"inventory-{scene}"
        _write_json(
            artifact,
            {
                "schema_version": "worldsim_v5_m1_processed_scene_v1",
                "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
                "status": "done",
                "scene_name": scene,
                "scene_index": index,
                "output": str(processed_root / f"{index:03d}"),
                "quality_read": False,
                "training_started": False,
                "model_inference_started": False,
                "inventory_sha256": inventory_sha,
                "inventory": [
                    {
                        "path": "images/000_0.jpg",
                        "bytes": 1,
                        "sha256": "0" * 64,
                    }
                ],
            },
        )
        artifacts[scene] = {
            "path": str(artifact),
            "file_sha256": sha256_file(artifact),
            "inventory_sha256": inventory_sha,
        }
        sky_run = tmp_path / "sky_runs" / scene
        masks = [
            {
                "image": f"{frame:03d}_{camera}.jpg",
                "mask": f"{frame:03d}_{camera}.png",
                "bytes": 1,
                "sha256": "0" * 64,
            }
            for frame in range(FRAME_CONTRACT[scene])
            for camera in (0, 1, 2)
        ]
        _write_json(
            sky_run / "summary.json",
            {
                "schema_version": "worldsim_v5_sky_mask_summary_v1",
                "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
                "status": "done",
                "scene": scene,
                "mask_count": len(masks),
                "segmentation_inference_started": True,
                "method_inference_started": False,
                "network_accessed": False,
                "test_quality_read": False,
            },
        )
        _write_json(sky_run / "manifest.json", {"status": "done"})
        _write_json(
            sky_run / "artifacts/sky_mask_manifest.json",
            {
                "schema_version": "worldsim_v5_sky_mask_manifest_v1",
                "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
                "status": "done",
                "scene": scene,
                "scene_index": index,
                "expected_timesteps": FRAME_CONTRACT[scene],
                "mask_count": len(masks),
                "model": {
                    "revision": "2c6f153e4c23c229e2fa2b188eb250607e030cd8"
                },
                "files": masks,
                "network_accessed": False,
                "test_quality_read": False,
            },
        )
        sky_runs[scene] = {
            "run": str(sky_run),
            "summary_sha256": sha256_file(sky_run / "summary.json"),
            "run_manifest_sha256": sha256_file(sky_run / "manifest.json"),
            "artifact_sha256": sha256_file(
                sky_run / "artifacts/sky_mask_manifest.json"
            ),
        }
    return {
        "schema_version": "worldsim_v5_streetgs_training_v1",
        "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
        "status": "running",
        "phase": "development_base_reconstruction",
        "implementation": {
            "upstream_root": "/upstream",
            "upstream_commit": "commit",
            "expected_git_status": "M datasets/driving_dataset.py",
            "compatibility_patch": patch.name,
            "compatibility_patch_sha256": sha256_file(patch),
            "environment": "/env",
            "config_file": "configs/streetgs.yaml",
        },
        "fresh_cohort_binding": {
            "config": cohort_path.name,
            "config_sha256": sha256_file(cohort_path),
            "cohort_sha256": COHORT_SHA256,
        },
        "preprocess_binding": {
            "run": str(preprocess_run),
            "summary_sha256": sha256_file(preprocess_run / "summary.json"),
            "source_commit": "preprocess-commit",
            "scene_artifacts": artifacts,
        },
        "sky_mask_binding": {
            "model_revision": "2c6f153e4c23c229e2fa2b188eb250607e030cd8",
            "camera_ids": [0, 1, 2],
            "total_mask_count": 4704,
            "scene_runs": sky_runs,
        },
        "data": {
            "processed_root": str(processed_root),
            "dataset_config": "nuscenes/3cams",
            "start_timestep": 0,
            "end_timestep": -1,
            "test_image_stride": 0,
            "load_smpl": False,
            "expected_cameras": 6,
            "expected_frames_by_scene": dict(FRAME_CONTRACT),
            "frame_partition": {
                "modulus": 5,
                "development_remainder": 2,
                "heldout_remainder": 4,
                "train_remainders": [0, 1, 3],
            },
        },
        "scenes": dict(SCENE_CONTRACT),
        "training": {"seed": 0, "modes": {"profile100": 100, "formal": 30000}},
        "restrictions": {
            "validation_quality_read": False,
            "test_quality_read": False,
            "post_train_render": False,
        },
    }


def test_v5_training_contract_binds_fresh_cohort_and_preprocess(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    result = validate_config(config, tmp_path)
    assert result["scene_count"] == 8
    assert set(result["preprocess_bindings"]) == set(SCENE_CONTRACT)
    assert set(result["sky_mask_bindings"]) == set(SCENE_CONTRACT)
    command, checkpoint, iterations = build_train_command(
        config, "scene-0471", "formal", tmp_path / "run"
    )
    assert iterations == 30000
    assert "data.scene_idx=382" in command
    assert "+data.pixel_source.excluded_remainders=[2,4]" in command
    assert "render.render_test=false" in command
    assert checkpoint.name == "checkpoint_final.pth"


def test_sky_bound_overlay_resolves_immutable_base(tmp_path: Path) -> None:
    base = _fixture(tmp_path)
    base.pop("sky_mask_binding")
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    overlay_path = tmp_path / "overlay.yaml"
    overlay = {
        "schema_version": "worldsim_v5_streetgs_training_binding_v1",
        "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
        "status": "running",
        "phase": "development_base_reconstruction",
        "base_config": {
            "path": base_path.name,
            "sha256": sha256_file(base_path),
        },
        "sky_mask_binding": _fixture(tmp_path)["sky_mask_binding"],
    }
    overlay_path.write_text(
        yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
    )

    resolved, binding = load_training_config(overlay_path, tmp_path)

    assert resolved["schema_version"] == "worldsim_v5_streetgs_training_v1"
    assert resolved["sky_mask_binding"] == overlay["sky_mask_binding"]
    assert binding == {
        "overlay_path": str(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "base_path": str(base_path),
        "base_sha256": sha256_file(base_path),
    }


def test_sky_bound_overlay_rejects_base_hash_drift(tmp_path: Path) -> None:
    base = _fixture(tmp_path)
    base.pop("sky_mask_binding")
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "worldsim_v5_streetgs_training_binding_v1",
                "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
                "status": "running",
                "phase": "development_base_reconstruction",
                "base_config": {"path": base_path.name, "sha256": "0" * 64},
                "sky_mask_binding": _fixture(tmp_path)["sky_mask_binding"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(V5StreetGSTrainingError, match="base reconstruction"):
        load_training_config(overlay_path, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["scenes"].pop("scene-0436"), "scenes"),
        (
            lambda value: value["data"]["expected_frames_by_scene"].update(
                {"scene-0471": 195}
            ),
            "frame",
        ),
        (lambda value: value["training"].update(seed=1), "seed/iteration"),
        (
            lambda value: value["data"]["frame_partition"].update(
                {"heldout_remainder": 3}
            ),
            "partition",
        ),
        (
            lambda value: value["restrictions"].update(
                {"validation_quality_read": True}
            ),
            "restriction",
        ),
    ],
)
def test_v5_training_contract_fails_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    config = copy.deepcopy(_fixture(tmp_path))
    mutation(config)
    with pytest.raises(V5StreetGSTrainingError, match=message):
        validate_config(config, tmp_path)


def test_preprocess_artifact_identity_drift_is_rejected(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    artifact = Path(
        config["preprocess_binding"]["scene_artifacts"]["scene-0471"]["path"]
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["scene_index"] = 381
    _write_json(artifact, payload)
    config["preprocess_binding"]["scene_artifacts"]["scene-0471"][
        "file_sha256"
    ] = sha256_file(artifact)
    with pytest.raises(V5StreetGSTrainingError, match="合同漂移"):
        validate_config(config, tmp_path)


def test_direct_script_entry_resolves_project_imports() -> None:
    project = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/run_worldsim_v5_streetgs_scene.py"),
            "--help",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "--scene" in process.stdout
