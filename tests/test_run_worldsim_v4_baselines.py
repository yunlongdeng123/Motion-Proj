from __future__ import annotations

import hashlib
from types import SimpleNamespace
from pathlib import Path

import yaml

from scripts import run_worldsim_v4_baselines as baselines
from scripts.run_worldsim_v4_baselines import (
    _checkpoint_registration_state,
    _single_checkpoint_registration_state,
    _v33_chain_registration_state,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registration(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fingerprint = tmp_path / "fingerprint.json"
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.json"
    fingerprint.write_bytes(b"fingerprint")
    manifest.write_bytes(b"manifest")
    summary.write_bytes(b"summary")
    files = {}
    for name, payload in {
        "point_cloud.ply": b"point-cloud",
        "deform.pth": b"deform",
        "env.pth": b"environment",
    }.items():
        path = tmp_path / name
        path.write_bytes(payload)
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {
        "run": str(tmp_path),
        "step": 60000,
        "fingerprint_sha256": _sha256(fingerprint),
        "manifest_sha256": _sha256(manifest),
        "summary_sha256": _sha256(summary),
        "files": files,
    }


def test_adgs_checkpoint_registration_requires_three_exact_files(tmp_path: Path) -> None:
    result = _checkpoint_registration_state(_registration(tmp_path))

    assert result["state"] == "executable_exact"
    assert result["executable_exact"] is True
    assert result["all_files_exact"] is True
    assert set(result["files"]) == {"point_cloud.ply", "deform.pth", "env.pth"}
    assert all(row["exact"] for row in result["files"].values())


def test_adgs_checkpoint_registration_rejects_tamper(tmp_path: Path) -> None:
    registration = _registration(tmp_path)
    Path(registration["files"]["deform.pth"]["path"]).write_bytes(b"tampered")

    result = _checkpoint_registration_state(registration)

    assert result["state"] == "checkpoint_mismatch"
    assert result["all_files_exact"] is False
    assert result["files"]["deform.pth"]["sha256_exact"] is False


def test_adgs_checkpoint_registration_rejects_incomplete_schema(tmp_path: Path) -> None:
    registration = _registration(tmp_path)
    del registration["files"]["env.pth"]

    result = _checkpoint_registration_state(registration)

    assert result == {
        "state": "invalid_registration",
        "all_files_exact": False,
        "files": {},
    }


def test_single_checkpoint_registration_rejects_hash_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    registered = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }
    assert _single_checkpoint_registration_state(registered)["exact"] is True

    checkpoint.write_bytes(b"changed")
    result = _single_checkpoint_registration_state(registered)
    assert result["exact"] is False
    assert result["sha256_exact"] is False


def _v33_registration(tmp_path: Path, *, scene: str = "scene-0242") -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    stages = {
        name: {"status": "done"}
        for name in baselines.V33_REQUIRED_STAGES
    }
    chain = {
        "schema_version": "worldsim_v4_v33_scene_chain_v1",
        "scene": scene,
        "algorithm_commit": "e6663e1",
        "base_checkpoint_sha256": "base-sha",
        "partition_contract": "sample_index_mod_5",
        "test_quality_read": False,
        "stages": stages,
    }
    status = {"status": "done", "scene": scene, "test_quality_read": False}
    for name, payload in (
        ("scene_chain.json", chain),
        (
            "render_manifest.json",
            {
                "scene": scene,
                "split": "development",
                "test_quality_read": False,
                "rows": [{"frame": 2, "camera": 0}],
            },
        ),
        (
            "metrics.json",
            {
                "scene": scene,
                "split": "development",
                "test_quality_read": False,
                "rows": [{"psnr": 20.0, "ssim": 0.8, "lpips_alex": 0.2}],
            },
        ),
        ("summary.json", {"scene": scene}),
        ("manifest.json", {"scene": scene}),
        ("status.json", status),
    ):
        (tmp_path / name).write_text(
            baselines.canonical_json_bytes(payload).decode("utf-8"), encoding="utf-8"
        )
    files = {
        name: {
            "path": str(tmp_path / name),
            "bytes": (tmp_path / name).stat().st_size,
            "sha256": _sha256(tmp_path / name),
        }
        for name in baselines.V33_REQUIRED_FILES
    }
    return {
        "run": str(tmp_path),
        "algorithm_commit": "e6663e1",
        "base_checkpoint_sha256": "base-sha",
        "summary_sha256": _sha256(tmp_path / "summary.json"),
        "manifest_sha256": _sha256(tmp_path / "manifest.json"),
        "status_sha256": _sha256(tmp_path / "status.json"),
        "files": files,
    }


def test_v33_registration_requires_exact_complete_scene_chain(tmp_path: Path) -> None:
    result = _v33_chain_registration_state(
        _v33_registration(tmp_path), expected_scene="scene-0242"
    )

    assert result["state"] == "executable_exact"
    assert result["executable_exact"] is True
    assert result["chain_semantics_exact"] is True
    assert set(result["stage_states"]) == set(baselines.V33_REQUIRED_STAGES)


def test_v33_registration_accepts_explicit_stage_abstain(tmp_path: Path) -> None:
    registration = _v33_registration(tmp_path)
    chain_path = Path(registration["files"]["scene_chain.json"]["path"])
    chain = yaml.safe_load(chain_path.read_text(encoding="utf-8"))
    chain["stages"]["asset_harvester"] = {
        "status": "abstain",
        "reason": "insufficient_real_views",
    }
    chain_path.write_bytes(baselines.canonical_json_bytes(chain))
    registration["files"]["scene_chain.json"].update(
        bytes=chain_path.stat().st_size, sha256=_sha256(chain_path)
    )

    result = _v33_chain_registration_state(
        registration, expected_scene="scene-0242"
    )

    assert result["executable_exact"] is True
    assert result["stage_states"]["asset_harvester"]["status"] == "abstain"


def test_v33_registration_rejects_missing_or_unreasoned_stage(tmp_path: Path) -> None:
    registration = _v33_registration(tmp_path)
    chain_path = Path(registration["files"]["scene_chain.json"]["path"])
    chain = yaml.safe_load(chain_path.read_text(encoding="utf-8"))
    chain["stages"]["roadpatch"] = {"status": "abstain"}
    chain_path.write_bytes(baselines.canonical_json_bytes(chain))
    registration["files"]["scene_chain.json"].update(
        bytes=chain_path.stat().st_size, sha256=_sha256(chain_path)
    )

    result = _v33_chain_registration_state(
        registration, expected_scene="scene-0242"
    )

    assert result["executable_exact"] is False
    assert result["chain_semantics_exact"] is False


def test_v33_registration_rejects_abstain_for_required_core_stage(tmp_path: Path) -> None:
    registration = _v33_registration(tmp_path)
    chain_path = Path(registration["files"]["scene_chain.json"]["path"])
    chain = yaml.safe_load(chain_path.read_text(encoding="utf-8"))
    chain["stages"]["instance_field"] = {
        "status": "abstain",
        "reason": "not_allowed_for_core_stage",
    }
    chain_path.write_bytes(baselines.canonical_json_bytes(chain))
    registration["files"]["scene_chain.json"].update(
        bytes=chain_path.stat().st_size, sha256=_sha256(chain_path)
    )

    result = _v33_chain_registration_state(
        registration, expected_scene="scene-0242"
    )

    assert result["executable_exact"] is False
    assert result["stage_states"]["instance_field"]["valid"] is False


def test_v33_registration_accepts_proven_abstain_no_actor_chain(
    tmp_path: Path,
) -> None:
    registration = _v33_registration(tmp_path)
    chain_path = Path(registration["files"]["scene_chain.json"]["path"])
    registry_path = tmp_path / "actor_registry.json"
    registry = {
        "checkpoint_sha256": "base-sha",
        "actors": [
            {
                "instance_token": "high-token",
                "processed_true_instance_id": 21,
                "availability": "unavailable_initialization_filter",
                "rigid_model_index": None,
                "checkpoint_tensor_slice": {
                    "gaussian_count": 0,
                    "flat_index_ranges_half_open": [],
                },
            }
        ],
    }
    registry_path.write_bytes(baselines.canonical_json_bytes(registry))
    chain = yaml.safe_load(chain_path.read_text(encoding="utf-8"))
    chain["abstention"] = {
        "reason": "ABSTAIN_NO_ACTOR",
        "actor": {
            "instance_token": "high-token",
            "dataset_instance_id": 21,
            "availability": "unavailable_initialization_filter",
        },
        "actor_registry": {
            "path": str(registry_path),
            "bytes": registry_path.stat().st_size,
            "sha256": _sha256(registry_path),
        },
    }
    for stage in baselines.V33_NO_ACTOR_STAGES:
        chain["stages"][stage] = {
            "status": "abstain",
            "reason": "ABSTAIN_NO_ACTOR",
        }
    chain_path.write_bytes(baselines.canonical_json_bytes(chain))
    registration["files"]["scene_chain.json"].update(
        bytes=chain_path.stat().st_size, sha256=_sha256(chain_path)
    )

    result = _v33_chain_registration_state(
        registration, expected_scene="scene-0242"
    )

    assert result["executable_exact"] is True
    assert result["no_actor_proof"]["exact"] is True
    assert result["stage_states"]["instance_field"]["no_actor_abstain"] is True


def test_audit_counts_only_exact_adgs_runtime_and_checkpoint(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    street_root = tmp_path / "street"
    street_env = tmp_path / "street-env"
    adgs_root = tmp_path / "adgs"
    adgs_env = tmp_path / "adgs-env"
    v33_run = tmp_path / "v33"
    for path in (street_root, street_env, adgs_root, adgs_env, v33_run):
        path.mkdir()
    street_checkpoint = tmp_path / "street.pth"
    street_checkpoint.write_bytes(b"street")
    historical = tmp_path / "historical.json"
    historical.write_text('{"scenes": []}', encoding="utf-8")
    patch = project / "compatibility.patch"
    patch.write_bytes(b"patch")
    registration = _registration(tmp_path / "adgs-checkpoint")
    scenes = ["scene-0230", "scene-0242", "scene-0255", "scene-0048", "scene-0994", "scene-0139"]
    metrics = {
        "image": {
            "primary": ["psnr", "ssim", "lpips_alex"],
            "regions": ["global", "static", "actor", "boundary", "edit_roi"],
            "region_protocol": {
                "actor": "drivestudio_dynamic_masks_all_nonzero",
                "static": "not_actor_and_not_egocar",
                "boundary": "dynamic_mask_morphological_band_l1_radius_3px",
                "baseline_edit_roi": "empty_undefined",
            },
        },
        "statistics": {"unit": "scene", "denominator_policy": "retain_failed_blocked_abstain"},
    }
    cohort = {"freeze": {"scene_roles": {"development": scenes}}}
    matrix = {
        "task_id": baselines.TASK_ID,
        "status": "running",
        "metrics_config": "metrics.yaml",
        "cohort_config": "cohort.yaml",
        "scene_contract": {scene: {} for scene in scenes},
        "resolution_contract": {
            "sensor_rgb": [1600, 900],
            "source_config_downscale": 2,
            "model_native_render": [800, 450],
            "metric_resolution": [800, 450],
        },
        "baselines": {
            "streetgs": {
                "implementation_root": str(street_root),
                "environment": str(street_env),
                "implementation_commit": "street-commit",
                "checkpoints": {"scene-0230": {
                    "path": str(street_checkpoint),
                    "bytes": street_checkpoint.stat().st_size,
                    "sha256": _sha256(street_checkpoint),
                }},
            },
            "v33_frozen": {
                "source_run": str(v33_run),
                "legacy_executable_scenes": ["scene-0230"],
                "executable_scene_chains": {},
                "implementation_commit": "e6663e1",
            },
            "ad_gs": {
                "implementation_root": str(adgs_root),
                "environment": str(adgs_env),
                "implementation_commit": "adgs-commit",
                "expected_modified_files": ["train.py"],
                "compatibility_patch": "compatibility.patch",
                "compatibility_patch_sha256": _sha256(patch),
                "historical_metrics": str(historical),
                "executable_checkpoints": {"scene-0230": registration},
            },
        },
        "completion_gate": {"required_scene_count_per_method": 6},
    }
    for name, value in (("metrics.yaml", metrics), ("cohort.yaml", cohort), ("matrix.yaml", matrix)):
        (project / name).write_text(yaml.safe_dump(value), encoding="utf-8")

    def fake_git(path: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "adgs-commit" if path == adgs_root else "street-commit"
        if args == ("status", "--short"):
            return " M train.py" if path == adgs_root else ""
        raise AssertionError(args)

    monkeypatch.setattr(baselines, "_git", fake_git)
    monkeypatch.setattr(baselines.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    inventory = baselines.audit_matrix(project / "matrix.yaml", project)

    assert inventory["executable_scene_counts"] == {"streetgs": 1, "v33_frozen": 1, "ad_gs": 1}
    scene = inventory["methods"]["ad_gs"]["scenes"]["scene-0230"]
    assert scene["state"] == "executable"
    assert scene["runtime_ready"] is True
    assert scene["checkpoint"]["all_files_exact"] is True
    assert scene["checkpoint"]["evidence_exact"] is True
