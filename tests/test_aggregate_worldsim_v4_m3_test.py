from __future__ import annotations

import json
from pathlib import Path
import subprocess

import yaml

from motion_proj.worldsim_v4.test_freeze import (
    canonical_json_bytes,
    sha256_file,
)
from scripts.aggregate_worldsim_v4_m3_test import aggregate
from scripts.build_worldsim_v4_test_freeze import build_freeze
from scripts.run_worldsim_v4_m3_test_exact_once import (
    attempt_payload,
    completion_payload,
)


TASK_ID = "WS-V4-M3-TEMPORAL-DELTA-01"


def dump_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sequence(operation: str, arm: str, candidate: bool) -> dict:
    return {
        "operation": operation,
        "arm": arm,
        "operation_success": True,
        "warp_l1_delta": 0.05 if candidate else 0.10,
        "temporal_lpips": 0.18 if candidate else 0.20,
        "identity_switch": 0,
        "semantic_reintroduction_pixels": 0,
        "rollback_exact": True,
        "non_target_psnr": 30.1 if candidate else 30.0,
        "non_target_ssim": 0.901 if candidate else 0.900,
        "non_target_lpips_alex": 0.099 if candidate else 0.100,
    }


def test_exact_once_test_aggregate_retains_seventeen_abstains(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    scenes = [f"scene-{index:04d}" for index in range(18)]
    selected = {
        "control_point_count": 4,
        "acceleration_regularization": 0.1,
        "evidence_retention": 0.5,
        "warp_blend_alpha": 0.4,
    }
    operations = ["REMOVE", "LATERAL", "INSERT"]
    config = {
        "execution_gate": {
            "m2_frozen_router": {"weight_name": "uncertainty_forward", "threshold": 1.0},
            "m2_frozen_matched_non_router": "TELEA",
        },
        "clip": {
            "operations": operations,
            "camera_ids": [0, 1, 2],
            "sample_protocol": "seven_consecutive_nuscenes_keyframes",
        },
        "trajectory": {"selected_parameters": selected},
        "gates": {
            "temporal_error_relative_improvement_min": 0.1,
            "identity_switch_relative_reduction_min": 0.25,
            "allow_identity_zero_remains_zero": True,
            "deleted_semantic_reintroduction": 0,
            "rollback_exact_fraction": 1.0,
        },
        "operations": {"minimum_rendered_effect_pixels": 16},
        "ablations": {"arms": ["FRAME_INDEPENDENT", "FULL_WARP_REGULARIZED"]},
        "metrics": ["warp_l1", "temporal_lpips", "rollback_exact"],
        "test_protocol": {"scene_order": scenes},
    }
    inventory = {
        "task_id": TASK_ID,
        "schema_version": "worldsim_v4_m3_test_asset_inventory_v1",
        "scene_order": scenes,
        "scenes": {
            scene: {
                "partition": "test",
                "status": "ready" if index == 0 else "abstain",
            }
            for index, scene in enumerate(scenes)
        },
        "drivestudio_python": "/env/bin/python",
        "test_quality_read": False,
    }
    cohort = {"freeze": {"scene_roles": {"test": scenes}}}
    p0 = {"baselines": {"streetgs": {}}}
    metrics = {
        "statistics": {
            "unit": "scene",
            "bootstrap_seed": 40117,
            "bootstrap_samples": 10000,
        }
    }
    paths = {}
    for name, value in (
        ("config.yaml", config),
        ("inventory.yaml", inventory),
        ("cohort.yaml", cohort),
        ("p0.yaml", p0),
        ("metrics.yaml", metrics),
    ):
        paths[name] = root / name
        dump_yaml(paths[name], value)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "source"], check=True)
    validation = tmp_path / "validation"
    validation.mkdir()
    dump_json(validation / "manifest.json", {"status": "done"})
    dump_json(
        validation / "summary.json",
        {
            "validation_gate_passed": True,
            "test_freeze_authorized": True,
            "test_quality_read": False,
            "selected_parameters": selected,
        },
    )
    dump_json(
        validation / "status.json",
        {
            "summary_sha256": sha256_file(validation / "summary.json"),
            "manifest_sha256": sha256_file(validation / "manifest.json"),
        },
    )
    freeze = build_freeze(
        project_root=root,
        config_path=paths["config.yaml"],
        inventory_path=paths["inventory.yaml"],
        cohort_path=paths["cohort.yaml"],
        validation_run=validation,
        p0_scope_path=paths["p0.yaml"],
        metrics_path=paths["metrics.yaml"],
        run_root=tmp_path / "runs",
        stamp="20260813T000000Z",
        first_run_id=317,
    )
    freeze_path = root / "V4_TEST_FREEZE.json"
    freeze_path.write_bytes(canonical_json_bytes(freeze))
    subprocess.run(["git", "-C", str(root), "add", "V4_TEST_FREEZE.json"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "freeze"], check=True)
    freeze_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    provenance = {
        "freeze_sha256": sha256_file(freeze_path),
        "freeze_commit": freeze_commit,
        "source_commit": freeze["source_commit"],
        "relative_path": "V4_TEST_FREEZE.json",
    }
    ledger = Path(freeze["ledger_dir"])
    (ledger / "attempts").mkdir(parents=True)
    (ledger / "completions").mkdir()
    completion_hashes = {}
    for index, planned in enumerate(freeze["execution_plan"]):
        run = Path(planned["run_dir"])
        run.mkdir(parents=True)
        is_ready = index == 0
        summary = {
            "task_id": TASK_ID,
            "scene": planned["scene"],
            "partition": "test",
            "status": "done" if is_ready else "abstain",
            "reason": None if is_ready else "ABSTAIN_NO_ACTOR",
            "test_scene_attempted": True,
            "test_quality_read": is_ready,
            "test_attempt": {
                "attempt_id": planned["attempt_id"],
                "freeze_commit": freeze_commit,
            },
            "project_git_head": freeze_commit,
            "project_git_dirty": False,
        }
        if is_ready:
            summary.update(
                parameters=selected,
                operations=operations,
                checkpoint_immutable=True,
                rollback_exact=True,
                development_optimization_read=False,
                validation_optimization_read=False,
                sequences=[
                    sequence(operation, arm, arm == "FULL_WARP_REGULARIZED")
                    for operation in operations
                    for arm in ("FRAME_INDEPENDENT", "FULL_WARP_REGULARIZED")
                ],
            )
            dump_json(run / "test_read_started.json", {"state": "started"})
        dump_json(run / "summary.json", summary)
        dump_json(run / "manifest.json", {"files": {}})
        dump_json(run / "fingerprint.json", {"freeze": provenance["freeze_sha256"]})
        dump_json(
            run / "status.json",
            {
                "task_id": TASK_ID,
                "scene": planned["scene"],
                "status": "done",
                "summary_sha256": sha256_file(run / "summary.json"),
                "manifest_sha256": sha256_file(run / "manifest.json"),
                "fingerprint_sha256": sha256_file(run / "fingerprint.json"),
                "test_quality_read": is_ready,
            },
        )
        attempt_path = ledger / "attempts" / f"{planned['attempt_id']}.json"
        dump_json(attempt_path, attempt_payload(planned, provenance))
        completion = completion_payload(
            freeze=freeze,
            provenance=provenance,
            planned=planned,
            inventory=inventory,
        )
        completion_path = ledger / "completions" / f"{planned['attempt_id']}.json"
        dump_json(completion_path, completion)
        completion_hashes[planned["scene"]] = sha256_file(completion_path)
    dump_json(
        ledger / "terminal.json",
        {
            "state": "done",
            "attempt_count": 18,
            "completion_count": 18,
            "freeze_sha256": provenance["freeze_sha256"],
            "scene_order": scenes,
            "completion_sha256": completion_hashes,
        },
    )
    result = aggregate(freeze_path, root, tmp_path / "aggregate")
    assert result["scene_denominator"] == 18
    assert result["evaluable_scene_count"] == 1
    assert result["abstain_scene_count"] == 17
    assert result["scene_operation_denominator"] == 3
    assert result["aggregate"]["warp_l1_relative_improvement"] == 0.5
    assert result["test_gate_passed"] is True
    assert result["conclusion"] == "confirmed"
    assert result["scene_statistics"]["candidate_warp_l1_delta"]["attempted"] == 18
    assert result["paired_statistics"]["warp_l1"]["paired"] == 1
