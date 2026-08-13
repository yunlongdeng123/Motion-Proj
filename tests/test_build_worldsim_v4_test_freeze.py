from __future__ import annotations

import json
from pathlib import Path
import subprocess

import yaml

from motion_proj.worldsim_v4.test_freeze import sha256_file, validate_execution_plan
from scripts.build_worldsim_v4_test_freeze import build_freeze


def dump_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_build_freeze_contains_all_immutable_fields_and_exact_plan(tmp_path: Path) -> None:
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
    config = {
        "execution_gate": {
            "m2_frozen_router": {"weight_name": "uncertainty_forward", "threshold": 1.0},
            "m2_frozen_matched_non_router": "TELEA",
        },
        "clip": {
            "operations": ["REMOVE", "LATERAL", "INSERT"],
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
        "task_id": "WS-V4-M3-TEMPORAL-DELTA-01",
        "schema_version": "worldsim_v4_m3_test_asset_inventory_v1",
        "scene_order": scenes,
        "scenes": {scene: {"partition": "test"} for scene in scenes},
        "drivestudio_python": "/env/bin/python",
        "test_quality_read": False,
    }
    cohort = {"freeze": {"scene_roles": {"test": scenes}}}
    p0 = {"baselines": {"v33_frozen": {}, "streetgs": {}}}
    metrics = {"statistics": {"unit": "scene", "bootstrap_seed": 40117}}
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
            "status": "done",
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
    assert freeze["source_commit"] == subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert freeze["method_selection"]["m3_parameters"] == selected
    assert freeze["thresholds"]["minimum_rendered_effect_pixels"] == 16
    assert freeze["baseline_list"]["test_comparison"] == [
        "FRAME_INDEPENDENT",
        "FULL_WARP_REGULARIZED",
    ]
    assert freeze["metrics_list"] == config["metrics"]
    assert freeze["resources"] == {
        "required_gpu": "NVIDIA GeForce RTX 3090",
        "minimum_disk_free_gib": 20,
        "maximum_gpu_used_at_attempt_start_mib": 2048,
    }
    plan = validate_execution_plan(freeze)
    assert plan[0]["attempt_id"].endswith("r317")
    assert plan[-1]["attempt_id"].endswith("r334")
