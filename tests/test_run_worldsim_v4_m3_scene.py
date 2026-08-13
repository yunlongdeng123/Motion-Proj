from __future__ import annotations

import importlib.util
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_worldsim_v4_m3_scene",
    ROOT / "scripts/run_worldsim_v4_m3_scene.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
SWEEP_SPEC = importlib.util.spec_from_file_location(
    "sweep_worldsim_v4_m3_warp",
    ROOT / "scripts/sweep_worldsim_v4_m3_warp.py",
)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SWEEP_SPEC)
SWEEP_SPEC.loader.exec_module(SWEEP)
AGG_SPEC = importlib.util.spec_from_file_location(
    "aggregate_worldsim_v4_m3_validation",
    ROOT / "scripts/aggregate_worldsim_v4_m3_validation.py",
)
assert AGG_SPEC is not None and AGG_SPEC.loader is not None
AGG = importlib.util.module_from_spec(AGG_SPEC)
AGG_SPEC.loader.exec_module(AGG)


def test_m3_inventory_freezes_six_plus_six_without_test() -> None:
    inventory = yaml.safe_load(
        (ROOT / "configs/worldsim_v4/m3_scene_inventory_v1.yaml").read_text()
    )
    scenes = inventory["scenes"]
    assert sum(row["partition"] == "development" for row in scenes.values()) == 6
    assert sum(row["partition"] == "validation" for row in scenes.values()) == 6
    assert sum(
        row["partition"] == "development" and row["status"] == "ready"
        for row in scenes.values()
    ) == 2
    assert sum(
        row["partition"] == "validation" and row["status"] == "ready"
        for row in scenes.values()
    ) == 3
    assert all(row["partition"] != "test" for row in scenes.values())


def test_scene_contract_rejects_test_partition() -> None:
    config = {"clip": {"camera_ids": [0, 1, 2]}}
    inventory = {
        "camera_ids": [0, 1, 2],
        "scenes": {
            "scene-x": {
                "partition": "test",
                "clip": {"start_index": 0, "end_index": 6, "duration_s": 3.0},
                "instance_token": "token",
            }
        },
    }
    cohort = {"freeze": {"scene_records": []}}
    with pytest.raises(RUNNER.M3SceneRunError, match="test partition"):
        RUNNER.validate_scene_contract(
            config=config,
            inventory=inventory,
            cohort=cohort,
            scene="scene-x",
        )


def test_effect_mask_and_boundary_f1_known_values() -> None:
    left = np.zeros((16, 16, 3), dtype=np.uint8)
    right = left.copy()
    right[4:8, 5:9] = 255
    mask = RUNNER.effect_mask(left, right)
    assert int(mask.sum()) == 16
    assert RUNNER.boundary_f1(mask, mask) == pytest.approx(1.0)
    assert RUNNER.masked_psnr(left, left, np.ones((16, 16), bool)) == 99.0


def test_zero_alpha_warp_variant_is_evidence_exact(tmp_path: Path) -> None:
    base_dir = tmp_path / "BASE"
    evidence_dir = tmp_path / "EVIDENCE"
    output_dir = tmp_path / "FULL"
    for root in (base_dir, evidence_dir):
        (root / "rgb").mkdir(parents=True)
    frames = [0, 5, 10, 15]
    for frame in frames:
        base = np.full((32, 48, 3), frame, dtype=np.uint8)
        evidence = base.copy()
        evidence[8:16, 12:20] = 200
        imageio.imwrite(RUNNER.rgb_path(base_dir, frame, 0), base)
        imageio.imwrite(RUNNER.rgb_path(evidence_dir, frame, 0), evidence)
    RUNNER.build_full_warp_variant(
        base_dir=base_dir,
        evidence_dir=evidence_dir,
        output_dir=output_dir,
        frames=frames,
        cameras=[0],
        alpha=0.0,
    )
    for frame in frames:
        assert np.array_equal(
            imageio.imread(RUNNER.rgb_path(output_dir, frame, 0)),
            imageio.imread(RUNNER.rgb_path(evidence_dir, frame, 0)),
        )
    warp_l1, success, by_frame = SWEEP.candidate_warp_l1(
        base_dir=base_dir,
        remove_dir=base_dir,
        candidate_dir=output_dir,
        frames=frames,
        cameras=[0],
        minimum_effect_pixels=1,
    )
    assert warp_l1 > 0.0
    assert success
    assert set(by_frame) == set(frames)


def test_validation_aggregate_retains_abstains_and_remove_zero_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = {
        "control_point_count": 4,
        "acceleration_regularization": 0.1,
        "evidence_retention": 0.5,
        "warp_blend_alpha": 0.4,
    }

    def write_binding(path: Path, payload: dict) -> dict:
        path.mkdir(parents=True)
        RUNNER.atomic_json(path / "summary.json", payload)
        RUNNER.atomic_json(path / "manifest.json", {"files": []})
        return {
            "path": str(path),
            "summary_sha256": RUNNER.sha256_file(path / "summary.json"),
            "manifest_sha256": RUNNER.sha256_file(path / "manifest.json"),
        }

    m3_path = tmp_path / "m3.yaml"
    m3_path.write_text(
        yaml.safe_dump({"trajectory": {"selected_parameters": selected}})
    )
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text(
        yaml.safe_dump(
            {
                "scenes": {
                    f"scene-{index}": {"partition": "validation"}
                    for index in range(6)
                }
            }
        )
    )
    development = write_binding(
        tmp_path / "development", {"selected_parameters": selected, "test_quality_read": False}
    )
    scene_runs = {}
    for index in range(6):
        if index >= 3:
            payload = {
                "scene": f"scene-{index}",
                "partition": "validation",
                "status": "abstain",
                "reason": "unavailable",
                "project_git_dirty": False,
                "development_content_read": False,
                "development_optimization_read": False,
                "validation_optimization_read": False,
                "test_quality_read": False,
            }
        else:
            sequences = []
            for operation in ("REMOVE", "LATERAL", "INSERT"):
                for arm in (
                    "FRAME_INDEPENDENT",
                    "LINEAR",
                    "CUBIC_BSPLINE",
                    "CUBIC_BSPLINE_TEMPORAL_EVIDENCE",
                    "FULL_WARP_REGULARIZED",
                ):
                    candidate = arm == "FULL_WARP_REGULARIZED"
                    remove = operation == "REMOVE"
                    sequences.append(
                        {
                            "operation": operation,
                            "arm": arm,
                            "operation_success": True,
                            "warp_l1_delta": 1.0 if remove or not candidate else 0.5,
                            "temporal_lpips": 1.0 if remove or not candidate else 0.8,
                            "identity_switch": 0,
                            "semantic_reintroduction_pixels": 0,
                            "rollback_exact": True,
                            "non_target_psnr": 20.0,
                            "non_target_ssim": 0.8,
                            "non_target_lpips_alex": 0.2,
                        }
                    )
            payload = {
                "scene": f"scene-{index}",
                "partition": "validation",
                "status": "done",
                "parameters": selected,
                "operations": ["REMOVE", "LATERAL", "INSERT"],
                "arms": [
                    "FRAME_INDEPENDENT",
                    "LINEAR",
                    "CUBIC_BSPLINE",
                    "CUBIC_BSPLINE_TEMPORAL_EVIDENCE",
                    "FULL_WARP_REGULARIZED",
                ],
                "checkpoint_immutable": True,
                "rollback_exact": True,
                "sequences": sequences,
                "project_git_dirty": False,
                "development_content_read": False,
                "development_optimization_read": False,
                "validation_optimization_read": False,
                "test_quality_read": False,
            }
        scene_runs[f"scene-{index}"] = write_binding(
            tmp_path / f"run-{index}", payload
        )
    config = {
        "m3_config": {"path": str(m3_path), "sha256": RUNNER.sha256_file(m3_path)},
        "scene_inventory": {
            "path": str(inventory_path),
            "sha256": RUNNER.sha256_file(inventory_path),
        },
        "development_freeze": development,
        "validation_scene_runs": scene_runs,
        "aggregation": {
            "scene_denominator": 6,
            "operations": ["REMOVE", "LATERAL", "INSERT"],
            "baseline_arm": "FRAME_INDEPENDENT",
            "candidate_arm": "FULL_WARP_REGULARIZED",
            "reduction": "equal_scene_operation_mean",
        },
        "gates": {
            "temporal_error_relative_improvement_min": 0.1,
            "identity_switch_relative_reduction_min": 0.25,
            "allow_identity_zero_remains_zero": True,
            "deleted_semantic_reintroduction": 0,
            "rollback_exact_fraction": 1.0,
        },
    }
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(AGG, "git_dirty", lambda: False)
    monkeypatch.setattr(AGG, "git_head", lambda: "a" * 40)
    summary = AGG.run(config_path=config_path, run_dir=tmp_path / "aggregate")
    assert summary["scene_denominator"] == 6
    assert summary["evaluable_scene_count"] == 3
    assert summary["abstain_scene_count"] == 3
    assert summary["scene_operation_denominator"] == 9
    # REMOVE 的零增益仍占九个 paired rows，因此 warp 改善为 1/3。
    assert summary["aggregate"]["warp_l1_relative_improvement"] == pytest.approx(1 / 3)
    assert summary["validation_gate_passed"]
