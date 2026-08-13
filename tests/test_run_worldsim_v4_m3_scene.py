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
