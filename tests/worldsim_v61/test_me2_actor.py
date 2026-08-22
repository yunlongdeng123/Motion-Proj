"""ME-2 actor compiler 的固定协议与几何回归测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import yaml

from motion_proj.worldsim_v61.me2_actor import _fixed_sample, actor_state, canonicalize_mesh


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/worldsim_v61/me2_hy3d_actor_v1.yaml"


def test_actor_zero_is_a_valid_me2_target() -> None:
    instances = {
        "0": {
            "class_name": "vehicle.construction",
            "frame_annotations": {
                "frame_idx": [52],
                "obj_to_world": [np.eye(4).tolist()],
                "box_size": [[4.0, 2.0, 2.5]],
            },
        }
    }
    pose, size, class_name = actor_state(instances, 0, 52)
    np.testing.assert_array_equal(pose, np.eye(4))
    np.testing.assert_array_equal(size, np.asarray([4.0, 2.0, 2.5]))
    assert class_name == "vehicle.construction"


def test_fixed_sample_is_deterministic_without_jitter() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    expected = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(_fixed_sample(points, 4), expected)


def test_canonicalization_uses_one_uniform_scale_only(tmp_path: Path) -> None:
    source = trimesh.creation.box(extents=[4.0, 2.0, 1.0])
    source_path = tmp_path / "source.glb"
    source.export(source_path)
    _, _, audit = canonicalize_mesh(
        source_path,
        native_size_lwh=np.asarray([8.0, 2.0, 4.0]),
        surface_count=128,
        seed=1234,
    )
    assert audit["axis_permutation_to_lwh"] == [0, 2, 1]
    assert np.isclose(audit["uniform_scale_m"], 2.0)
    np.testing.assert_allclose(audit["canonical_extent_ratios"], np.ones(3), atol=1e-6)
    assert audit["anisotropic_scale_used"] is False


def test_me2_preregistration_has_no_sweep_or_placeholder() -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    assert "TO_FILL" not in raw
    config = yaml.safe_load(raw)
    assert config["arms"] == ["A0-image", "A1-bbox", "A2-point", "A3-voxel"]
    assert len(config["units"]) == 4
    assert len(config["case_to_unit"]) == 6
    assert set(config["case_to_unit"].values()) == {
        unit["unit_id"] for unit in config["units"]
    }
    assert config["primary_gate"]["arm"] == "A3-voxel"
    assert config["primary_gate"]["minimum_accepted_cases"] == 2
    assert config["primary_gate"]["maximum_false_safe_count"] == 0
    assert config["method_and_eval_gate"]["require_free_space_conflict_count"] == 0
    assert config["method_and_eval_gate"]["require_swept_collision_count"] == 0
    assert config["generation"]["A0-image"]["num_inference_steps"] == 50
    assert config["generation"]["omni_controls"]["num_inference_steps"] == 50
    assert config["generation"]["A0-image"]["octree_resolution"] == 256
    assert config["generation"]["omni_controls"]["octree_resolution"] == 256
    assert "no_texture_prompt_seed_step_or_threshold_sweep" in config["claim_boundary"]


def test_only_exact_truck_trailer_pair_is_filtered() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    relations = config["kinematic_relations"]
    assert relations["policy"] == "exact_pair_only_no_global_collision_threshold_relaxation"
    assert relations["legal_collision_pairs"] == {
        "scene-0242": [[4, 15]],
        "scene-0048": [],
    }
