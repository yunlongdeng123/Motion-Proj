from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.feature_evaluation import (
    actor_feature_metrics,
    deterministic_actor_pairs,
    evaluate_h_gate,
    reproject_feature_arms,
    repeatability_against_aggregate,
    row_cosine,
    single_view_gaussian_feature,
)
from scripts.run_worldsim_v51_h_evaluation import _load_evaluation_config, validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_b_h_evaluation_v2.yaml"


def test_row_cosine_uses_explicit_nonzero_denominator() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    right = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, -1.0]])
    values, valid = row_cosine(left, right, epsilon=1e-8)
    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_allclose(values[valid], [1.0, 0.0], atol=1e-15)
    assert np.isnan(values[1])


def test_actor_pairs_are_unique_bounded_and_seeded() -> None:
    small = deterministic_actor_pairs(np.asarray([4, 1, 9]), seed=7, maximum_pairs=8)
    np.testing.assert_array_equal(small, [[1, 4], [1, 9], [4, 9]])
    indices = np.arange(100, dtype=np.int64)
    first = deterministic_actor_pairs(indices, seed=20260814, maximum_pairs=256)
    second = deterministic_actor_pairs(indices, seed=20260814, maximum_pairs=256)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (256, 2)
    assert np.unique(first, axis=0).shape[0] == 256
    assert np.all(first[:, 0] < first[:, 1])


def test_single_view_lift_repeatability_and_common_reprojection() -> None:
    patch_grid = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    lifted = single_view_gaussian_feature(
        gaussian_id=np.asarray([0, 0, 1, 1, 2]),
        pixel_id=np.asarray([0, 2, 1, 3, 0]),
        contribution_weight=np.asarray([1.0, 1.0, 1.0, 1.0, 1e-5]),
        patch_grid=patch_grid,
        gaussian_count=3,
        image_height=2,
        image_width=2,
        minimum_intersection_contribution=1e-4,
        minimum_gaussian_view_mass=1e-3,
        epsilon=1e-8,
    )
    np.testing.assert_array_equal(lifted["gaussian_id"], [0, 1])
    np.testing.assert_allclose(lifted["feature"], [[1.0, 0.0], [0.0, 1.0]], atol=1e-7)
    repeat = repeatability_against_aggregate(
        aggregate_feature=np.asarray([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]),
        aggregate_covered=np.asarray([True, True, False]),
        view_gaussian_id=lifted["gaussian_id"],
        view_feature=lifted["feature"],
        epsilon=1e-8,
    )
    assert repeat["valid_cosine_count"] == 2
    assert repeat["mean_cosine"] == 1.0

    reprojection = reproject_feature_arms(
        features_by_arm={
            "B0": np.asarray([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]),
            "B1": np.asarray([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]),
        },
        common_covered=np.asarray([True, True, False]),
        gaussian_id=np.asarray([0, 0, 1, 1, 2]),
        pixel_id=np.asarray([0, 2, 1, 3, 0]),
        contribution_weight=np.asarray([1.0, 1.0, 1.0, 1.0, 100.0]),
        patch_grid=patch_grid,
        image_height=2,
        image_width=2,
        minimum_intersection_contribution=1e-4,
        minimum_pixel_mass=1e-3,
        cosine_epsilon=1e-8,
    )
    assert reprojection["supported_pixel_count"] == 4
    assert reprojection["valid_cosine_count"] == 4
    assert reprojection["B0_mean_cosine"] == 1.0
    assert reprojection["B1_mean_cosine"] == 1.0
    assert reprojection["B1_minus_B0"] == 0.0


def test_actor_metrics_use_active_model_membership_and_nearest_background() -> None:
    feature = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.1],
            [1.0, 0.1],
            [1.0, 0.1],
            [1.0, 0.1],
        ],
        dtype=np.float32,
    )
    report = actor_feature_metrics(
        feature=feature,
        covered=np.ones(6, dtype=bool),
        background_count=2,
        rigid_actor_id=np.zeros(4, dtype=np.int64),
        active_actor=np.asarray([True]),
        background_world_position=np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        rigid_world_position=np.asarray(
            [[10.0, 0.0, 0.1], [10.0, 0.0, 0.2], [10.0, 0.0, 0.3], [10.0, 0.0, 0.4]]
        ),
        scene="scene-test",
        seed=20260814,
        minimum_actor_gaussians=4,
        maximum_pairs_per_actor=6,
        cosine_epsilon=1e-8,
    )
    assert report["eligible_actor_count"] == 1
    actor = report["actor_reports"][0]
    assert actor["pair_count"] == 6
    assert actor["same_actor_cosine"] > actor["actor_background_cosine"]
    assert report["scene_margin"] > 0.0


def test_h_gate_is_scene_balanced_and_fail_closed() -> None:
    scenes = [
        {
            "evaluable": True,
            "actor_metrics": {"B1": {"scene_margin": margin}},
            "coverage": {"rigid": coverage},
            "heldout_reprojection": {"scene_B1_minus_B0": delta},
        }
        for margin, coverage, delta in ((0.2, 0.7, 0.0), (0.1, 0.6, -0.01), (-0.1, 0.8, 0.0))
    ]
    gate = {
        "scene_count": 3,
        "minimum_evaluable_scenes": 2,
        "minimum_positive_b1_margin_scenes": 2,
        "minimum_scene_balanced_b1_margin_exclusive": 0.0,
        "minimum_scene_balanced_rigid_coverage": 0.60,
        "minimum_scene_balanced_heldout_b1_minus_b0": -0.01,
    }
    result = evaluate_h_gate(scenes, gate)
    assert result["pass"] is True
    assert result["positive_b1_margin_scene_count"] == 2
    scenes[1]["heldout_reprojection"]["scene_B1_minus_B0"] = -0.1
    failed = evaluate_h_gate(scenes, gate)
    assert failed["pass"] is False
    assert failed["checks"]["scene_balanced_heldout_non_degradation"] is False


def test_h_evaluation_config_binds_proxy_split_gate_and_locks() -> None:
    config, evidence, evaluation, uplift = validate_config(CONFIG)
    assert len(evidence) == 45
    assert len(evaluation) == 45
    assert len(uplift) == 6
    assert config["evaluation"]["membership_declaration"] == (
        "model_membership_proxy_not_ground_truth"
    )
    assert config["evaluation"]["minimum_actor_covered_gaussians"] == 32
    assert config["evaluation"]["maximum_pairs_per_actor"] == 4096
    assert config["h_gate"]["minimum_evaluable_scenes"] == 2
    assert config["h_gate"]["minimum_positive_b1_margin_scenes"] == 2
    assert config["h_gate"]["minimum_scene_balanced_rigid_coverage"] == 0.60
    assert config["h_gate"]["minimum_scene_balanced_heldout_b1_minus_b0"] == -0.01
    assert config["locks"]["membership_proxy_read_evaluation_only"] is True
    assert config["locks"]["proxy_as_method_input"] is False
    assert config["locks"]["pca_fit"] is False
    assert config["locks"]["uplift_recompute"] is False
    assert config["locks"]["screening_quality_read"] is False
    assert config["locks"]["confirmation_quality_read"] is False
    assert config["locks"]["final_heldout_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
    assert config["resources"]["maximum_nvidia_peak_mib"] == 24000
    assert config["resources"]["maximum_torch_reserved_peak_mib"] == 24000
    assert config["recovery"]["reuse_blocked_outputs"] is False


def test_h_evaluation_recovery_changes_only_two_resource_ceilings_and_metadata() -> None:
    base = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/stage_b_h_evaluation_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    merged = _load_evaluation_config(CONFIG)
    assert merged["resources"]["maximum_nvidia_peak_mib"] == 24000
    assert merged["resources"]["maximum_torch_reserved_peak_mib"] == 24000
    merged["resources"]["maximum_nvidia_peak_mib"] = 22528
    merged["resources"]["maximum_torch_reserved_peak_mib"] = 22528
    merged.pop("recovery")
    merged["failure_ledger_refs"] = base["failure_ledger_refs"]
    merged["failure_ledger_delta"] = base["failure_ledger_delta"]
    assert merged == base
