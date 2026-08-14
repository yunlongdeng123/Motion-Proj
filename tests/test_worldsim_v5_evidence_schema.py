from pathlib import Path

import numpy as np
import pytest
import yaml

from motion_proj.worldsim_v5.bayesian_unary import (
    accumulate_unary_arm_statistics,
    accumulate_effective_count_statistics,
    effective_count_unary,
    empty_effective_count_statistics,
    empty_unary_arm_statistics,
    finalize_effective_count_unary,
    finalize_unary_arms,
    observation_reliability,
)
from motion_proj.worldsim_v5.evidence_schema import (
    atomic_save_npz,
    sha256_file,
    validate_edge_table,
    validate_gaussian_table,
    validate_observation_chunk,
)


ROOT = Path(__file__).resolve().parents[1]


def _observations() -> dict[str, np.ndarray]:
    count = 4
    return {
        "scene": np.asarray("scene-0001"),
        "role": np.asarray("high_support"),
        "sam_probability_source": np.asarray("sigmoid_sam2_logit"),
        "gaussian_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "view_id": np.arange(count, dtype=np.int32),
        "frame_id": np.asarray([2, 7, 2, 7], dtype=np.int32),
        "camera_id": np.asarray([0, 1, 0, 1], dtype=np.int8),
        "projected_pixel": np.asarray([[10, 20], [11, 20], [4, 5], [5, 5]], dtype=np.float32),
        "visibility": np.asarray([1.0, 0.8, 1.0, 1.0], dtype=np.float32),
        "sam_probability": np.asarray([0.95, 0.9, 0.8, 0.2], dtype=np.float32),
        "sam_logit": np.asarray([2.94, 2.20, 1.39, -1.39], dtype=np.float32),
        "sam_probability_available": np.ones(count, dtype=np.int8),
        "mask_quality_accepted": np.ones(count, dtype=np.int8),
        "mask_boundary_distance": np.asarray([8.0, 6.0, 0.1, -0.1], dtype=np.float32),
        "depth_residual": np.asarray([0.1, 0.2, 0.1, 0.1], dtype=np.float32),
        "depth_consistent": np.ones(count, dtype=np.int8),
        "lidar_support": np.asarray([1, 1, 0, 0], dtype=np.int8),
        "lidar_support_available": np.ones(count, dtype=np.int8),
        "view_angle_cosine": np.asarray([1.0, 0.9, 1.0, 1.0], dtype=np.float32),
        "positive_observation": np.asarray([1, 1, 1, 0], dtype=np.int8),
        "negative_observation": np.asarray([0, 0, 0, 1], dtype=np.int8),
        "reliability": np.ones(count, dtype=np.float32),
        "contribution_weight": np.ones(count, dtype=np.float32),
    }


def test_observation_schema_and_effective_count_unary_preserve_disagreement() -> None:
    observations = _observations()
    validate_observation_chunk(observations, gaussian_count=2)
    result = effective_count_unary(
        prior_probability=np.asarray([0.1, 0.1]),
        prior_strength=2.0,
        observations=observations,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    assert result["unary_posterior"][0] > result["unary_posterior"][1]
    assert result["effective_evidence_count"][0] > result["effective_evidence_count"][1]
    assert result["multi_view_disagreement"][1] > result["multi_view_disagreement"][0]
    assert result["boundary_ambiguity"][1] > result["boundary_ambiguity"][0]


def test_depth_inconsistent_observation_has_zero_reliability() -> None:
    observations = _observations()
    observations["depth_consistent"] = np.asarray([0, 1, 1, 1], dtype=np.int8)
    reliability = observation_reliability(
        observations,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    assert reliability[0] == 0.0
    assert np.all(reliability[1:] > 0.0)


def test_streaming_effective_count_matches_batch() -> None:
    observations = _observations()
    batch = effective_count_unary(
        prior_probability=np.asarray([0.1, 0.1]),
        prior_strength=2.0,
        observations=observations,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    statistics = empty_effective_count_statistics(2)
    for indices in (np.asarray([0, 1]), np.asarray([2, 3])):
        chunk = {
            name: value if np.asarray(value).shape == () else np.asarray(value)[indices]
            for name, value in observations.items()
        }
        accumulate_effective_count_statistics(
            statistics,
            observations=chunk,
            gaussian_count=2,
            sam_confidence_floor=0.1,
            boundary_distance_scale_px=4.0,
            depth_residual_scale_m=0.5,
        )
    streamed = finalize_effective_count_unary(
        prior_probability=np.asarray([0.1, 0.1]),
        prior_strength=2.0,
        statistics=statistics,
    )
    for name in streamed:
        assert np.allclose(streamed[name], batch[name])


def test_b0_b1_b3_are_mechanistically_distinct_and_streamable() -> None:
    observations = _observations()
    statistics = empty_unary_arm_statistics(2)
    weights = accumulate_unary_arm_statistics(
        statistics,
        observations=observations,
        gaussian_count=2,
        sam_confidence_floor=0.1,
        boundary_distance_scale_px=4.0,
        depth_residual_scale_m=0.5,
    )
    arms = finalize_unary_arms(
        prior_probability=np.asarray([0.1, 0.1]),
        prior_strength=2.0,
        statistics=statistics,
    )
    assert tuple(arms) == ("B0", "B1", "B3")
    assert np.all(weights["B0"] == 1.0)
    assert np.all(weights["B1"] < weights["B0"])
    assert np.allclose(weights["B1"], weights["B3"])
    assert not np.allclose(
        arms["B0"]["unary_posterior"], arms["B1"]["unary_posterior"]
    )
    assert not np.allclose(
        arms["B1"]["unary_posterior"], arms["B3"]["unary_posterior"]
    )

    streamed_statistics = empty_unary_arm_statistics(2)
    for indices in (np.asarray([0, 2]), np.asarray([1, 3])):
        chunk = {
            name: value if np.asarray(value).shape == () else np.asarray(value)[indices]
            for name, value in observations.items()
        }
        accumulate_unary_arm_statistics(
            streamed_statistics,
            observations=chunk,
            gaussian_count=2,
            sam_confidence_floor=0.1,
            boundary_distance_scale_px=4.0,
            depth_residual_scale_m=0.5,
        )
    streamed = finalize_unary_arms(
        prior_probability=np.asarray([0.1, 0.1]),
        prior_strength=2.0,
        statistics=streamed_statistics,
    )
    for arm in arms:
        for name in arms[arm]:
            assert np.allclose(arms[arm][name], streamed[arm][name])


def test_structured_npz_is_byte_stable_and_validates_geometry(tmp_path: Path) -> None:
    table = {
        "scene": np.asarray("scene-0001"),
        "role": np.asarray("high_support"),
        "gaussian_id": np.arange(2, dtype=np.int64),
        "base_model": np.asarray([0, 1], dtype=np.int8),
        "base_index": np.asarray([0, 0], dtype=np.int64),
        "center": np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "covariance": np.stack((np.eye(3), np.eye(3))).astype(np.float32),
        "normal_proxy": np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
        "normal_available": np.ones(2, dtype=np.int8),
        "prior": np.asarray([0.1, 0.1], dtype=np.float32),
        "unary_posterior": np.asarray([0.8, 0.2], dtype=np.float32),
        "unary_uncertainty": np.asarray([0.2, 0.3], dtype=np.float32),
        "effective_evidence_count": np.asarray([2.0, 1.0], dtype=np.float32),
        "multi_view_disagreement": np.asarray([0.0, 0.2], dtype=np.float32),
        "boundary_ambiguity": np.asarray([0.1, 0.9], dtype=np.float32),
        "depth_support": np.asarray([1.0, 0.5], dtype=np.float32),
        "lidar_support": np.asarray([1.0, 0.0], dtype=np.float32),
        "lidar_support_available": np.ones(2, dtype=np.int8),
        "motion_consistency": np.asarray([1.0, 1.0], dtype=np.float32),
        "motion_consistency_available": np.ones(2, dtype=np.int8),
    }
    validate_gaussian_table(table)
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    atomic_save_npz(first, table)
    atomic_save_npz(second, table)
    assert sha256_file(first) == sha256_file(second)
    broken = dict(table)
    broken_covariance = np.stack((np.eye(3), np.eye(3))).astype(np.float32)
    broken_covariance[0, 0, 1] = 1.0
    broken["covariance"] = broken_covariance
    with pytest.raises(ValueError, match="对称"):
        validate_gaussian_table(broken)


def test_edge_schema_rejects_cross_table_index_drift() -> None:
    edges = {
        "scene": np.asarray("scene-0001"),
        "role": np.asarray("high_support"),
        "source_gaussian_id": np.asarray([0], dtype=np.int64),
        "target_gaussian_id": np.asarray([1], dtype=np.int64),
        "mahalanobis_distance": np.asarray([1.0], dtype=np.float32),
        "normal_distance": np.asarray([0.0], dtype=np.float32),
        "motion_distance": np.asarray([0.0], dtype=np.float32),
        "boundary_barrier": np.asarray([1.0], dtype=np.float32),
        "edge_affinity": np.asarray([0.5], dtype=np.float32),
    }
    validate_edge_table(edges, gaussian_count=2)
    edges["target_gaussian_id"] = np.asarray([2], dtype=np.int64)
    with pytest.raises(ValueError, match="越界"):
        validate_edge_table(edges, gaussian_count=2)


def test_m1_contract_binds_processed_and_derived_sky_masks() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v5/m1_structured_ownership_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "running"
    assert config["phase"] == "structured_unary_diagnostic_complete_graph_protocol_pending"
    assert len(config["fresh_cohort_binding"]["development_scenes"]) == 8
    assert config["data_readiness"]["processed_scene_count"] == 8
    assert config["data_readiness"]["processed_total_bytes"] == 2497238886
    assert config["data_readiness"]["processed_frame_contract"]["scene-0379"] == 191
    assert config["data_readiness"]["processed_frame_contract"]["scene-0535"] == 201
    assert (
        config["data_readiness"]["formal_preprocess"]["summary_sha256"]
        == "dcdd3450328669c26eed0316e2088e1f501fad965ed10ad8d344c37fda36f9c0"
    )
    assert (
        config["data_readiness"]["state"]
        == "formal30k_complete_scene0471_unary_complete_graph_protocol_pending"
    )
    sky = config["data_readiness"]["derived_sky_masks"]
    assert sky["required_count"] == sky["present_count"] == 4704
    assert sky["all_payload_rehashed_exact"] is True
    assert len(sky["scene_summary_sha256"]) == 8
    assert (
        sky["training_binding_config"]
        == "configs/worldsim_v5/m1_development_reconstruction_skybound_v1.yaml"
    )
    assert config["data_readiness"]["blocked_base_profile"]["status"] == "blocked"
    profile = config["data_readiness"]["profile100_gate"]
    assert profile["status"] == "done"
    assert profile["scene_count"] == 8
    assert profile["iteration_count_each"] == 100
    assert profile["all_checkpoint_payload_rehashed_exact"] is True
    formal = config["data_readiness"]["formal30k_gate"]
    assert formal["status"] == "done"
    assert formal["scene_count"] == formal["completed_scene_count"] == 8
    assert formal["iteration_count_each"] == 30000
    assert formal["all_checkpoint_payload_rehashed_exact"] is True
    sam = config["data_readiness"]["scene0471_sam_diagnostic"]
    assert sam["status"] == "done"
    assert sam["view_count"] == 30
    assert sam["accepted_view_count"] == 18
    assert sam["heldout_quality_read"] is False
    unary = config["data_readiness"]["scene0471_unary_diagnostic"]
    assert unary["status"] == "done"
    assert unary["gaussian_count"] == 859613
    assert unary["accepted_evaluation_view_count"] == 8
    assert unary["abstained_evaluation_view_count"] == 7
    assert unary["checkpoint_sha256_before"] == unary["checkpoint_sha256_after"]
    assert unary["evaluation_delta_vs_b0"]["B1"]["boundary_f1"] > 0.10
    assert unary["evaluation_delta_vs_b0"]["B1"]["false_negative_semantic_mass"] > 0.09
    assert unary["arm_selected"] is False
    assert unary["graph_inference_started"] is False
    assert unary["parameter_search_performed"] is False
    assert config["data_readiness"]["development_raw_present_keyframes"] == 0
    assert (
        config["data_readiness"]["exact_preprocess_raw_contract"]["present_files"]
        == 14220
    )
    assert config["graph"]["status"] == "locked_pending_preregistered_graph_protocol"
    assert config["unary_development_arms"]["B0"].startswith("hard_unweighted")
    assert config["unary_development_arms"]["B1"].startswith(
        "reliability_weighted_hard"
    )
    assert config["unary_development_arms"]["B3"].startswith(
        "reliability_weighted_soft_sam"
    )
    assert config["graph"]["transformer_allowed"] is False
    assert config["restrictions"]["development_content_read"] is True
    assert config["restrictions"]["training_started"] is True
    assert config["restrictions"]["training_iteration_started"] is True
    assert config["restrictions"]["segmentation_inference_started"] is True
    assert config["restrictions"]["method_inference_started"] is True
    assert config["restrictions"]["validation_content_read"] is False
    assert config["restrictions"]["test_quality_read"] is False
