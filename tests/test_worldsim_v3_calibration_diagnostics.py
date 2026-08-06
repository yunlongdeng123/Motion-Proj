from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from motion_proj.worldsim_v3.calibration_diagnostics import (
    affine_matrices_from_state,
    factorized_affine_matrices,
    load_input_speed_contract,
    pose_residuals_from_state,
    rotation_6d_to_matrix,
    scalar_summary,
    speed_tier_masks,
    summarize_affines,
    summarize_pose_residuals,
    validate_diagnostic_contract,
)


def diagnostic_contract() -> dict:
    return {
        "schema_version": 1,
        "diagnostic_version": "A1-D0-v1",
        "task_id": "WS-V3-A1-CALIBRATION-01",
        "frozen_before_diagnostic_result_access": True,
        "scene": {
            "num_frames": 4,
            "num_cameras": 3,
            "camera_id_to_name": {
                0: "CAM_FRONT",
                1: "CAM_FRONT_LEFT",
                2: "CAM_FRONT_RIGHT",
            },
        },
        "speed_tiers": {
            "near_static_upper_mps": 0.5,
            "low_speed_upper_mps": 2.0,
            "frame_alignment": "previous_interval_right_aligned",
        },
        "isp": {
            "camera_pairs": [
                ["CAM_FRONT_LEFT", "CAM_FRONT"],
                ["CAM_FRONT", "CAM_FRONT_RIGHT"],
            ]
        },
    }


def test_validate_diagnostic_contract() -> None:
    validate_diagnostic_contract(diagnostic_contract())
    bad = diagnostic_contract()
    bad["scene"]["camera_id_to_name"][0] = "CAM_FRONT_LEFT"
    with pytest.raises(ValueError, match="camera mapping"):
        validate_diagnostic_contract(bad)


def test_frozen_yaml_contract_loads() -> None:
    path = Path("configs/worldsim_v3/a1_diagnostics_v1.yaml")
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_diagnostic_contract(contract)
    assert contract["speed_tiers"]["expected_input_only_counts"] == {
        "near_static": 2,
        "low_speed": 18,
        "normal": 176,
    }


def test_scalar_summary() -> None:
    summary = scalar_summary(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(1.5)
    assert summary["median"] == pytest.approx(1.5)
    assert summary["p90"] == pytest.approx(2.7)
    assert summary["max"] == pytest.approx(3.0)


def test_rotation_6d_identity() -> None:
    matrix = rotation_6d_to_matrix(
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(matrix[0], torch.eye(3, dtype=torch.float64))


def test_speed_tiers_are_disjoint_and_complete() -> None:
    masks = speed_tier_masks(
        torch.tensor([0.0, 0.5, 1.99, 2.0, 5.0]),
        near_static_upper_mps=0.5,
        low_speed_upper_mps=2.0,
    )
    assert {key: int(value.sum()) for key, value in masks.items()} == {
        "near_static": 1,
        "low_speed": 2,
        "normal": 2,
    }
    assert torch.stack(list(masks.values())).sum(dim=0).eq(1).all()


def test_input_speed_contract_uses_previous_interval(tmp_path: Path) -> None:
    extrinsics = tmp_path / "extrinsics"
    extrinsics.mkdir()
    for frame, x in enumerate((0.0, 0.0, 0.1, 0.4)):
        matrix = np.eye(4)
        matrix[0, 3] = x
        np.savetxt(extrinsics / f"{frame:03d}_0.txt", matrix)
    speed, summary = load_input_speed_contract(
        tmp_path,
        num_frames=4,
        source_camera_id=0,
        processed_hz=10.0,
        near_static_upper_mps=0.5,
        low_speed_upper_mps=2.0,
    )
    torch.testing.assert_close(speed, torch.tensor([0.0, 0.0, 1.0, 3.0], dtype=torch.float64))
    assert summary["tier_frame_counts"] == {
        "near_static": 2,
        "low_speed": 1,
        "normal": 1,
    }


def test_c0_pose_is_identity() -> None:
    translation, rotation, kind = pose_residuals_from_state(
        None,
        variant="c0-off",
        num_frames=4,
        num_cameras=3,
        bounded_translation_max_m=0.15,
        bounded_rotation_max_deg=2.0,
    )
    assert kind == "absent_identity"
    assert torch.count_nonzero(translation) == 0
    torch.testing.assert_close(
        rotation, torch.eye(3, dtype=torch.float64).expand(12, 3, 3)
    )


def test_native_and_bounded_pose_contracts() -> None:
    native = {
        "identity": torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        "embeds.weight": torch.zeros(12, 9),
    }
    native["embeds.weight"][0, 0] = 0.25
    translation, _, kind = pose_residuals_from_state(
        native,
        variant="c1-native",
        num_frames=4,
        num_cameras=3,
        bounded_translation_max_m=0.15,
        bounded_rotation_max_deg=2.0,
    )
    assert kind == "native_unbounded"
    assert translation[0, 0] == pytest.approx(0.25)
    bounded = {"embeds.weight": torch.full((12, 6), 100.0)}
    translation, rotation, kind = pose_residuals_from_state(
        bounded,
        variant="c3-bounded-pose",
        num_frames=4,
        num_cameras=3,
        bounded_translation_max_m=0.15,
        bounded_rotation_max_deg=2.0,
    )
    assert kind == "bounded_axis_angle"
    assert torch.linalg.vector_norm(translation, dim=-1).max() <= 0.15 + 1e-12
    trace = torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1)
    angle = torch.rad2deg(torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0)))
    assert angle.max() <= 2.0 + 1e-9


def test_pose_summary_reports_temporal_and_speed_tiers() -> None:
    translation = torch.zeros(12, 3, dtype=torch.float64)
    translation[3:, 0] = 0.01
    rotation = torch.eye(3, dtype=torch.float64).expand(12, 3, 3).clone()
    summary = summarize_pose_residuals(
        translation,
        rotation,
        frame_speed_mps=torch.tensor([0.0, 1.0, 3.0, 3.0]),
        camera_id_to_name={
            0: "CAM_FRONT",
            1: "CAM_FRONT_LEFT",
            2: "CAM_FRONT_RIGHT",
        },
        near_static_upper_mps=0.5,
        low_speed_upper_mps=2.0,
        minimum_frames_per_tier=1,
    )
    assert summary["by_speed_tier"]["near_static"]["frame_count"] == 1
    assert summary["first_difference"]["translation_delta_norm_m"]["max"] == pytest.approx(
        0.01
    )
    assert summary["second_difference"]["translation_jitter_norm_m"]["max"] == pytest.approx(
        0.01
    )


def zero_factorized_state() -> dict[str, torch.Tensor]:
    return {
        "camera_embedding.weight": torch.zeros(3, 4),
        "time_encoder.0.weight": torch.zeros(8, 5),
        "time_encoder.0.bias": torch.zeros(8),
        "decoder.0.weight": torch.zeros(64, 12),
        "decoder.0.bias": torch.zeros(64),
        "decoder.2.weight": torch.zeros(12, 64),
        "decoder.2.bias": torch.zeros(12),
    }


def test_factorized_zero_state_is_identity() -> None:
    matrices = factorized_affine_matrices(
        zero_factorized_state(), num_frames=4, num_cameras=3
    )
    expected = torch.zeros(12, 3, 4, dtype=torch.float64)
    expected[:, :, :3] = torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(matrices, expected)


def test_c0_affine_summary_is_zero() -> None:
    matrices, kind, auxiliary = affine_matrices_from_state(
        None, variant="c0-off", num_frames=4, num_cameras=3
    )
    assert kind == "absent_identity"
    assert auxiliary is None
    summary = summarize_affines(
        matrices,
        num_frames=4,
        camera_id_to_name={
            0: "CAM_FRONT",
            1: "CAM_FRONT_LEFT",
            2: "CAM_FRONT_RIGHT",
        },
        camera_pairs=[
            ("CAM_FRONT_LEFT", "CAM_FRONT"),
            ("CAM_FRONT", "CAM_FRONT_RIGHT"),
        ],
    )
    assert summary["overall"]["residual_l2"]["max"] == 0.0
    assert summary["temporal_first_difference_l2"]["max"] == 0.0


def test_native_affine_heldout_policy_uses_mean_embedding() -> None:
    state = {
        "embedding.weight": torch.arange(48, dtype=torch.float32).reshape(12, 4) / 48,
        "decoder.0.weight": torch.zeros(64, 4),
        "decoder.0.bias": torch.ones(64),
        "decoder.2.weight": torch.zeros(12, 64),
        "decoder.2.bias": torch.zeros(12),
    }
    state["decoder.2.weight"][0, 0] = 0.1
    deployed, kind, indexed = affine_matrices_from_state(
        state, variant="c1-native", num_frames=4, num_cameras=3
    )
    assert kind == "native_mean_embedding_heldout_policy"
    assert indexed is not None
    torch.testing.assert_close(deployed, deployed[0].expand_as(deployed))
    assert deployed[0, 0, 0] == pytest.approx(1.1)


def test_bounded_pose_exact_zero_has_zero_rotation() -> None:
    state = {"embeds.weight": torch.zeros(12, 6)}
    _, rotation, _ = pose_residuals_from_state(
        state,
        variant="c3-bounded-pose",
        num_frames=4,
        num_cameras=3,
        bounded_translation_max_m=0.15,
        bounded_rotation_max_deg=2.0,
    )
    expected = torch.eye(3, dtype=torch.float64).expand(12, 3, 3)
    torch.testing.assert_close(rotation, expected, atol=1e-12, rtol=0.0)
