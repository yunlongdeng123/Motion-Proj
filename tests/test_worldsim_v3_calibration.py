import math

import pytest
import torch

from motion_proj.worldsim_v3.calibration import (
    BoundedCameraOptModule,
    FactorizedAffineTransform,
    axis_angle_to_matrix,
)


def test_factorized_affine_starts_as_identity_and_supports_heldout_time() -> None:
    module = FactorizedAffineTransform(
        class_name="Affine", n=12, num_cameras=3, device=torch.device("cpu")
    )
    first = module(
        {"img_idx": torch.tensor([5]), "normed_time": torch.tensor([0.4])}
    )
    module.in_test_set = True
    heldout = module(
        {"img_idx": torch.tensor([8]), "normed_time": torch.tensor([0.7])}
    )
    expected = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=-1)
    assert torch.equal(first, expected)
    assert torch.equal(heldout, expected)


def test_factorized_affine_validates_image_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        FactorizedAffineTransform(class_name="Affine", n=10, num_cameras=3)


def test_axis_angle_rotation() -> None:
    matrix = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, math.pi / 2]]))[0]
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(matrix, expected, atol=1e-6)
    assert torch.allclose(matrix.T @ matrix, torch.eye(3), atol=1e-6)


def test_bounded_camera_pose_respects_norm_and_has_smoothness_loss() -> None:
    module = BoundedCameraOptModule(
        class_name="CamPose",
        n=12,
        num_cameras=3,
        max_translation_m=0.15,
        max_rotation_deg=2.0,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        module.embeds.weight.fill_(100.0)
    translation, rotation = module.bounded_residuals()
    assert torch.linalg.vector_norm(translation, dim=-1).max() <= 0.15 + 1e-6
    assert torch.linalg.vector_norm(rotation, dim=-1).max() <= math.radians(2.0) + 1e-6
    regularization = module.compute_regularization()
    assert set(regularization) == {
        "translation_prior",
        "rotation_prior",
        "temporal_smoothness",
    }
    assert regularization["translation_prior"] > 0
    assert regularization["rotation_prior"] > 0
    assert regularization["temporal_smoothness"] == 0


def test_bounded_camera_pose_identity_at_zero() -> None:
    module = BoundedCameraOptModule(
        class_name="CamPose", n=6, num_cameras=3, device=torch.device("cpu")
    )
    camera = torch.eye(4)[None]
    result = module(camera, torch.tensor([0]))
    assert torch.equal(result, camera)


def test_bounded_camera_pose_has_nonzero_gradient_at_zero() -> None:
    module = BoundedCameraOptModule(
        class_name="CamPose", n=6, num_cameras=3, device=torch.device("cpu")
    )
    translation, _ = module.bounded_residuals()
    translation.sum().backward()
    gradient = module.embeds.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.equal(
        gradient[:, :3], torch.full_like(gradient[:, :3], 0.15)
    )
