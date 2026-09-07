from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from motion_proj.worldsim_v72.data.camera_schema import CameraFramePayload, CameraWindow
from motion_proj.worldsim_v72.eas_vggt.models import (
    EvidenceConditionedSurfaceAdapter,
    MatchedScalarSurfaceAdapter,
    lift_actor_surface_to_world,
    trainable_parameter_count,
)
from motion_proj.worldsim_v72.eas_vggt.types import BackboneGeometry
from motion_proj.worldsim_v72.eas_vggt.visual_pooling import observe_actor_candidates
from motion_proj.worldsim_v72.eas_vggt.appearance import render_detached_gaussian_appearance


def _window(tmp_path: Path) -> CameraWindow:
    image_path = (tmp_path / "unused.jpg").resolve()
    image_path.write_bytes(b"rgb")
    frame = CameraFramePayload(
        frame_id="frame-0",
        sample_id="sample-0",
        camera_id="CAM_FRONT",
        time_ns=0,
        image_path=str(image_path),
        image_sha256="0" * 64,
        original_size_wh=np.asarray([4, 4]),
        intrinsics_px=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        world_from_camera_opencv=np.eye(4),
        distortion_model="pinhole_rectified",
        distortion_parameters=np.empty(0),
        provenance={"payload_role": "build_input"},
    )
    return CameraWindow(
        dataset="synthetic",
        role="train",
        log_id="log-0",
        scene_id="scene-0",
        window_id="sample-0",
        frames=(frame,),
        provenance={"target_access": False},
    )


def _geometry() -> BackboneGeometry:
    grid = np.zeros((1, 2, 2, 2), dtype=np.float32)
    grid[0, 0, 0] = [0.0, 0.0]
    grid[0, 0, 1] = [2.0, 0.0]
    grid[0, 1, 0] = [0.0, 2.0]
    grid[0, 1, 1] = [2.0, 2.0]
    return BackboneGeometry(
        backbone_id="mock",
        checkpoint_id="sha256:" + "a" * 64,
        repository_commit="b" * 40,
        frame_ids=np.asarray(["frame-0"]),
        image_sha256=np.asarray(["0" * 64]),
        model_from_original_px=np.eye(3)[None],
        points_reference=np.zeros((1, 4, 4, 3), dtype=np.float32),
        confidence=np.ones((1, 4, 4), dtype=np.float32),
        valid_mask=np.ones((1, 4, 4), dtype=bool),
        reference_from_camera_opencv=np.eye(4)[None],
        intrinsics_model_px=np.eye(3)[None],
        feature_grid=grid,
        scale_status="metric_aligned",
        provenance={"payload_role": "build_input"},
    )


def test_candidate_projection_pools_visible_features_and_rejects_behind_camera(tmp_path: Path) -> None:
    candidates = np.asarray([[1.5, 1.5, 1.0], [0.0, 0.0, -1.0], [9.0, 9.0, 1.0]])
    observation = observe_actor_candidates(candidates, np.eye(4), _window(tmp_path), _geometry())
    assert observation.observation_count.tolist() == [1, 0, 0]
    np.testing.assert_allclose(observation.pooled_features[0], [1.0, 1.0], atol=1.0e-6)
    assert observation.pooled_geometry_features.shape == (3, 5)
    np.testing.assert_array_equal(observation.pooled_features[1:], 0.0)


def _inputs(count: int = 7) -> dict[str, torch.Tensor]:
    return {
        "base_features": torch.randn(count, 11),
        "canonical_xyz": torch.randn(count, 3),
        "evidence_fou": torch.softmax(torch.randn(count, 3), dim=-1),
        "opportunity_count": torch.randint(0, 9, (count,)),
        "visual_features": torch.randn(count, 32),
        "visual_observed": torch.randint(0, 2, (count,), dtype=torch.bool),
        "geometric_features": torch.randn(count, 5),
        "geometry_observed": torch.randint(0, 2, (count,), dtype=torch.bool),
    }


def test_evidence_adapter_outputs_normalized_measure_and_finite_gradients() -> None:
    model = EvidenceConditionedSurfaceAdapter(base_feature_dim=11, visual_feature_dim=32, hidden_dim=64)
    output = model(**_inputs())
    assert output.surface_delta_actor_m.shape == (7, 3)
    assert output.evidence_fou.shape == (7, 3)
    assert torch.allclose(output.evidence_fou.sum(dim=-1), torch.ones(7), atol=1.0e-6)
    assert torch.all(output.evidence_strength > 0.0)
    assert torch.all(output.surface_delta_actor_m.abs() <= 0.300001)
    loss = (
        output.surface_delta_actor_m.square().mean()
        + output.evidence_strength.mean()
        + output.evidence_fou.square().mean()
        + output.blocking_logit.square().mean()
        + output.detection_logit.square().mean()
    )
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_scalar_control_matches_trunk_capacity() -> None:
    kwargs = {"base_feature_dim": 11, "visual_feature_dim": 2048, "hidden_dim": 256}
    evidence = EvidenceConditionedSurfaceAdapter(**kwargs)
    scalar = MatchedScalarSurfaceAdapter(**kwargs)
    ratio = abs(trainable_parameter_count(evidence) - trainable_parameter_count(scalar)) / trainable_parameter_count(evidence)
    assert ratio < 0.005
    delta, response = scalar(**_inputs(count=7) | {"visual_features": torch.randn(7, 2048)})
    assert delta.shape == (7, 3)
    assert response.shape == (7,)


def test_surface_geometry_is_appearance_invariant_and_se3_equivariant() -> None:
    torch.manual_seed(11)
    model = EvidenceConditionedSurfaceAdapter(base_feature_dim=11, visual_feature_dim=32, hidden_dim=64)
    inputs = _inputs()
    first = model(**inputs).surface_delta_actor_m
    inputs["visual_features"] = 100.0 * torch.randn_like(inputs["visual_features"])
    second = model(**inputs).surface_delta_actor_m
    assert torch.equal(first, second)

    angle = torch.tensor(0.7)
    c, s = torch.cos(angle), torch.sin(angle)
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = torch.tensor([3.0, -2.0, 0.5])
    candidates = inputs["canonical_xyz"]
    base_world = lift_actor_surface_to_world(candidates, first, torch.eye(4))
    transformed_world = lift_actor_surface_to_world(candidates, first, transform)
    expected = base_world @ transform[:3, :3].T + transform[:3, 3]
    assert torch.allclose(transformed_world, expected, atol=1.0e-6)


def test_rgb_render_gradients_stop_at_physical_points() -> None:
    points = torch.tensor([[2.0, 2.0, 1.0], [1.0, 1.0, 2.0]], requires_grad=True)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    opacity = torch.tensor([0.8, 0.6], requires_grad=True)
    rendered = render_detached_gaussian_appearance(
        points,
        colors,
        opacity,
        torch.eye(4),
        torch.eye(3),
        height=5,
        width=5,
    )
    rendered.rgb.square().mean().backward()
    assert points.grad is None
    assert colors.grad is not None and torch.isfinite(colors.grad).all()
    assert opacity.grad is not None and torch.isfinite(opacity.grad).all()
