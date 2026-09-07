from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from motion_proj.worldsim_v72.data.camera_schema import CameraFramePayload, CameraWindow
from motion_proj.worldsim_v72.eas_vggt.alignment import (
    aligned_camera_center_rmse_m,
    nearest_pixel_indices,
)
from motion_proj.worldsim_v72.eas_vggt.cache import load_backbone_geometry, save_backbone_geometry
from motion_proj.worldsim_v72.eas_vggt.types import BackboneGeometry


def _window(image_path: Path) -> CameraWindow:
    image_path.write_bytes(b"rgb")
    frames = []
    for index in range(3):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(index)
        frames.append(
            CameraFramePayload(
                frame_id=f"frame-{index}",
                sample_id="sample-1",
                camera_id=f"cam-{index}",
                time_ns=index,
                image_path=str(image_path.resolve()),
                image_sha256="0" * 64,
                original_size_wh=np.asarray([1600, 900]),
                intrinsics_px=np.asarray([[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]),
                world_from_camera_opencv=pose,
                distortion_model="pinhole_rectified",
                distortion_parameters=np.empty(0),
                provenance={"payload_role": "build_input"},
            )
        )
    return CameraWindow(
        dataset="nuScenes",
        role="train",
        log_id="log-1",
        scene_id="scene-1",
        window_id="sample-1",
        frames=tuple(frames),
        provenance={"target_access": False},
    )


def test_camera_contract_contains_no_supervision_and_has_stable_fingerprint(tmp_path: Path) -> None:
    window = _window(tmp_path / "frame.jpg")
    names = {field.name for field in fields(CameraFramePayload)} | {field.name for field in fields(CameraWindow)}
    assert not {"target", "label", "ground_truth", "heldout"} & names
    assert window.fingerprint == window.fingerprint


def test_backbone_cache_round_trip_and_similarity_alignment(tmp_path: Path) -> None:
    count, height, width = 3, 2, 2
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (count, 1, 1))
    poses[:, 0, 3] = [0.0, 1.0, 2.0]
    world = poses.copy()
    world[:, :3, 3] = 2.5 * poses[:, :3, 3] + np.asarray([3.0, -1.0, 0.5])
    _, rmse = aligned_camera_center_rmse_m(poses, world)
    assert rmse < 1.0e-6
    geometry = BackboneGeometry(
        backbone_id="mock",
        checkpoint_id="sha256:" + "a" * 64,
        repository_commit="b" * 40,
        frame_ids=np.asarray([f"frame-{i}" for i in range(count)]),
        image_sha256=np.asarray(["0" * 64] * count),
        model_from_original_px=np.tile(np.eye(3)[None], (count, 1, 1)),
        points_reference=np.zeros((count, height, width, 3), dtype=np.float32),
        confidence=np.ones((count, height, width), dtype=np.float32),
        valid_mask=np.ones((count, height, width), dtype=bool),
        reference_from_camera_opencv=poses,
        intrinsics_model_px=np.tile(np.eye(3)[None], (count, 1, 1)),
        feature_grid=np.zeros((count, 1, 1, 4), dtype=np.float16),
        scale_status="arbitrary",
        provenance={"payload_role": "build_input"},
    )
    output = tmp_path / "geometry.npz"
    save_backbone_geometry(geometry, output)
    restored = load_backbone_geometry(output)
    assert restored.backbone_id == "mock"
    assert restored.points_reference.shape == geometry.points_reference.shape


def test_lidar_projection_keeps_nearest_depth_per_model_pixel() -> None:
    selected = nearest_pixel_indices(
        np.asarray([2, 2, 3, 3]),
        np.asarray([1, 1, 1, 1]),
        np.asarray([8.0, 3.0, 4.0, 6.0]),
        image_width=8,
    )
    assert selected.tolist() == [1, 2]
