"""WorldSim V3 A1 的 ISP、位姿与速度分层只读诊断。"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from motion_proj.worldsim_v3.calibration import _bounded_vector


VARIANTS = (
    "c0-off",
    "c1-native",
    "c2-factorized-isp",
    "c3-bounded-pose",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_diagnostic_contract(contract: Mapping[str, Any]) -> None:
    if int(contract.get("schema_version", -1)) != 1:
        raise ValueError("diagnostic schema_version must be 1")
    if contract.get("diagnostic_version") != "A1-D0-v1":
        raise ValueError("unexpected diagnostic_version")
    if contract.get("task_id") != "WS-V3-A1-CALIBRATION-01":
        raise ValueError("unexpected task_id")
    if contract.get("frozen_before_diagnostic_result_access") is not True:
        raise ValueError("diagnostic contract must be frozen before result access")
    scene = contract.get("scene") or {}
    if int(scene.get("num_frames", 0)) <= 2 or int(scene.get("num_cameras", 0)) != 3:
        raise ValueError("diagnostic scene dimensions are invalid")
    camera_map = {int(key): value for key, value in scene.get("camera_id_to_name", {}).items()}
    expected_map = {0: "CAM_FRONT", 1: "CAM_FRONT_LEFT", 2: "CAM_FRONT_RIGHT"}
    if camera_map != expected_map:
        raise ValueError(f"camera mapping mismatch: {camera_map}")
    speed = contract.get("speed_tiers") or {}
    near = float(speed.get("near_static_upper_mps", -1.0))
    low = float(speed.get("low_speed_upper_mps", -1.0))
    if not (0.0 < near < low):
        raise ValueError("speed thresholds must satisfy 0 < near < low")
    if speed.get("frame_alignment") != "previous_interval_right_aligned":
        raise ValueError("unexpected speed frame alignment")
    isp = contract.get("isp") or {}
    pairs = [tuple(row) for row in isp.get("camera_pairs", [])]
    if pairs != [
        ("CAM_FRONT_LEFT", "CAM_FRONT"),
        ("CAM_FRONT", "CAM_FRONT_RIGHT"),
    ]:
        raise ValueError(f"camera pairs mismatch: {pairs}")


def scalar_summary(values: Tensor | np.ndarray) -> dict[str, float | int | None]:
    tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "max": float(tensor.max()),
    }


def rotation_6d_to_matrix(values: Tensor) -> Tensor:
    if values.shape[-1] != 6:
        raise ValueError("6D rotation tensor must end in dimension 6")
    first, second = values[..., :3], values[..., 3:]
    basis1 = torch.nn.functional.normalize(first, dim=-1)
    projection = (basis1 * second).sum(dim=-1, keepdim=True)
    basis2 = torch.nn.functional.normalize(second - projection * basis1, dim=-1)
    basis3 = torch.cross(basis1, basis2, dim=-1)
    return torch.stack((basis1, basis2, basis3), dim=-2)


def rotation_angle_degrees(matrices: Tensor) -> Tensor:
    trace = torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1)
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def load_input_speed_contract(
    scene_root: Path,
    *,
    num_frames: int,
    source_camera_id: int,
    processed_hz: float,
    near_static_upper_mps: float,
    low_speed_upper_mps: float,
) -> tuple[Tensor, dict[str, Any]]:
    paths = [
        scene_root / "extrinsics" / f"{frame:03d}_{source_camera_id}.txt"
        for frame in range(num_frames)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input extrinsics: {missing[:3]}")
    matrices = np.stack([np.loadtxt(path) for path in paths])
    if matrices.shape != (num_frames, 4, 4):
        raise ValueError(f"unexpected input pose shape: {matrices.shape}")
    positions = matrices[:, :3, 3]
    step_speed = np.linalg.norm(np.diff(positions, axis=0), axis=-1) * processed_hz
    frame_speed = np.empty(num_frames, dtype=np.float64)
    frame_speed[0] = step_speed[0]
    frame_speed[1:] = step_speed
    tiers = speed_tier_masks(
        torch.from_numpy(frame_speed),
        near_static_upper_mps=near_static_upper_mps,
        low_speed_upper_mps=low_speed_upper_mps,
    )
    return torch.from_numpy(frame_speed), {
        "source_camera_id": int(source_camera_id),
        "processed_hz": float(processed_hz),
        "frame_alignment": "previous_interval_right_aligned",
        "extrinsics_sha256": {str(path): sha256_file(path) for path in paths},
        "speed_mps": scalar_summary(frame_speed),
        "tier_frame_counts": {name: int(mask.sum()) for name, mask in tiers.items()},
    }


def speed_tier_masks(
    frame_speed_mps: Tensor,
    *,
    near_static_upper_mps: float,
    low_speed_upper_mps: float,
) -> dict[str, Tensor]:
    speed = torch.as_tensor(frame_speed_mps, dtype=torch.float64)
    return {
        "near_static": speed < near_static_upper_mps,
        "low_speed": (speed >= near_static_upper_mps) & (speed < low_speed_upper_mps),
        "normal": speed >= low_speed_upper_mps,
    }


def pose_residuals_from_state(
    state: Mapping[str, Tensor] | None,
    *,
    variant: str,
    num_frames: int,
    num_cameras: int,
    bounded_translation_max_m: float,
    bounded_rotation_max_deg: float,
) -> tuple[Tensor, Tensor, str]:
    count = num_frames * num_cameras
    identity = torch.eye(3, dtype=torch.float64).expand(count, 3, 3).clone()
    if state is None:
        if variant != "c0-off":
            raise ValueError(f"{variant} unexpectedly lacks CamPose")
        return torch.zeros(count, 3, dtype=torch.float64), identity, "absent_identity"
    raw = state.get("embeds.weight")
    if raw is None or int(raw.shape[0]) != count:
        raise ValueError(f"invalid CamPose state for {variant}")
    raw = raw.detach().cpu().to(torch.float64)
    if "identity" in state:
        if raw.shape[1] != 9:
            raise ValueError("native CamPose must use 9 parameters")
        translation = raw[:, :3]
        identity6 = state["identity"].detach().cpu().to(torch.float64)
        rotation = rotation_6d_to_matrix(raw[:, 3:] + identity6)
        return translation, rotation, "native_unbounded"
    if variant != "c3-bounded-pose" or raw.shape[1] != 6:
        raise ValueError(f"invalid bounded CamPose state for {variant}")
    translation = _bounded_vector(raw[:, :3], bounded_translation_max_m)
    rotation_vector = _bounded_vector(raw[:, 3:], math.radians(bounded_rotation_max_deg))
    from motion_proj.worldsim_v3.calibration import axis_angle_to_matrix

    return translation, axis_angle_to_matrix(rotation_vector), "bounded_axis_angle"


def _reshape_frame_camera(values: Tensor, num_frames: int, num_cameras: int) -> Tensor:
    if int(values.shape[0]) != num_frames * num_cameras:
        raise ValueError("first tensor dimension does not match frame-camera contract")
    return values.reshape(num_frames, num_cameras, *values.shape[1:])


def summarize_pose_residuals(
    translation: Tensor,
    rotation: Tensor,
    *,
    frame_speed_mps: Tensor,
    camera_id_to_name: Mapping[int, str],
    near_static_upper_mps: float,
    low_speed_upper_mps: float,
    minimum_frames_per_tier: int,
) -> dict[str, Any]:
    num_frames = int(frame_speed_mps.numel())
    num_cameras = len(camera_id_to_name)
    translation = _reshape_frame_camera(translation, num_frames, num_cameras)
    rotation = _reshape_frame_camera(rotation, num_frames, num_cameras)
    translation_norm = torch.linalg.vector_norm(translation, dim=-1)
    rotation_angle = rotation_angle_degrees(rotation)
    tiers = speed_tier_masks(
        frame_speed_mps,
        near_static_upper_mps=near_static_upper_mps,
        low_speed_upper_mps=low_speed_upper_mps,
    )
    by_tier: dict[str, Any] = {}
    for name, mask in tiers.items():
        frame_count = int(mask.sum())
        if frame_count < minimum_frames_per_tier:
            by_tier[name] = {
                "status": "ABSTAIN",
                "reason": "INSUFFICIENT_SPEED_TIER_FRAMES",
                "frame_count": frame_count,
            }
        else:
            by_tier[name] = {
                "status": "done",
                "reason": None,
                "frame_count": frame_count,
                "translation_norm_m": scalar_summary(translation_norm[mask]),
                "rotation_angle_deg": scalar_summary(rotation_angle[mask]),
            }
    by_camera = {
        camera_id_to_name[index]: {
            "translation_norm_m": scalar_summary(translation_norm[:, index]),
            "rotation_angle_deg": scalar_summary(rotation_angle[:, index]),
        }
        for index in range(num_cameras)
    }
    translation_first = translation[1:] - translation[:-1]
    translation_second = translation_first[1:] - translation_first[:-1]
    rotation_first_matrix = rotation[:-1].transpose(-1, -2) @ rotation[1:]
    rotation_second_matrix = (
        rotation_first_matrix[:-1].transpose(-1, -2) @ rotation_first_matrix[1:]
    )
    first = {
        "translation_delta_norm_m": scalar_summary(
            torch.linalg.vector_norm(translation_first, dim=-1)
        ),
        "rotation_delta_deg": scalar_summary(rotation_angle_degrees(rotation_first_matrix)),
    }
    second = {
        "translation_jitter_norm_m": scalar_summary(
            torch.linalg.vector_norm(translation_second, dim=-1)
        ),
        "rotation_jitter_deg": scalar_summary(rotation_angle_degrees(rotation_second_matrix)),
    }
    return {
        "truth_tier": "learned_camera_correction_diagnostic_not_pose_ground_truth",
        "nominal_input_pose_error": "not_measurable_without_independent_pose_ground_truth",
        "overall": {
            "translation_norm_m": scalar_summary(translation_norm),
            "rotation_angle_deg": scalar_summary(rotation_angle),
        },
        "by_camera": by_camera,
        "by_speed_tier": by_tier,
        "first_difference": first,
        "second_difference": second,
    }


def _linear(values: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return values @ weight.transpose(0, 1) + bias


def native_affine_matrices(
    state: Mapping[str, Tensor], *, num_frames: int, num_cameras: int
) -> tuple[Tensor, Tensor]:
    count = num_frames * num_cameras
    embedding = state["embedding.weight"].detach().cpu().to(torch.float64)
    if embedding.shape != (count, 4):
        raise ValueError(f"unexpected native affine embedding shape: {embedding.shape}")
    w0 = state["decoder.0.weight"].detach().cpu().to(torch.float64)
    b0 = state["decoder.0.bias"].detach().cpu().to(torch.float64)
    w2 = state["decoder.2.weight"].detach().cpu().to(torch.float64)
    b2 = state["decoder.2.bias"].detach().cpu().to(torch.float64)

    def decode(features: Tensor) -> Tensor:
        residual = _linear(torch.relu(_linear(features, w0, b0)), w2, b2).reshape(-1, 3, 4)
        residual[:, :, :3] += torch.eye(3, dtype=residual.dtype)
        return residual

    indexed = decode(embedding)
    deployed = decode(embedding.mean(dim=0, keepdim=True)).expand(count, 3, 4).clone()
    return deployed, indexed


def factorized_affine_matrices(
    state: Mapping[str, Tensor],
    *,
    num_frames: int,
    num_cameras: int,
    num_time_frequencies: int = 2,
) -> Tensor:
    camera_embedding = state["camera_embedding.weight"].detach().cpu().to(torch.float64)
    if camera_embedding.shape[0] != num_cameras:
        raise ValueError("factorized affine camera count mismatch")
    normalized_time = torch.linspace(0.0, 1.0, num_frames, dtype=torch.float64)
    time_features = [normalized_time]
    for frequency in range(num_time_frequencies):
        angle = normalized_time * (2.0**frequency) * (2.0 * math.pi)
        time_features.extend((torch.sin(angle), torch.cos(angle)))
    time_features_tensor = torch.stack(time_features, dim=-1)
    time_weight = state["time_encoder.0.weight"].detach().cpu().to(torch.float64)
    time_bias = state["time_encoder.0.bias"].detach().cpu().to(torch.float64)
    time_embedding = torch.tanh(_linear(time_features_tensor, time_weight, time_bias))
    features = torch.cat(
        (
            camera_embedding.unsqueeze(0).expand(num_frames, -1, -1),
            time_embedding.unsqueeze(1).expand(-1, num_cameras, -1),
        ),
        dim=-1,
    ).reshape(num_frames * num_cameras, -1)
    w0 = state["decoder.0.weight"].detach().cpu().to(torch.float64)
    b0 = state["decoder.0.bias"].detach().cpu().to(torch.float64)
    w2 = state["decoder.2.weight"].detach().cpu().to(torch.float64)
    b2 = state["decoder.2.bias"].detach().cpu().to(torch.float64)
    residual = _linear(torch.relu(_linear(features, w0, b0)), w2, b2).reshape(-1, 3, 4)
    residual[:, :, :3] += torch.eye(3, dtype=residual.dtype)
    return residual


def affine_matrices_from_state(
    state: Mapping[str, Tensor] | None,
    *,
    variant: str,
    num_frames: int,
    num_cameras: int,
) -> tuple[Tensor, str, Tensor | None]:
    count = num_frames * num_cameras
    if state is None:
        if variant != "c0-off":
            raise ValueError(f"{variant} unexpectedly lacks Affine")
        identity = torch.zeros(count, 3, 4, dtype=torch.float64)
        identity[:, :, :3] = torch.eye(3, dtype=torch.float64)
        return identity, "absent_identity", None
    if "embedding.weight" in state:
        deployed, indexed = native_affine_matrices(
            state, num_frames=num_frames, num_cameras=num_cameras
        )
        return deployed, "native_mean_embedding_heldout_policy", indexed
    if "camera_embedding.weight" in state:
        deployed = factorized_affine_matrices(
            state, num_frames=num_frames, num_cameras=num_cameras
        )
        return deployed, "factorized_camera_continuous_time", None
    raise ValueError(f"unrecognized Affine state for {variant}")


def summarize_affines(
    matrices: Tensor,
    *,
    num_frames: int,
    camera_id_to_name: Mapping[int, str],
    camera_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    num_cameras = len(camera_id_to_name)
    matrices = _reshape_frame_camera(matrices, num_frames, num_cameras)
    identity = torch.eye(3, dtype=matrices.dtype)
    gain_residual = matrices[..., :3] - identity
    bias = matrices[..., 3]
    residual = torch.cat((gain_residual.reshape(num_frames, num_cameras, 9), bias), dim=-1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    gain_norm = torch.linalg.matrix_norm(gain_residual, ord="fro")
    bias_norm = torch.linalg.vector_norm(bias, dim=-1)
    temporal_first = residual[1:] - residual[:-1]
    temporal_second = temporal_first[1:] - temporal_first[:-1]
    name_to_id = {name: index for index, name in camera_id_to_name.items()}
    pair_metrics = {}
    for first, second in camera_pairs:
        distance = torch.linalg.vector_norm(
            residual[:, name_to_id[first]] - residual[:, name_to_id[second]], dim=-1
        )
        pair_metrics[f"{first}<->{second}"] = scalar_summary(distance)
    return {
        "truth_tier": "learned_rgb_affine_diagnostic",
        "overall": {
            "residual_l2": scalar_summary(residual_norm),
            "gain_minus_identity_frobenius": scalar_summary(gain_norm),
            "bias_l2": scalar_summary(bias_norm),
        },
        "by_camera": {
            camera_id_to_name[index]: {
                "residual_l2": scalar_summary(residual_norm[:, index]),
                "gain_minus_identity_frobenius": scalar_summary(gain_norm[:, index]),
                "bias_l2": scalar_summary(bias_norm[:, index]),
            }
            for index in range(num_cameras)
        },
        "temporal_first_difference_l2": scalar_summary(
            torch.linalg.vector_norm(temporal_first, dim=-1)
        ),
        "temporal_second_difference_l2": scalar_summary(
            torch.linalg.vector_norm(temporal_second, dim=-1)
        ),
        "camera_pair_difference_l2": pair_metrics,
    }


def evaluate_checkpoint_diagnostics(
    checkpoint: Path,
    *,
    variant: str,
    contract: Mapping[str, Any],
    frame_speed_mps: Tensor,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unexpected variant: {variant}")
    scene = contract["scene"]
    pose_cfg = contract["pose"]
    speed_cfg = contract["speed_tiers"]
    isp_cfg = contract["isp"]
    num_frames = int(scene["num_frames"])
    num_cameras = int(scene["num_cameras"])
    camera_map = {int(key): value for key, value in scene["camera_id_to_name"].items()}
    checkpoint_sha_before = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    models = payload.get("models", {})
    translation, rotation, pose_kind = pose_residuals_from_state(
        models.get("CamPose"),
        variant=variant,
        num_frames=num_frames,
        num_cameras=num_cameras,
        bounded_translation_max_m=float(pose_cfg["bounded_translation_max_m"]),
        bounded_rotation_max_deg=float(pose_cfg["bounded_rotation_max_deg"]),
    )
    deployed_affine, affine_kind, indexed_affine = affine_matrices_from_state(
        models.get("Affine"),
        variant=variant,
        num_frames=num_frames,
        num_cameras=num_cameras,
    )
    pose_summary = summarize_pose_residuals(
        translation,
        rotation,
        frame_speed_mps=frame_speed_mps,
        camera_id_to_name=camera_map,
        near_static_upper_mps=float(speed_cfg["near_static_upper_mps"]),
        low_speed_upper_mps=float(speed_cfg["low_speed_upper_mps"]),
        minimum_frames_per_tier=int(speed_cfg["minimum_frames_per_tier"]),
    )
    camera_pairs = [tuple(row) for row in isp_cfg["camera_pairs"]]
    isp_summary = summarize_affines(
        deployed_affine,
        num_frames=num_frames,
        camera_id_to_name=camera_map,
        camera_pairs=camera_pairs,
    )
    if indexed_affine is not None:
        isp_summary["native_training_index_auxiliary"] = summarize_affines(
            indexed_affine,
            num_frames=num_frames,
            camera_id_to_name=camera_map,
            camera_pairs=camera_pairs,
        )
    del payload, models
    checkpoint_sha_after = sha256_file(checkpoint)
    return {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after,
        "pose_kind": pose_kind,
        "pose": pose_summary,
        "isp_kind": affine_kind,
        "isp": isp_summary,
    }
