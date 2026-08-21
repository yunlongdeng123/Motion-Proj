"""StreetGS 与 ReconDrive 到 SceneIR v0 的无训练适配器。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def normalize_quaternions(value: np.ndarray) -> np.ndarray:
    """按最后一维归一化 wxyz quaternion，并拒绝退化输入。"""
    quaternions = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if not np.isfinite(quaternions).all() or (norms <= 1e-12).any():
        raise ValueError("quaternion 含非有限值或零范数")
    return np.ascontiguousarray(quaternions / norms)


def quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result[..., 1:] *= -1
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """计算 wxyz Hamilton product，支持 NumPy broadcasting。"""
    a, b, c, d = np.moveaxis(np.asarray(left), -1, 0)
    e, f, g, h = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ),
        axis=-1,
    )


def quaternion_to_matrix(value: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(value)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    result = np.empty_like(source)
    positive = source >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exponential = np.exp(source[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def _base_document(
    *,
    episode_id: str,
    seed: int,
    timestamps_us: Sequence[int],
    source_sha256: str,
    source_uri: str,
    reconstructor_version: str,
    adapter_id: str,
    support_boundary: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "worldsim.sceneir.v0",
        "content_sha256": "0" * 64,
        "coordinate_system": {
            "id": "native_world_m",
            "handedness": "right",
            "length_unit": "meter",
            "axes": {"x": "native_axis_0", "y": "native_axis_1", "z": "native_axis_2"},
            "transform_convention": "T_dst_src",
            "quaternion_convention": "wxyz_hamilton_active",
        },
        "episode": {
            "id": episode_id,
            "seed": int(seed),
            "start_timestamp_us": int(timestamps_us[0]),
            "end_timestamp_us": int(timestamps_us[-1]),
        },
        "frames": [{"id": "world", "frame_type": "world"}],
        "transforms": [],
        "static_world": {
            "chunk_ids": [],
            "surfaces": [],
            "collision_proxy": None,
            "map_binding": None,
        },
        "actors": [],
        "sensors": [],
        "provenance": [
            {
                "id": "source_reconstruction",
                "kind": "reconstructed",
                "source_uri": source_uri,
                "source_sha256": source_sha256,
                "reconstructor_version": reconstructor_version,
                "adapter_id": adapter_id,
            }
        ],
        "support": [
            {
                "id": "episode_support",
                "observed_timestamp_us": [int(value) for value in timestamps_us],
                "observed_view_ids": [],
                "boundary": support_boundary,
            }
        ],
        "validity": {
            "assessment_status": "unassessed_interface_only",
            "q_photo": 0.0,
            "q_geometry": 0.0,
            "q_semantic": 0.0,
            "q_dynamics": 0.0,
        },
        "chunks": [],
    }


def _gaussian_arrays(
    *,
    means: np.ndarray,
    scales: np.ndarray,
    quaternions: np.ndarray,
    opacities: np.ndarray,
    features_dc: np.ndarray,
    features_rest: np.ndarray,
    source_indices: np.ndarray,
    velocities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    arrays = {
        "means_m": np.asarray(means, dtype=np.float32),
        "scales_m": np.asarray(scales, dtype=np.float32),
        "quaternions_wxyz": normalize_quaternions(quaternions),
        "opacities": np.asarray(opacities, dtype=np.float32),
        "features_dc": np.asarray(features_dc, dtype=np.float32),
        "features_rest": np.asarray(features_rest, dtype=np.float32),
        "source_indices": np.asarray(source_indices, dtype=np.int64),
    }
    if velocities is not None:
        arrays["velocities_mps"] = np.asarray(velocities, dtype=np.float32)
    return {name: np.ascontiguousarray(value) for name, value in arrays.items()}


def _append_chunk(
    document: dict[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
    *,
    chunk_id: str,
    role: str,
    frame_id: str,
    actor_id: str | None,
    values: dict[str, np.ndarray],
) -> None:
    count = int(values["means_m"].shape[0])
    if any(value.shape[0] != count for value in values.values()):
        raise ValueError(f"{chunk_id} 数组第一维不一致")
    document["chunks"].append(
        {
            "id": chunk_id,
            "role": role,
            "frame_id": frame_id,
            "actor_id": actor_id,
            "primitive_type": "gaussian_splat",
            "primitive_count": count,
            "provenance_id": "source_reconstruction",
            "support_id": "episode_support",
            "content_sha256": "0" * 64,
            "arrays": {},
        }
    )
    arrays[chunk_id] = values


def streetgs_to_sceneir(
    checkpoint: Mapping[str, Any],
    *,
    source_sha256: str,
    source_uri: str,
    reconstructor_version: str,
    seed: int = 0,
    frame_period_us: int = 100_000,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """把 StreetGS checkpoint 的受支持 Gaussian core 转为 SceneIR。"""
    models = checkpoint["models"]
    background = models["Background"]
    rigid = models["RigidNodes"]
    translations = _numpy(rigid["instances_trans"]).astype(np.float32)
    actor_quaternions = normalize_quaternions(_numpy(rigid["instances_quats"]))
    visibility = _numpy(rigid["instances_fv"]).astype(bool)
    point_actor = _numpy(rigid["points_ids"])[..., 0].astype(np.int64)
    timestamps = [index * int(frame_period_us) for index in range(translations.shape[0])]
    document = _base_document(
        episode_id="streetgs_sceneir_reference",
        seed=seed,
        timestamps_us=timestamps,
        source_sha256=source_sha256,
        source_uri=source_uri,
        reconstructor_version=reconstructor_version,
        adapter_id="streetgs_checkpoint_v0",
        support_boundary=[
            "supported:Background_and_RigidNodes_gaussian_core",
            "excluded:Sky_Affine_CamPose_and_dataset_camera_calibration",
            "no_training_inference_or_quality_read",
        ],
    )
    arrays: dict[str, dict[str, np.ndarray]] = {}
    static_arrays = _gaussian_arrays(
        means=_numpy(background["_means"]),
        scales=np.exp(_numpy(background["_scales"])),
        quaternions=_numpy(background["_quats"]),
        opacities=_sigmoid(_numpy(background["_opacities"])),
        features_dc=_numpy(background["_features_dc"]),
        features_rest=_numpy(background["_features_rest"]),
        source_indices=np.arange(_numpy(background["_means"]).shape[0], dtype=np.int64),
    )
    _append_chunk(
        document,
        arrays,
        chunk_id="streetgs_background",
        role="static",
        frame_id="world",
        actor_id=None,
        values=static_arrays,
    )
    document["static_world"]["chunk_ids"].append("streetgs_background")

    rigid_common = {
        "means": _numpy(rigid["_means"]),
        "scales": np.exp(_numpy(rigid["_scales"])),
        "quaternions": _numpy(rigid["_quats"]),
        "opacities": _sigmoid(_numpy(rigid["_opacities"])),
        "features_dc": _numpy(rigid["_features_dc"]),
        "features_rest": _numpy(rigid["_features_rest"]),
    }
    for actor_index in sorted(int(value) for value in np.unique(point_actor)):
        if actor_index < 0 or actor_index >= translations.shape[1]:
            raise ValueError(f"StreetGS points_ids 越界：{actor_index}")
        mask = point_actor == actor_index
        actor_id = f"actor_{actor_index:04d}"
        frame_id = f"actor_{actor_index:04d}_canonical"
        chunk_id = f"streetgs_actor_{actor_index:04d}"
        document["frames"].append({"id": frame_id, "frame_type": "actor"})
        trajectory = []
        actor_visibility = []
        for frame_index, timestamp_us in enumerate(timestamps):
            document["transforms"].append(
                {
                    "name": f"T_world_{frame_id}",
                    "src_frame": frame_id,
                    "dst_frame": "world",
                    "timestamp_us": timestamp_us,
                    "translation_m": translations[frame_index, actor_index].astype(float).tolist(),
                    "rotation_wxyz": actor_quaternions[frame_index, actor_index].astype(float).tolist(),
                }
            )
            trajectory.append({"transform_name": f"T_world_{frame_id}", "timestamp_us": timestamp_us})
            actor_visibility.append({"timestamp_us": timestamp_us, "visible": bool(visibility[frame_index, actor_index])})
        document["actors"].append(
            {
                "id": actor_id,
                "class": "unknown",
                "canonical_representation": "gaussian_splat",
                "canonical_frame": frame_id,
                "chunk_ids": [chunk_id],
                "trajectory": trajectory,
                "visibility": actor_visibility,
            }
        )
        actor_arrays = _gaussian_arrays(
            means=rigid_common["means"][mask],
            scales=rigid_common["scales"][mask],
            quaternions=rigid_common["quaternions"][mask],
            opacities=rigid_common["opacities"][mask],
            features_dc=rigid_common["features_dc"][mask],
            features_rest=rigid_common["features_rest"][mask],
            source_indices=np.nonzero(mask)[0],
        )
        _append_chunk(
            document,
            arrays,
            chunk_id=chunk_id,
            role="actor",
            frame_id=frame_id,
            actor_id=actor_id,
            values=actor_arrays,
        )
    return document, arrays


def recondrive_to_sceneir(
    outputs: Mapping[str, np.ndarray],
    actor_assignments: np.ndarray,
    actor_poses: Mapping[int, Mapping[str, np.ndarray]],
    *,
    timestamps_us: Sequence[int],
    source_sha256: str,
    source_uri: str,
    reconstructor_version: str,
    camera: Mapping[str, Any],
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """把 ReconDrive 标准化输出与显式 actor assignment 转为 SceneIR。"""
    required = {"xyz", "rot_maps", "scale_maps", "opacity_maps", "sh_maps", "forward_flow"}
    if not required.issubset(outputs):
        raise ValueError(f"ReconDrive 输出缺少：{sorted(required - set(outputs))}")
    xyz = np.asarray(outputs["xyz"], dtype=np.float32)
    quaternions = normalize_quaternions(outputs["rot_maps"])
    scales = np.asarray(outputs["scale_maps"], dtype=np.float32)
    opacities = np.asarray(outputs["opacity_maps"], dtype=np.float32)
    sh = np.asarray(outputs["sh_maps"], dtype=np.float32)
    flow = np.asarray(outputs["forward_flow"], dtype=np.float32)
    assignments = np.asarray(actor_assignments, dtype=np.int64)
    count = xyz.shape[0]
    if xyz.shape != (count, 3) or quaternions.shape != (count, 4) or scales.shape != (count, 3):
        raise ValueError("ReconDrive Gaussian core shape 非法")
    if opacities.shape != (count, 1) or sh.ndim != 3 or sh.shape[:2] != (count, 3):
        raise ValueError("ReconDrive opacity/SH shape 非法")
    if flow.shape != (count, 3) or assignments.shape != (count,):
        raise ValueError("ReconDrive flow/assignment shape 非法")
    if (scales <= 0).any() or ((opacities < 0) | (opacities > 1)).any():
        raise ValueError("ReconDrive physical scale/opacity 越界")
    features_dc = sh[:, :, 0]
    features_rest = np.transpose(sh[:, :, 1:], (0, 2, 1))
    document = _base_document(
        episode_id="recondrive_sceneir_reference",
        seed=seed,
        timestamps_us=timestamps_us,
        source_sha256=source_sha256,
        source_uri=source_uri,
        reconstructor_version=reconstructor_version,
        adapter_id="recondrive_output_v0",
        support_boundary=[
            "supported:get_recontrast_data_standardized_gaussian_output",
            "actor_assignment_and_trajectory_are_explicit_adapter_inputs",
            "conformance_fixture_only:no_model_inference_or_quality_read",
        ],
    )
    sensor_frame = "camera_reference"
    document["frames"].append({"id": sensor_frame, "frame_type": "sensor"})
    document["sensors"].append(
        {
            "id": "camera_reference",
            "sensor_type": "camera",
            "frame_id": sensor_frame,
            "camera_model": camera["camera_model"],
            "resolution_px": list(camera["resolution_px"]),
            "calibration": {"intrinsics_3x3": camera["intrinsics_3x3"]},
        }
    )
    identity = [1.0, 0.0, 0.0, 0.0]
    document["transforms"].append(
        {
            "name": f"T_world_{sensor_frame}",
            "src_frame": sensor_frame,
            "dst_frame": "world",
            "timestamp_us": int(timestamps_us[0]),
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_wxyz": identity,
        }
    )
    arrays: dict[str, dict[str, np.ndarray]] = {}
    static_mask = assignments == -1
    static_values = _gaussian_arrays(
        means=xyz[static_mask],
        scales=scales[static_mask],
        quaternions=quaternions[static_mask],
        opacities=opacities[static_mask],
        features_dc=features_dc[static_mask],
        features_rest=features_rest[static_mask],
        source_indices=np.nonzero(static_mask)[0],
        velocities=flow[static_mask],
    )
    _append_chunk(
        document,
        arrays,
        chunk_id="recondrive_static",
        role="static",
        frame_id="world",
        actor_id=None,
        values=static_values,
    )
    document["static_world"]["chunk_ids"].append("recondrive_static")

    for actor_index in sorted(int(value) for value in np.unique(assignments) if value >= 0):
        if actor_index not in actor_poses:
            raise ValueError(f"actor {actor_index} 缺少 pose")
        poses = actor_poses[actor_index]
        translations = np.asarray(poses["translation_m"], dtype=np.float32)
        rotations = normalize_quaternions(poses["rotation_wxyz"])
        visible = np.asarray(poses["visibility"], dtype=bool)
        if translations.shape != (len(timestamps_us), 3) or rotations.shape != (len(timestamps_us), 4) or visible.shape != (len(timestamps_us),):
            raise ValueError(f"actor {actor_index} pose shape 非法")
        mask = assignments == actor_index
        rotation0 = quaternion_to_matrix(rotations[0])
        canonical_means = (xyz[mask] - translations[0]) @ rotation0
        canonical_quaternions = quaternion_multiply(quaternion_conjugate(rotations[0]), quaternions[mask])
        canonical_velocities = flow[mask] @ rotation0
        actor_id = f"actor_{actor_index:04d}"
        frame_id = f"actor_{actor_index:04d}_canonical"
        chunk_id = f"recondrive_actor_{actor_index:04d}"
        document["frames"].append({"id": frame_id, "frame_type": "actor"})
        trajectory = []
        visibility_rows = []
        for time_index, timestamp_us in enumerate(timestamps_us):
            document["transforms"].append(
                {
                    "name": f"T_world_{frame_id}",
                    "src_frame": frame_id,
                    "dst_frame": "world",
                    "timestamp_us": int(timestamp_us),
                    "translation_m": translations[time_index].astype(float).tolist(),
                    "rotation_wxyz": rotations[time_index].astype(float).tolist(),
                }
            )
            trajectory.append({"transform_name": f"T_world_{frame_id}", "timestamp_us": int(timestamp_us)})
            visibility_rows.append({"timestamp_us": int(timestamp_us), "visible": bool(visible[time_index])})
        document["actors"].append(
            {
                "id": actor_id,
                "class": str(poses.get("class", "unknown")),
                "canonical_representation": "gaussian_splat",
                "canonical_frame": frame_id,
                "chunk_ids": [chunk_id],
                "trajectory": trajectory,
                "visibility": visibility_rows,
            }
        )
        actor_values = _gaussian_arrays(
            means=canonical_means,
            scales=scales[mask],
            quaternions=canonical_quaternions,
            opacities=opacities[mask],
            features_dc=features_dc[mask],
            features_rest=features_rest[mask],
            source_indices=np.nonzero(mask)[0],
            velocities=canonical_velocities,
        )
        _append_chunk(
            document,
            arrays,
            chunk_id=chunk_id,
            role="actor",
            frame_id=frame_id,
            actor_id=actor_id,
            values=actor_values,
        )
    return document, arrays


def native_recondrive_view(outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """把 ReconDrive 标准输出投影到前端中立 Gaussian runtime view。"""
    sh = np.asarray(outputs["sh_maps"], dtype=np.float32)
    return {
        "means_m": np.asarray(outputs["xyz"], dtype=np.float32),
        "scales_m": np.asarray(outputs["scale_maps"], dtype=np.float32),
        "quaternions_wxyz": normalize_quaternions(outputs["rot_maps"]),
        "opacities": np.asarray(outputs["opacity_maps"], dtype=np.float32),
        "features_dc": sh[:, :, 0],
        "features_rest": np.transpose(sh[:, :, 1:], (0, 2, 1)),
        "velocities_mps": np.asarray(outputs["forward_flow"], dtype=np.float32),
        "source_indices": np.arange(sh.shape[0], dtype=np.int64),
    }


def native_streetgs_views(checkpoint: Mapping[str, Any], timestamp_index: int) -> dict[str, dict[str, np.ndarray]]:
    """复刻 StreetGS 受支持 core 的 native physical/runtime view。"""
    models = checkpoint["models"]
    background = models["Background"]
    rigid = models["RigidNodes"]
    static = _gaussian_arrays(
        means=_numpy(background["_means"]),
        scales=np.exp(_numpy(background["_scales"])),
        quaternions=_numpy(background["_quats"]),
        opacities=_sigmoid(_numpy(background["_opacities"])),
        features_dc=_numpy(background["_features_dc"]),
        features_rest=_numpy(background["_features_rest"]),
        source_indices=np.arange(_numpy(background["_means"]).shape[0], dtype=np.int64),
    )
    actor_ids = _numpy(rigid["points_ids"])[..., 0].astype(np.int64)
    poses_q = normalize_quaternions(_numpy(rigid["instances_quats"]))[timestamp_index]
    poses_t = _numpy(rigid["instances_trans"]).astype(np.float32)[timestamp_index]
    canonical_means = _numpy(rigid["_means"]).astype(np.float32)
    canonical_q = normalize_quaternions(_numpy(rigid["_quats"]))
    per_point_q = poses_q[actor_ids]
    per_point_t = poses_t[actor_ids]
    rotations = quaternion_to_matrix(per_point_q)
    world_means = np.einsum("nij,nj->ni", rotations, canonical_means) + per_point_t
    dynamic = _gaussian_arrays(
        means=world_means,
        scales=np.exp(_numpy(rigid["_scales"])),
        quaternions=quaternion_multiply(per_point_q, canonical_q),
        opacities=_sigmoid(_numpy(rigid["_opacities"])),
        features_dc=_numpy(rigid["_features_dc"]),
        features_rest=_numpy(rigid["_features_rest"]),
        source_indices=np.arange(canonical_means.shape[0], dtype=np.int64),
    )
    return {"static": static, "dynamic": dynamic}
