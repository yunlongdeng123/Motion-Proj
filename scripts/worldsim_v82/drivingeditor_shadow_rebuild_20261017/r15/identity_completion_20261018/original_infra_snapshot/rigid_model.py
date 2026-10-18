"""独立刚体实例高斯：局部参数 + 实测轨迹，不叠加 PVG 时间/速度运动。"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


SH_C0 = 0.28209479177387814
_POINT_PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
_POINT_BUFFERS = ("point_ids", "color_known", "gaussian_ids", "parent_gaussian_ids",
                  "max_radii2D", "xyz_gradient_accum", "denom")


def normalize_quaternion(q: Tensor) -> Tensor:
    """四元数顺序始终为 wxyz；全零参数安全回退到单位旋转。"""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(norm > 1e-12, q / norm.clamp_min(1e-12), identity)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    w, x, y, z = normalize_quaternion(q).unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
                       dim=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """选择数值最稳定的四元数分量，供固定观测旋转初始化使用。"""
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    magnitude = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                             1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1).clamp_min(0).sqrt()
    candidates = torch.stack((
        torch.stack((magnitude[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, magnitude[..., 1].square(), m01 + m10, m02 + m20), -1),
        torch.stack((m02 - m20, m01 + m10, magnitude[..., 2].square(), m12 + m21), -1),
        torch.stack((m10 - m01, m02 + m20, m12 + m21, magnitude[..., 3].square()), -1),
    ), dim=-2) / (2 * magnitude.clamp_min(0.1)[..., :, None])
    choice = magnitude.argmax(-1)
    q = candidates.gather(-2, choice[..., None, None].expand(choice.shape + (1, 4))).squeeze(-2)
    q = normalize_quaternion(q)
    return torch.where(q[..., :1] < 0, -q, q)


def axis_angle_to_quaternion(vector: Tensor) -> Tensor:
    """sinc 形式在零残差处有有限梯度。"""
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * math.pi)) * vector), -1)


def quaternion_slerp(a: Tensor, b: Tensor, weight: Tensor) -> Tensor:
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(0, 1)
    # acos 的端点梯度未定义；近单位旋转显式采用归一化线性插值。
    angle = torch.acos(dot.clamp_max(1 - 1e-6))
    denominator = torch.sin(angle).clamp_min(1e-8)
    curved = (torch.sin((1 - weight) * angle) * a + torch.sin(weight * angle) * b) / denominator
    linear = (1 - weight) * a + weight * b
    return normalize_quaternion(torch.where(dot > 0.9995, linear, curved))


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _require_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class WorldGaussians:
    """某个查询时刻的有效实例视图；row_ids 指向当前打包参数的行。"""

    xyz: Tensor
    scaling: Tensor
    rotation: Tensor
    opacity: Tensor
    features: Tensor
    point_ids: Tensor
    row_ids: Tensor
    object_rotation: Tensor
    object_translation: Tensor
    active_sh_degree: int

    @property
    def world_from_local(self) -> Tensor:
        result = self.xyz.new_zeros((self.xyz.shape[0], 4, 4))
        result[:, :3, :3] = quaternion_to_matrix(self.object_rotation)
        result[:, :3, 3] = self.object_translation
        result[:, 3, 3] = 1
        return result

    def local_view_directions(self, camera_center: Tensor | Sequence[float]) -> Tensor:
        """SH 方向采用物体局部坐标，方向约定为相机指向高斯。"""
        center = torch.as_tensor(camera_center, device=self.xyz.device, dtype=self.xyz.dtype)
        if center.shape != (3,):
            raise ValueError("camera_center must have shape [3]")
        direction = self.xyz - center
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("nij,nj->ni", quaternion_to_matrix(self.object_rotation).transpose(-1, -2), direction)

    def covariance(self, scaling_modifier: float = 1.0, packed: bool = True) -> Tensor:
        rotation = quaternion_to_matrix(self.rotation)
        factor = rotation * (self.scaling * scaling_modifier)[:, None, :]
        covariance = factor @ factor.transpose(-1, -2)
        if not packed:
            return covariance
        return torch.stack((covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
                            covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]), -1)

    get_xyz = property(lambda self: self.xyz)
    get_scaling = property(lambda self: self.scaling)
    get_rotation = property(lambda self: self.rotation)
    get_opacity = property(lambda self: self.opacity)
    get_features = property(lambda self: self.features)
    get_max_sh_channels = property(lambda self: self.features.shape[1])

    def get_covariance(self, scaling_modifier: float = 1.0) -> Tensor:
        return self.covariance(scaling_modifier, packed=True)

    def get_xyz_SHM(self, timestamp: Any = None, times: float = 1.0) -> Tensor:
        # 兼容旧渲染接口，但运动已经由对象轨迹完成，不能再加 PVG 速度。
        return self.xyz

    def get_marginal_t(self, timestamp: Any = None) -> Tensor:
        return torch.ones_like(self.opacity)


class RigidInstanceGaussianModel(nn.Module):
    """局部高斯及每条轨迹的独立状态，默认轨迹固定。

    timestamps 是预处理声明的模型时间；不隐式改成物理秒。缺失帧不补轨迹，
    仅两端 valid 的相邻帧可插值。编辑旋转以当前物体中心为轴，平移是世界偏移。
    clone_points/prune_points 需要传入外部优化器，或使用 make_optimizer 创建的优化器。
    """

    checkpoint_version = 1

    def __init__(
        self, local_xyz: Any, rgb: Any, scale: Any, point_ids: Any,
        instance_ids: Sequence[str], timestamps: Any, poses: Any, valid: Any,
        *, sh_degree: int = 3, initial_opacity: float = 0.01,
        refine_poses: bool = False, pose_translation_limit: float = 0.5,
        pose_rotation_limit: float = math.radians(10), interpolation: str = "linear",
        time_tolerance: float = 1e-6, box_size: Any = None, color_known: Any = None,
        physical_timestamps: Any = None, raw_timestamp_micros: Any = None,
        device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if sh_degree < 0 or int(sh_degree) != sh_degree:
            raise ValueError("sh_degree must be a nonnegative integer")
        if interpolation not in ("linear", "exact"):
            raise ValueError("interpolation must be linear or exact")
        if not 0 < initial_opacity < 1:
            raise ValueError("initial_opacity must lie strictly between 0 and 1")
        if (not all(math.isfinite(value) for value in (pose_translation_limit, pose_rotation_limit, time_tolerance))
                or pose_translation_limit <= 0 or pose_rotation_limit <= 0 or time_tolerance < 0):
            raise ValueError("pose limits must be positive and time_tolerance nonnegative")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.instance_ids = tuple(str(value) for value in instance_ids)
        if not self.instance_ids or len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("instance_ids must be nonempty and unique")
        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = 0
        self.refine_poses = bool(refine_poses)
        self.pose_translation_limit = float(pose_translation_limit)
        self.pose_rotation_limit = float(pose_rotation_limit)
        self.interpolation = interpolation
        self.time_tolerance = float(time_tolerance)
        self.has_box_size = box_size is not None
        self.optimizer: torch.optim.Optimizer | None = None

        def floating(value: Any) -> Tensor:
            return torch.as_tensor(value, device=device, dtype=dtype).detach().clone()

        xyz, colors, scales = floating(local_xyz), floating(rgb), floating(scale)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or colors.shape != xyz.shape or scales.shape != xyz.shape:
            raise ValueError("local_xyz, rgb and scale must all have shape [N,3]")
        for name, value in (("local_xyz", xyz), ("rgb", colors), ("scale", scales)):
            _require_finite(value, name)
        if torch.any((colors < 0) | (colors > 1)) or torch.any(scales <= 0):
            raise ValueError("RGB must lie in [0,1] and scale must be positive")
        n, k = xyz.shape[0], len(self.instance_ids)
        ids_input = torch.as_tensor(point_ids, device=device)
        if ids_input.shape != (n,) or torch.any(ids_input != ids_input.long()):
            raise ValueError("point_ids must contain N integer instance indices")
        ids = ids_input.long().clone()
        if torch.any((ids < 0) | (ids >= k)):
            raise ValueError("point_ids must be contiguous instance-table indices in [0,K)")
        times = torch.as_tensor(timestamps, device=device, dtype=torch.float64).detach().clone()
        if times.ndim != 1 or times.numel() == 0:
            raise ValueError("timestamps must be a nonempty vector")
        _require_finite(times, "timestamps")
        if torch.any(times[1:] <= times[:-1]):
            raise ValueError("timestamps must be strictly increasing")
        count = times.numel()
        trajectory, validity = floating(poses), torch.as_tensor(valid, device=device, dtype=torch.bool).clone()
        if trajectory.shape != (count, k, 4, 4) or validity.shape != (count, k):
            raise ValueError("poses must have shape [T,K,4,4] and valid [T,K]")
        observed = trajectory[validity]
        _require_finite(observed, "valid poses")
        if observed.numel():
            rotations = observed[:, :3, :3]
            unit = torch.eye(3, device=device, dtype=dtype).expand_as(rotations)
            expected_bottom = floating([0, 0, 0, 1]).expand(observed.shape[0], -1)
            if not torch.allclose(rotations.transpose(-1, -2) @ rotations, unit, atol=2e-4, rtol=2e-4):
                raise ValueError("valid pose rotations must be orthonormal")
            if not torch.allclose(torch.linalg.det(rotations), torch.ones_like(rotations[:, 0, 0]), atol=2e-4, rtol=2e-4):
                raise ValueError("valid pose rotations must have determinant +1")
            if not torch.allclose(observed[:, 3], expected_bottom, atol=1e-6, rtol=0):
                raise ValueError("valid poses must be rigid homogeneous transforms")
        # 无观测位置统一成单位变换，防止缺失标注中的 NaN 污染后续计算。
        identity_pose = torch.eye(4, device=device, dtype=dtype).expand_as(trajectory)
        trajectory = torch.where(validity[..., None, None], trajectory, identity_pose)
        self.register_buffer("timestamps", times)
        self.register_buffer("base_poses", trajectory)
        self.register_buffer("base_rotation", matrix_to_quaternion(trajectory[:, :, :3, :3]))
        self.register_buffer("pose_valid", validity)
        self.register_buffer("point_ids", ids)
        self.register_buffer("box_size", floating(box_size) if box_size is not None else xyz.new_zeros((count, k, 3)))
        if self.box_size.shape != (count, k, 3):
            raise ValueError("box_size must have shape [T,K,3]")
        if box_size is not None:
            _require_finite(self.box_size[validity], "valid box_size")
            if torch.any(self.box_size[validity] <= 0):
                raise ValueError("observed box sizes must be positive")
        self.register_buffer("color_known", torch.as_tensor(color_known, device=device, dtype=torch.bool).clone()
                             if color_known is not None else torch.ones(n, device=device, dtype=torch.bool))
        if self.color_known.shape != (n,):
            raise ValueError("color_known must have shape [N]")
        self.register_buffer("physical_timestamps", torch.as_tensor(physical_timestamps, device=device, dtype=torch.float64).clone()
                             if physical_timestamps is not None else times.new_empty(0))
        self.register_buffer("raw_timestamp_micros", torch.as_tensor(raw_timestamp_micros, device=device, dtype=torch.long).clone()
                             if raw_timestamp_micros is not None else torch.empty(0, device=device, dtype=torch.long))
        for name in ("physical_timestamps", "raw_timestamp_micros"):
            if getattr(self, name).shape not in ((0,), (count,)):
                raise ValueError(f"{name} must contain T entries when provided")
        _require_finite(self.physical_timestamps, "physical_timestamps")
        self.register_buffer("deleted", torch.zeros(k, device=device, dtype=torch.bool))
        self.register_buffer("replacement_source", torch.full((k,), -1, device=device, dtype=torch.long))
        self.register_buffer("edit_translation", xyz.new_zeros((k, 3)))
        edit_rotation = xyz.new_zeros((k, 4))
        edit_rotation[:, 0] = 1
        self.register_buffer("edit_rotation", edit_rotation)
        self.register_buffer("gaussian_ids", torch.arange(n, device=device))
        self.register_buffer("parent_gaussian_ids", torch.arange(n, device=device))
        self.register_buffer("next_gaussian_id", torch.tensor(n, device=device, dtype=torch.long))
        self.register_buffer("max_radii2D", xyz.new_zeros(n))
        self.register_buffer("xyz_gradient_accum", xyz.new_zeros((n, 1)))
        self.register_buffer("denom", xyz.new_zeros((n, 1)))
        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter(((colors - 0.5) / SH_C0)[:, None, :])
        self._features_rest = nn.Parameter(xyz.new_zeros((n, (self.max_sh_degree + 1) ** 2 - 1, 3)))
        self._scaling = nn.Parameter(torch.log(scales))
        local_rotation = xyz.new_zeros((n, 4))
        local_rotation[:, 0] = 1
        self._rotation = nn.Parameter(local_rotation)
        self._opacity = nn.Parameter(xyz.new_full((n, 1), math.log(initial_opacity / (1 - initial_opacity))))
        self.pose_translation_raw = nn.Parameter(xyz.new_zeros((count, k, 3)), requires_grad=refine_poses)
        self.pose_rotation_raw = nn.Parameter(xyz.new_zeros((count, k, 3)), requires_grad=refine_poses)

    @property
    def track_ids(self) -> tuple[str, ...]:
        return self.instance_ids

    @property
    def local_xyz(self) -> Tensor:
        return self._xyz

    @property
    def valid(self) -> Tensor:
        return self.pose_valid

    @property
    def num_points(self) -> int:
        return self._xyz.shape[0]

    @property
    def num_instances(self) -> int:
        return len(self.instance_ids)

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any], **kwargs: Any) -> "RigidInstanceGaussianModel":
        def value(*names: str) -> Any:
            for name in names:
                if name in bundle:
                    return bundle[name]
            raise KeyError(f"bundle requires one of {names}")

        scale = bundle["scale"] if "scale" in bundle else bundle.get("scaling")
        if scale is None:
            scale = torch.as_tensor(value("log_scaling", "_scaling")).exp()
        optional = {key: bundle[key] for key in ("box_size", "color_known", "physical_timestamps", "raw_timestamp_micros")
                    if key in bundle and key not in kwargs}
        return cls(value("local_xyz", "xyz_local"), value("rgb", "colors"), scale,
                   value("point_ids", "instance_indices"), value("instance_ids", "track_ids"),
                   value("timestamps", "times"), value("poses", "world_from_local"),
                   value("valid", "pose_valid"), **optional, **kwargs)

    @classmethod
    def from_npz(cls, path: str | Path, **kwargs: Any) -> "RigidInstanceGaussianModel":
        import numpy as np

        with np.load(path, allow_pickle=False) as bundle:
            return cls.from_bundle(bundle, **kwargs)

    def _instance_index(self, instance: str | int) -> int:
        if isinstance(instance, str):
            try:
                return self.instance_ids.index(instance)
            except ValueError as error:
                raise KeyError(f"unknown instance {instance!r}") from error
        index = int(instance)
        if index != instance or not 0 <= index < self.num_instances:
            raise KeyError(f"unknown instance index {instance!r}")
        return index

    @torch.no_grad()
    def set_deleted(self, instance: str | int, deleted: bool = True) -> None:
        self.deleted[self._instance_index(instance)] = bool(deleted)

    @torch.no_grad()
    def replace(self, target: str | int, source: str | int | None) -> None:
        """复用 source 的原始局部几何/外观，保留 target 轨迹及渲染归属。"""
        target_index = self._instance_index(target)
        source_index = -1 if source is None else self._instance_index(source)
        if source_index >= 0 and not (self.point_ids == source_index).any():
            raise ValueError("replacement source has no initialized Gaussian points")
        self.replacement_source[target_index] = source_index

    @torch.no_grad()
    def translate(self, instance: str | int, translation: Any, *, relative: bool = True) -> None:
        index = self._instance_index(instance)
        delta = torch.as_tensor(translation, device=self._xyz.device, dtype=self._xyz.dtype)
        if delta.shape != (3,):
            raise ValueError("translation must have shape [3]")
        _require_finite(delta, "translation")
        self.edit_translation[index] = self.edit_translation[index] + delta if relative else delta

    def _edit_quaternion(self, quaternion: Any = None, yaw_radians: float | None = None) -> Tensor:
        if (quaternion is None) == (yaw_radians is None):
            raise ValueError("provide exactly one of quaternion or yaw_radians")
        if quaternion is None:
            quaternion = [math.cos(yaw_radians / 2), 0, 0, math.sin(yaw_radians / 2)]
        q = torch.as_tensor(quaternion, device=self._xyz.device, dtype=self._xyz.dtype)
        if q.shape != (4,):
            raise ValueError("quaternion must have shape [4] in wxyz order")
        _require_finite(q, "quaternion")
        if torch.linalg.vector_norm(q) < 1e-12:
            raise ValueError("edit quaternion cannot be zero")
        return normalize_quaternion(q)

    @torch.no_grad()
    def rotate(self, instance: str | int, quaternion: Any = None, *,
               yaw_radians: float | None = None, relative: bool = True) -> None:
        index = self._instance_index(instance)
        q = self._edit_quaternion(quaternion, yaw_radians)
        self.edit_rotation[index] = normalize_quaternion(quaternion_multiply(q, self.edit_rotation[index])) if relative else q

    @torch.no_grad()
    def reset_edits(self, instance: str | int | None = None) -> None:
        indices = slice(None) if instance is None else self._instance_index(instance)
        self.deleted[indices] = False
        self.replacement_source[indices] = -1
        self.edit_translation[indices] = 0
        self.edit_rotation[indices] = 0
        self.edit_rotation[indices, 0] = 1

    def enable_pose_refinement(self, enabled: bool = True) -> None:
        self.refine_poses = bool(enabled)
        self.pose_translation_raw.requires_grad_(enabled)
        self.pose_rotation_raw.requires_grad_(enabled)

    def bounded_pose_residuals(self) -> tuple[Tensor, Tensor]:
        def bounded(raw: Tensor, limit: float) -> Tensor:
            return limit * raw / torch.sqrt(1 + raw.square().sum(-1, keepdim=True))

        return (bounded(self.pose_translation_raw, self.pose_translation_limit),
                bounded(self.pose_rotation_raw, self.pose_rotation_limit))

    def _frame_pose(self, frame: int) -> tuple[Tensor, Tensor]:
        translation, rotation = self.bounded_pose_residuals()
        position = self.base_poses[frame, :, :3, 3] + translation[frame]
        orientation = quaternion_multiply(axis_angle_to_quaternion(rotation[frame]), self.base_rotation[frame])
        return position, normalize_quaternion(orientation)

    def trajectory_at(self, timestamp: float | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """返回每个实例的平移、wxyz 旋转与实际有效标记。"""
        time = torch.as_tensor(timestamp, device=self.timestamps.device, dtype=self.timestamps.dtype)
        if time.numel() != 1 or not torch.isfinite(time).all():
            raise ValueError("timestamp must be one finite scalar")
        time = time.reshape(())
        right = int(torch.searchsorted(self.timestamps, time).item())
        candidates = [index for index in (right - 1, right) if 0 <= index < self.timestamps.numel()]
        closest = min(candidates, key=lambda index: abs(float((self.timestamps[index] - time).item())))
        if abs(float((self.timestamps[closest] - time).item())) <= self.time_tolerance:
            position, orientation = self._frame_pose(closest)
            return position, orientation, self.pose_valid[closest].clone()
        if self.interpolation == "linear" and 0 < right < self.timestamps.numel():
            left = right - 1
            weight = ((time - self.timestamps[left]) / (self.timestamps[right] - self.timestamps[left])).to(self._xyz.dtype)
            left_position, left_rotation = self._frame_pose(left)
            right_position, right_rotation = self._frame_pose(right)
            return ((1 - weight) * left_position + weight * right_position,
                    quaternion_slerp(left_rotation, right_rotation, weight),
                    self.pose_valid[left] & self.pose_valid[right])
        position, orientation = self._frame_pose(closest)
        return position, orientation, torch.zeros(self.num_instances, device=self._xyz.device, dtype=torch.bool)

    def evaluate(self, timestamp: float | Tensor, edits: Mapping[str | int, Mapping[str, Any]] | None = None) -> WorldGaussians:
        position, orientation, valid = self.trajectory_at(timestamp)
        position = position + self.edit_translation
        orientation = quaternion_multiply(self.edit_rotation, orientation)
        valid = valid & ~self.deleted
        replacements = self.replacement_source
        if edits:
            for instance, specification in edits.items():
                index = self._instance_index(instance)
                unknown = set(specification) - {"delete", "translation", "quaternion", "yaw_radians", "replacement"}
                if unknown:
                    raise ValueError(f"unknown edit keys: {sorted(unknown)}")
                if specification.get("delete", False):
                    valid = valid.clone()
                    valid[index] = False
                if "translation" in specification:
                    delta = torch.as_tensor(specification["translation"], device=self._xyz.device, dtype=self._xyz.dtype)
                    if delta.shape != (3,):
                        raise ValueError("translation must have shape [3]")
                    _require_finite(delta, "translation")
                    position = position.clone()
                    position[index] = position[index] + delta
                if "quaternion" in specification or "yaw_radians" in specification:
                    q = self._edit_quaternion(specification.get("quaternion"), specification.get("yaw_radians"))
                    orientation = orientation.clone()
                    orientation[index] = quaternion_multiply(q, orientation[index])
                if "replacement" in specification:
                    source = specification["replacement"]
                    source_index = -1 if source is None else self._instance_index(source)
                    if source_index >= 0 and not (self.point_ids == source_index).any():
                        raise ValueError("replacement source has no initialized Gaussian points")
                    replacements = replacements.clone()
                    replacements[index] = source_index
        rows = torch.nonzero(valid[self.point_ids] & (replacements[self.point_ids] < 0), as_tuple=False).flatten()
        ids = self.point_ids[rows]
        replacement_targets = torch.nonzero(valid & (replacements >= 0), as_tuple=False).flatten()
        # row_ids 仍指向 source 参数，所以替换产生的梯度回到真实源高斯；point_ids 属于 target。
        row_chunks, id_chunks = [rows], [ids]
        for target_index in replacement_targets.tolist():
            source_rows = torch.nonzero(self.point_ids == replacements[target_index], as_tuple=False).flatten()
            row_chunks.append(source_rows)
            id_chunks.append(torch.full_like(source_rows, target_index))
        rows, ids = torch.cat(row_chunks), torch.cat(id_chunks)
        object_rotation = normalize_quaternion(orientation[ids])
        object_translation = position[ids]
        world_xyz = torch.einsum("nij,nj->ni", quaternion_to_matrix(object_rotation), self._xyz[rows]) + object_translation
        rotation = normalize_quaternion(quaternion_multiply(object_rotation, normalize_quaternion(self._rotation[rows])))
        features = torch.cat((self._features_dc[rows], self._features_rest[rows]), dim=1)
        return WorldGaussians(world_xyz, self._scaling[rows].exp(), rotation, self._opacity[rows].sigmoid(),
                              features, ids, rows, object_rotation, object_translation, self.active_sh_degree)

    forward = evaluate

    def oneupSHdegree(self) -> None:
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    def pose_regularization(self, *, translation_weight: float = 1.0, rotation_weight: float = 1.0,
                            smoothness_weight: float = 1.0) -> dict[str, Tensor]:
        translation, rotation = self.bounded_pose_residuals()
        valid = self.pose_valid
        zero = (translation.sum() + rotation.sum()) * 0

        def prior(value: Tensor, limit: float) -> Tensor:
            return value[valid].square().mean() / (limit * limit) if valid.any() else zero

        adjacent = valid[1:] & valid[:-1]
        intervals = (self.timestamps[1:] - self.timestamps[:-1]).to(self._xyz.dtype)[:, None, None]

        def smooth(value: Tensor, limit: float) -> Tensor:
            velocity = (value[1:] - value[:-1]) / intervals
            return velocity[adjacent].square().mean() / (limit * limit) if adjacent.any() else zero

        terms = {"translation_prior": prior(translation, self.pose_translation_limit),
                 "rotation_prior": prior(rotation, self.pose_rotation_limit),
                 "translation_smoothness": smooth(translation, self.pose_translation_limit),
                 "rotation_smoothness": smooth(rotation, self.pose_rotation_limit)}
        terms["total"] = (translation_weight * terms["translation_prior"] + rotation_weight * terms["rotation_prior"]
                          + smoothness_weight * (terms["translation_smoothness"] + terms["rotation_smoothness"]))
        return terms

    def make_optimizer(self, learning_rates: Mapping[str, float] | None = None, *,
                       lr: float = 1e-3, pose_lr: float = 1e-4, **adam_options: Any) -> torch.optim.Adam:
        rates = dict(learning_rates or {})
        names = {"_xyz": "xyz", "_features_dc": "f_dc", "_features_rest": "f_rest",
                 "_scaling": "scaling", "_rotation": "rotation", "_opacity": "opacity",
                 "pose_translation_raw": "pose_translation", "pose_rotation_raw": "pose_rotation"}
        groups = [{"params": [parameter], "name": names[name],
                   "lr": rates.get(names[name], pose_lr if name.startswith("pose_") else lr)}
                  for name, parameter in self.named_parameters() if parameter.requires_grad]
        unknown = set(rates) - set(names.values())
        if unknown:
            raise ValueError(f"unknown learning-rate groups: {sorted(unknown)}")
        self.optimizer = torch.optim.Adam(groups, lr=lr, **adam_options)
        return self.optimizer

    def _replace_point_rows(self, indices: Tensor, retained_count: int,
                            optimizer: torch.optim.Optimizer | None) -> None:
        """保留行继承 Adam 动量；新克隆行从零动量开始。"""
        optimizer = optimizer if optimizer is not None else self.optimizer
        old_count = self.num_points
        for name in _POINT_PARAMETERS:
            old = getattr(self, name)
            new = nn.Parameter(old.detach().index_select(0, indices), requires_grad=old.requires_grad)
            if optimizer is not None:
                old_state = optimizer.state.pop(old, None)
                found = False
                for group in optimizer.param_groups:
                    for offset, parameter in enumerate(group["params"]):
                        if parameter is old:
                            group["params"][offset] = new
                            found = True
                if old_state is not None and not found:
                    raise ValueError(f"optimizer state for {name} has no matching parameter group")
                if old_state is not None:
                    new_state = {}
                    for key, value in old_state.items():
                        if isinstance(value, Tensor) and value.ndim > 0 and value.shape == old.shape:
                            replacement = value.index_select(0, indices)
                            replacement[retained_count:] = 0
                            new_state[key] = replacement
                        else:
                            new_state[key] = value.clone() if isinstance(value, Tensor) else copy.deepcopy(value)
                    optimizer.state[new] = new_state
            setattr(self, name, new)
        for name in _POINT_BUFFERS:
            old = getattr(self, name)
            if old.shape[0] != old_count:
                raise RuntimeError(f"point buffer {name} does not match point count")
            setattr(self, name, old.index_select(0, indices))
        for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
            getattr(self, name)[retained_count:] = 0

    @torch.no_grad()
    def clone_points(self, indices: Any, *, offsets: Any = None,
                     optimizer: torch.optim.Optimizer | None = None) -> Tensor:
        source = torch.as_tensor(indices, device=self._xyz.device)
        if source.ndim != 1 or torch.any(source != source.long()):
            raise ValueError("clone indices must be a vector of integer rows")
        source = source.long()
        if torch.any((source < 0) | (source >= self.num_points)):
            raise IndexError("clone row index out of range")
        count, start = source.numel(), self.num_points
        delta = self._xyz.new_zeros((count, 3)) if offsets is None else torch.as_tensor(offsets, device=self._xyz.device, dtype=self._xyz.dtype)
        if delta.shape != (count, 3):
            raise ValueError("clone offsets must have shape [number of clones,3]")
        _require_finite(delta, "clone offsets")
        if count == 0:
            return source
        parents = self.gaussian_ids[source].clone()
        indices_all = torch.cat((torch.arange(start, device=self._xyz.device), source))
        self._replace_point_rows(indices_all, start, optimizer)
        self._xyz[start:] += delta
        first_id = int(self.next_gaussian_id.item())
        self.gaussian_ids[start:] = torch.arange(first_id, first_id + count, device=self._xyz.device)
        self.parent_gaussian_ids[start:] = parents
        self.next_gaussian_id += count
        return torch.arange(start, start + count, device=self._xyz.device)

    @torch.no_grad()
    def prune_points(self, remove_mask: Any, *, optimizer: torch.optim.Optimizer | None = None) -> Tensor:
        mask = torch.as_tensor(remove_mask, device=self._xyz.device)
        if mask.dtype != torch.bool or mask.shape != (self.num_points,):
            raise ValueError("remove_mask must be a bool vector with one entry per point")
        retained = torch.nonzero(~mask, as_tuple=False).flatten()
        if not mask.any():
            return retained
        self._replace_point_rows(retained, retained.numel(), optimizer)
        return retained

    @torch.no_grad()
    def accumulate_densification_stats(self, row_ids: Any, gradients: Any, *,
                                       visible: Any = None, radii: Any = None) -> None:
        """同一 source 被替换复用时，重复 row_ids 的梯度和计数都累加。"""
        ids = torch.as_tensor(row_ids, device=self._xyz.device)
        if ids.ndim != 1 or torch.any(ids != ids.long()):
            raise ValueError("row_ids must be an integer vector")
        ids = ids.long()
        if torch.any((ids < 0) | (ids >= self.num_points)):
            raise IndexError("densification row index out of range")
        gradient = torch.as_tensor(gradients, device=self._xyz.device, dtype=self._xyz.dtype)
        if gradient.ndim != 2 or gradient.shape[0] != ids.numel() or gradient.shape[1] not in (2, 3):
            raise ValueError("gradients must have shape [number of rendered rows,2 or 3]")
        mask = torch.ones_like(ids, dtype=torch.bool) if visible is None else torch.as_tensor(visible, device=self._xyz.device)
        if mask.dtype != torch.bool or mask.shape != ids.shape:
            raise ValueError("visible must be a bool vector matching row_ids")
        _require_finite(gradient[mask], "visible screen-space gradients")
        selected = ids[mask]
        magnitude = torch.linalg.vector_norm(gradient[mask, :2], dim=-1, keepdim=True)
        radius = None
        if radii is not None:
            radius = torch.as_tensor(radii, device=self._xyz.device, dtype=self._xyz.dtype)
            if radius.shape != ids.shape:
                raise ValueError("radii must match row_ids")
            _require_finite(radius[mask], "visible radii")
        self.xyz_gradient_accum.index_add_(0, selected, magnitude)
        self.denom.index_add_(0, selected, torch.ones_like(magnitude))
        if radius is not None:
            self.max_radii2D.scatter_reduce_(0, selected, radius[mask], reduce="amax", include_self=True)

    @torch.no_grad()
    def densify_and_prune(self, gradient_threshold: float, min_opacity: float, *,
                          max_screen_radius: float | None = None, max_world_scale: float | None = None,
                          max_new_points: int | None = None, optimizer: torch.optim.Optimizer | None = None) -> dict[str, int]:
        """保守的克隆/剪枝基线；不伪装成已实现各向异性 split 或时间增密。"""
        if not math.isfinite(gradient_threshold) or gradient_threshold < 0 or not 0 <= min_opacity <= 1:
            raise ValueError("gradient_threshold must be finite/nonnegative and min_opacity in [0,1]")
        if max_new_points is not None and (int(max_new_points) != max_new_points or max_new_points < 0):
            raise ValueError("max_new_points must be a nonnegative integer")
        if max_new_points is not None:
            max_new_points = int(max_new_points)
        remove = self._opacity.sigmoid().flatten() < min_opacity
        if max_screen_radius is not None:
            if not math.isfinite(max_screen_radius) or max_screen_radius <= 0:
                raise ValueError("max_screen_radius must be finite and positive")
            remove |= self.max_radii2D > max_screen_radius
        if max_world_scale is not None:
            if not math.isfinite(max_world_scale) or max_world_scale <= 0:
                raise ValueError("max_world_scale must be finite and positive")
            remove |= self._scaling.exp().amax(-1) > max_world_scale
        average = (self.xyz_gradient_accum / self.denom.clamp_min(1)).flatten()
        selected = torch.nonzero((average >= gradient_threshold) & (self.denom.flatten() > 0) & ~remove,
                                 as_tuple=False).flatten()
        if max_new_points is not None and selected.numel() > max_new_points:
            selected = selected[torch.argsort(average[selected], descending=True, stable=True)[:max_new_points]]
        cloned, pruned = selected.numel(), int(remove.sum().item())
        self.clone_points(selected, optimizer=optimizer)
        self.prune_points(torch.cat((remove, torch.zeros(cloned, device=self._xyz.device, dtype=torch.bool))), optimizer=optimizer)
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()
        return {"cloned": cloned, "pruned": pruned, "points": self.num_points}

    def capture(self, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        optimizer = optimizer if optimizer is not None else self.optimizer
        result = {"version": self.checkpoint_version, "instance_ids": self.instance_ids,
                  "config": {"sh_degree": self.max_sh_degree, "refine_poses": self.refine_poses,
                             "pose_translation_limit": self.pose_translation_limit,
                             "pose_rotation_limit": self.pose_rotation_limit, "interpolation": self.interpolation,
                             "time_tolerance": self.time_tolerance}, "active_sh_degree": self.active_sh_degree,
                  "training": self.training, "state_dict": _cpu_copy(self.state_dict()), "optimizer": None}
        result["box_size_available"] = self.has_box_size
        if optimizer is not None:
            if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
                raise TypeError("checkpoint optimizer restoration supports Adam and AdamW")
            by_identity = {id(parameter): name for name, parameter in self.named_parameters()}
            groups = []
            for group in optimizer.param_groups:
                if any(id(parameter) not in by_identity for parameter in group["params"]):
                    raise ValueError("capture requires an instance-only optimizer; save mixed optimizers in the training checkpoint")
                groups.append({**_cpu_copy({key: value for key, value in group.items() if key != "params"}),
                               "parameter_names": [by_identity[id(parameter)] for parameter in group["params"]]})
            result["optimizer"] = {"class": type(optimizer).__name__, "groups": groups,
                                   "state_dict": _cpu_copy(optimizer.state_dict())}
        return result

    @classmethod
    def from_capture(cls, checkpoint: Mapping[str, Any], *, device: str | torch.device = "cpu",
                     restore_optimizer: bool = True) -> "RigidInstanceGaussianModel":
        if checkpoint.get("version") != cls.checkpoint_version:
            raise ValueError("unsupported rigid-instance checkpoint version")
        state = checkpoint["state_dict"]
        rgb = (state["_features_dc"][:, 0] * SH_C0 + 0.5).clamp(0, 1)
        model = cls(state["_xyz"], rgb, state["_scaling"].exp(), state["point_ids"], checkpoint["instance_ids"],
                    state["timestamps"], state["base_poses"], state["pose_valid"], device=device,
                    dtype=state["_xyz"].dtype,
                    box_size=state["box_size"] if checkpoint.get("box_size_available", True) else None,
                    color_known=state["color_known"],
                    physical_timestamps=state["physical_timestamps"] if state["physical_timestamps"].numel() else None,
                    raw_timestamp_micros=state["raw_timestamp_micros"] if state["raw_timestamp_micros"].numel() else None,
                    **checkpoint["config"])
        model.load_state_dict(state, strict=True)
        model.active_sh_degree = int(checkpoint["active_sh_degree"])
        model.train(bool(checkpoint.get("training", True)))
        saved_optimizer = checkpoint.get("optimizer")
        if saved_optimizer is not None and restore_optimizer:
            parameters = dict(model.named_parameters())
            groups = [{**{key: value for key, value in group.items() if key != "parameter_names"},
                       "params": [parameters[name] for name in group["parameter_names"]]}
                      for group in saved_optimizer["groups"]]
            optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}.get(saved_optimizer["class"])
            if optimizer_class is None:
                raise ValueError("unsupported saved optimizer")
            model.optimizer = optimizer_class(groups)
            # 同设备/同dtype时PyTorch可能复用state tensor；恢复后的Adam不能改写输入checkpoint。
            model.optimizer.load_state_dict(copy.deepcopy(saved_optimizer["state_dict"]))
        return model

    def restore(self, checkpoint: Mapping[str, Any], *, restore_optimizer: bool = True) -> torch.optim.Optimizer | None:
        """原地替换模型状态，并返回重建的优化器；调用方应更新旧优化器引用。"""
        restored = type(self).from_capture(checkpoint, device=self._xyz.device, restore_optimizer=restore_optimizer)
        self.__dict__.update(restored.__dict__)
        return self.optimizer
