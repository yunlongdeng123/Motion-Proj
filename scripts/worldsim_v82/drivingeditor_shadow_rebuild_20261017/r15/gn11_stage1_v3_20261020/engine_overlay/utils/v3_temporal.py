"""一个背景 PVG 集合的证据选择性时间正则，无参数冻结/前向改写。

固定三维证据场只决定损失权重；每次根据当前 Gaussian 位置/尺度重新查询，
clone/split/prune 无需维护可失配的逐行标签。没有证据的点保留弱先验。
"""

from pathlib import Path
import json
import math

import numpy as np
import torch


TEMPORAL_VERSION = "V3_SELECTIVE_BACKGROUND_TIME_20261020"
TEMPORAL_POLICY = {
    "voxel_m": 0.2,
    "minimum_observation_frames": 3,
    "minimum_time_span_model": 0.2,
    "object_box_padding_m": 0.15,
    "road_protection_height_m": 0.2,
    "maximum_stable_sigma_m": 0.2,
    "stable_motion_weight": 0.1,
    "stable_visibility_weight": 0.1,
    "context_motion_weight": 0.001,
    "context_visibility_weight": 0.001,
    "road_geometry_confidence": 0.5,
}
VOXEL_BITS = 20
VOXEL_OFFSET = 1 << (VOXEL_BITS - 1)


def voxel_keys_numpy(xyz, cell_size):
    coordinates = np.floor(np.asarray(xyz) / cell_size).astype(np.int64)
    if np.any(coordinates < -VOXEL_OFFSET) or np.any(coordinates >= VOXEL_OFFSET):
        raise ValueError("geometry exceeds supported voxel-coordinate extent")
    shifted = coordinates + VOXEL_OFFSET
    return ((shifted[:, 0] << (2 * VOXEL_BITS))
            | (shifted[:, 1] << VOXEL_BITS) | shifted[:, 2])


def repeated_surface_cells(keys_per_frame, timestamps, minimum_frames, minimum_span):
    """每帧每体素只记一次；同一扫描的多返回不能伪造跨时刻证据。"""
    keys = np.concatenate([np.unique(keys) for keys in keys_per_frame])
    times = np.concatenate([
        np.full(len(np.unique(keys)), time, dtype=np.float64)
        for keys, time in zip(keys_per_frame, timestamps)
    ])
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    first = np.full(len(unique), np.inf)
    last = np.full(len(unique), -np.inf)
    np.minimum.at(first, inverse, times)
    np.maximum.at(last, inverse, times)
    accepted = (counts >= minimum_frames) & (last - first >= minimum_span - 1e-8)
    return unique[accepted], counts[accepted], last[accepted] - first[accepted]


class TemporalEvidence:
    def __init__(self, metadata_path, device):
        metadata_path = Path(metadata_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (self.metadata.get("version") != TEMPORAL_VERSION
                or self.metadata.get("policy") != TEMPORAL_POLICY):
            raise ValueError("background temporal evidence policy mismatch")
        with np.load(metadata_path.parent / self.metadata["field"], allow_pickle=False) as field:
            self.keys = torch.as_tensor(field["keys"].copy(), device=device, dtype=torch.int64)
            self.confidence = torch.as_tensor(
                field["confidence"].copy(), device=device, dtype=torch.float32
            )
        if len(self.keys) == 0 or not bool((self.keys[1:] > self.keys[:-1]).all()):
            raise ValueError("background temporal field must contain sorted unique cells")
        if not torch.isfinite(self.confidence).all() or not bool(
            ((self.confidence >= 0) & (self.confidence <= 1)).all()
        ):
            raise ValueError("invalid stability confidence")
        self.scale = float(self.metadata["meter_to_model"])
        self.cell_size = self.scale * TEMPORAL_POLICY["voxel_m"]

    @torch.no_grad()
    def weights(self, xyz, scaling):
        coordinates = torch.floor(xyz.detach() / self.cell_size).to(torch.int64)
        valid = ((coordinates >= -VOXEL_OFFSET) & (coordinates < VOXEL_OFFSET)).all(1)
        shifted = (coordinates + VOXEL_OFFSET).clamp(0, 2 * VOXEL_OFFSET - 1)
        keys = ((shifted[:, 0] << (2 * VOXEL_BITS))
                | (shifted[:, 1] << VOXEL_BITS) | shifted[:, 2])
        indices = torch.searchsorted(self.keys, keys).clamp_max(len(self.keys) - 1)
        matched = valid & (self.keys[indices] == keys)
        # 大支撑不能因中心碰到一小块稳定表面而全部被判为稳定。
        compact = scaling.detach().amax(1) <= (
            self.scale * TEMPORAL_POLICY["maximum_stable_sigma_m"]
        )
        return self.confidence[indices].to(xyz) * (matched & compact).to(xyz)


def selective_temporal_loss(model, confidence, time_range, meter_to_model, policy=None):
    """直接约束 raw velocity 的 SHM 位移上界和实际时间显隐。

    不读取指数衰减后的 feature velocity，避免宽寿命把运动约束变成零。
    稳定区强约束；上下文/未知区的弱先验不要求位移为零或 alpha 恒为1。
    """
    policy = TEMPORAL_POLICY if policy is None else policy
    if confidence.shape != (len(model._xyz),):
        raise ValueError("temporal confidence/topology mismatch")
    if not math.isfinite(meter_to_model) or meter_to_model <= 0:
        raise ValueError("positive meter-to-model scale required")
    stable = confidence.detach().to(model._xyz)
    context = 1.0 - stable
    motion = torch.linalg.vector_norm(model._velocity, dim=1) * (
        float(model.T) / math.pi / meter_to_model
    )
    center = model._t[:, 0]
    # -log temporal marginal 在窗口两个端点的均值；在 log 域保持数值稳定。
    log_width = model._scaling_t[:, 0]
    inverse_variance = torch.exp((-2 * log_width).clamp(max=30))
    start, end = map(float, time_range)
    visibility = 0.25 * ((center - start).square() + (center - end).square()) * inverse_variance

    def average(values, weights):
        # 分母按域内点数，保留0.5道路置信度的真实减权，不把它归一化回1。
        return (values * weights).sum() / (weights > 0).sum().clamp_min(1)

    terms = {
        "stable_motion": average(motion, stable),
        "stable_visibility": average(visibility, stable),
        "context_motion": average(motion, context),
        "context_visibility": average(visibility, context),
    }
    loss = sum(terms[name] * policy[name + "_weight"] for name in terms)
    metrics = {name: value.detach() for name, value in terms.items()}
    metrics.update(
        stable_confidence_mass=stable.sum(),
        trusted_stable_points=(stable > 0).sum(),
        context_points=(stable == 0).sum(),
        background_points=torch.tensor(len(stable), device=stable.device),
    )
    return loss, metrics
