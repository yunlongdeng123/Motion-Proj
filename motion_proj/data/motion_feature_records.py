"""A1 sampled motion-feature records、局部相关和小型 probe 工具。"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class MotionFeatureRecordError(RuntimeError):
    """A1 scene split、feature query 或 probe 输入不合法。"""


def stable_scene_split(
    records: Sequence[Mapping[str, Any]],
    *,
    train_count: int,
    dev_count: int,
    holdout_count: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """按 scene 名称稳定取每 scene 首个 clip，三个 split 严格不交叉。"""
    required = int(train_count) + int(dev_count) + int(holdout_count)
    if min(int(train_count), int(dev_count), int(holdout_count)) < 0 or required <= 0:
        raise ValueError("scene split counts 非法")
    indexed = [dict(row, dataset_index=index) for index, row in enumerate(records)]
    indexed.sort(
        key=lambda row: (
            str(row.get("scene_name", "")),
            int(row.get("start_index", -1)),
            str(row.get("sample_id", "")),
        )
    )
    distinct: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in indexed:
        scene = str(row.get("scene_token") or row.get("scene_name") or "")
        if not scene or scene in seen:
            continue
        seen.add(scene)
        distinct.append(row)
        if len(distinct) == required:
            break
    if len(distinct) != required:
        raise MotionFeatureRecordError(
            f"scene-distinct clips 不足: required={required}, actual={len(distinct)}"
        )
    left = int(train_count)
    middle = left + int(dev_count)
    result = {
        "train": distinct[:left],
        "dev": distinct[left:middle],
        "holdout": distinct[middle:],
    }
    scene_sets = [
        {str(row["scene_token"]) for row in result[name]}
        for name in ("train", "dev", "holdout")
    ]
    if any(scene_sets[i] & scene_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise MotionFeatureRecordError("scene split 泄漏")
    return result


def split_fingerprint(split: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    payload = {
        name: [
            (str(row.get("scene_token")), str(row.get("sample_id")))
            for row in split.get(name, [])
        ]
        for name in ("train", "dev", "holdout")
    }
    raw = repr(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _pixel_grid(points: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    height, width = (int(image_hw[0]), int(image_hw[1]))
    points = torch.as_tensor(points)
    return torch.stack(
        [
            2.0 * (points[..., 0] + 0.5) / float(width) - 1.0,
            2.0 * (points[..., 1] + 0.5) / float(height) - 1.0,
        ],
        dim=-1,
    )


def sample_temporal_features(
    features: torch.Tensor,
    time_indices: torch.Tensor,
    points: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """从 ``[T,C,Hf,Wf]`` 在任意 ``(t,u,v)`` query 双线性采样。"""
    if features.ndim != 4 or points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("features/points 必须为 [T,C,H,W] / [N,2]")
    times = torch.as_tensor(time_indices, dtype=torch.long, device=features.device)
    if times.ndim != 1 or times.shape[0] != points.shape[0]:
        raise ValueError("time_indices 与 points 数量不一致")
    if bool((times < 0).any() or (times >= features.shape[0]).any()):
        raise ValueError("time index 越界")
    result = torch.empty(
        (points.shape[0], features.shape[1]), device=features.device, dtype=features.dtype,
    )
    for time in torch.unique(times).tolist():
        selected = times == int(time)
        grid = _pixel_grid(points[selected].to(features), image_hw).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(
            features[int(time): int(time) + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[0, :, 0].transpose(0, 1)
        result[selected] = sampled
    return result


def local_correlation_window(
    features: torch.Tensor,
    time_indices: torch.Tensor,
    source_points: torch.Tensor,
    center_points: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    radius_cells: int,
) -> torch.Tensor:
    """source@t 与以 center@t+1 为中心的局部 feature window 余弦相关。"""
    radius = int(radius_cells)
    if radius < 0:
        raise ValueError("radius_cells 不得为负")
    times = torch.as_tensor(time_indices, dtype=torch.long, device=features.device)
    if bool((times + 1 >= features.shape[0]).any()):
        raise ValueError("local correlation 需要有效 t+1")
    source = F.normalize(
        sample_temporal_features(features, times, source_points, image_hw=image_hw).float(),
        dim=-1,
        eps=1.0e-8,
    )
    image_h, image_w = image_hw
    feature_h, feature_w = features.shape[-2:]
    offsets = torch.tensor(
        [
            (dx * image_w / feature_w, dy * image_h / feature_h)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ],
        device=features.device,
        dtype=torch.float32,
    )
    window_points = center_points.to(features).unsqueeze(1) + offsets.unsqueeze(0)
    repeated_times = (times + 1).unsqueeze(1).expand(-1, offsets.shape[0]).reshape(-1)
    sampled = sample_temporal_features(
        features,
        repeated_times,
        window_points.reshape(-1, 2),
        image_hw=image_hw,
    ).reshape(center_points.shape[0], offsets.shape[0], -1)
    sampled = F.normalize(sampled.float(), dim=-1, eps=1.0e-8)
    return torch.einsum("nc,nkc->nk", source, sampled)


def deterministic_subsample_indices(count: int, maximum: int) -> torch.Tensor:
    if count <= 0 or maximum <= 0:
        return torch.empty(0, dtype=torch.long)
    if count <= maximum:
        return torch.arange(count, dtype=torch.long)
    return torch.linspace(0, count - 1, maximum).round().long().unique(sorted=True)


def random_projection_matrix(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions 必须为正")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((input_dim, output_dim), generator=generator, dtype=dtype)
    return matrix / math.sqrt(float(input_dim))


def projection_fingerprint(matrix: torch.Tensor) -> str:
    value = matrix.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class RidgeModel:
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    regularization: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        value = (features.float() - self.mean) / self.scale
        value = torch.cat([value, torch.ones((value.shape[0], 1), dtype=value.dtype)], dim=1)
        return value @ self.weight


def fit_ridge(features: torch.Tensor, targets: torch.Tensor, *, regularization: float) -> RidgeModel:
    """固定容量 multi-output linear ridge；只以 train statistics 标准化。"""
    x = torch.as_tensor(features, dtype=torch.float32).cpu()
    y = torch.as_tensor(targets, dtype=torch.float32).cpu()
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("ridge 输入必须为同样本数的二维张量")
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("ridge 输入包含 NaN/Inf")
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    normalized = (x - mean) / scale
    design = torch.cat([normalized, torch.ones((x.shape[0], 1))], dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float32) * float(regularization)
    penalty[-1, -1] = 0.0
    gram = design.transpose(0, 1) @ design + penalty
    rhs = design.transpose(0, 1) @ y
    weight = torch.linalg.solve(gram, rhs)
    return RidgeModel(mean=mean, scale=scale, weight=weight, regularization=float(regularization))


def vector_epe(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 2:
        raise ValueError("EPE 输入必须是同形状 [N,2]")
    return float(torch.linalg.vector_norm(prediction - target, dim=-1).mean())


def angular_error_deg(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum_target_magnitude: float = 1.0e-4,
) -> float | None:
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    pred_norm = torch.linalg.vector_norm(prediction, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    valid = target_norm >= float(minimum_target_magnitude)
    if not bool(valid.any()):
        return None
    cosine = (
        (prediction[valid] * target[valid]).sum(dim=-1)
        / (pred_norm[valid] * target_norm[valid]).clamp_min(1.0e-8)
    ).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).mean())


def permute_learned_features(
    features: torch.Tensor,
    *,
    learned_width: int,
    seed: int,
) -> torch.Tensor:
    """只打乱 learned feature/cost 部分，保留位置与 dt auxiliary。"""
    value = torch.as_tensor(features).clone()
    if value.shape[0] < 2 or not 0 < int(learned_width) <= value.shape[1]:
        raise ValueError("learned_width 或样本数非法")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(value.shape[0], generator=generator)
    value[:, : int(learned_width)] = value[permutation, : int(learned_width)]
    return value


def permute_targets(targets: torch.Tensor, *, seed: int) -> torch.Tensor:
    value = torch.as_tensor(targets).clone()
    if value.shape[0] < 2:
        raise ValueError("target permutation 至少需要两个样本")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return value[torch.randperm(value.shape[0], generator=generator)]


def permute_instance_targets(
    targets: torch.Tensor,
    instance_ids: Sequence[str],
    *,
    seed: int,
) -> torch.Tensor:
    """按 instance 交换完整 target 序列，保留每条轨迹内部的时间顺序。

    不同轨迹长度允许不同；接收方按自身长度循环读取 donor 序列。这个 control
    只破坏 feature--instance 对应关系，不把一条轨迹内部的时间结构打散。
    """
    value = torch.as_tensor(targets).clone()
    if value.ndim != 2 or value.shape[0] != len(instance_ids) or value.shape[0] < 2:
        raise ValueError("instance target permutation 输入非法")
    ordered_ids = list(dict.fromkeys(str(item) for item in instance_ids))
    if len(ordered_ids) < 2:
        raise ValueError("instance target permutation 至少需要两个 instance")
    groups = {
        instance_id: [index for index, value_id in enumerate(instance_ids) if str(value_id) == instance_id]
        for instance_id in ordered_ids
    }
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(len(ordered_ids), generator=generator).tolist()
    # 避免偶然的固定点；循环平移仍保持由 seed 决定的 donor 顺序。
    if any(index == donor for index, donor in enumerate(permutation)):
        offset = int(seed) % (len(ordered_ids) - 1) + 1
        permutation = [(index + offset) % len(ordered_ids) for index in range(len(ordered_ids))]
    output = value.clone()
    for receiver_index, receiver_id in enumerate(ordered_ids):
        donor_id = ordered_ids[permutation[receiver_index]]
        receiver_rows = groups[receiver_id]
        donor_rows = groups[donor_id]
        for position, row_index in enumerate(receiver_rows):
            output[row_index] = value[donor_rows[position % len(donor_rows)]]
    return output


def relative_improvement(error: float, baseline_error: float) -> float | None:
    if not math.isfinite(float(error)) or not math.isfinite(float(baseline_error)) or baseline_error <= 0:
        return None
    return 1.0 - float(error) / float(baseline_error)
