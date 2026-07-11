"""在真实 nuScenes clip 上注入轨迹 corruption，供 P1 人工检查导出。"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..auditor.state import MotionState, Track
from ..projector import DynamicsProjector
from ..projector.components import EgoWarpBackground, TemporalBorderSupport
from ..projector.warper import composite_objects
from .synthetic_projection import CORRUPTIONS

PRIOR_WEIGHT = 0.1


def _rand(generator: torch.Generator, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(shape, generator=generator) * (high - low) + low


def _clone_track(track: Track) -> Track:
    return Track(
        instance_token=track.instance_token,
        category=track.category,
        xyxy=track.xyxy.clone(),
        depth=track.depth.clone(),
        present=track.present.clone(),
    )


def corrupt_track(track: Track, corruption: str, generator: torch.Generator) -> None:
    """就地对轨迹注入与 P0 同类的几何 corruption。"""
    k = int(track.xyxy.shape[0])
    centers = track.center.clone()
    log_scales = torch.log(track.scale.clamp_min(1e-3))
    present = track.present.clone()
    inner = 1 + int(torch.randint(0, max(k - 2, 1), (1,), generator=generator))
    direction = _rand(generator, -1.0, 1.0, (2,))
    direction = direction / direction.norm().clamp_min(1e-6)

    if corruption == "center_impulse":
        centers[inner] += direction * float(_rand(generator, 9.0, 16.0, ()).item())
    elif corruption == "center_jitter":
        centers += torch.randn((k, 2), generator=generator) * 4.5
    elif corruption == "constant_acceleration":
        time = torch.arange(k, dtype=torch.float32)
        curve = (time - (k - 1) / 2.0).square()
        centers += curve[:, None] * direction[None] * float(_rand(generator, 0.7, 1.2, ()).item())
    elif corruption == "scale_impulse":
        log_scales[inner] += direction * float(_rand(generator, 0.45, 0.75, ()).item())
    elif corruption == "temporal_gap":
        present[inner] = False
        spike_index = inner - 1 if inner > 1 else min(inner + 1, k - 1)
        centers[spike_index] += direction * float(_rand(generator, 8.0, 13.0, ()).item())
    else:
        raise ValueError(f"未知 corruption: {corruption}")

    half = torch.exp(log_scales) * 0.5
    xyxy = torch.cat([centers - half, centers + half], dim=-1)
    xyxy[~present] = float("nan")
    depth = track.depth.clone()
    depth[~present] = float("nan")
    track.xyxy = xyxy
    track.depth = depth
    track.present = present


def select_track_index(tracks: list[Track], case_index: int, seed: int) -> int:
    """优先选择可见帧数最多的轨迹，保证人工检查有足够目标区域。"""
    if not tracks:
        raise ValueError("clip 不含可用轨迹")
    scores = [(index, int(track.present.sum())) for index, track in enumerate(tracks)]
    max_present = max(score for _, score in scores)
    candidates = [index for index, score in scores if score == max_present]
    pick = (case_index + seed) % len(candidates)
    return candidates[pick]


def build_corrupted_input(
    frames: torch.Tensor,
    state: MotionState,
    corrupted_tracks: list[Track],
    anchor: int = 0,
) -> torch.Tensor:
    """将 corruption 直接渲染到像域，作为人工检查的左侧输入 y。"""
    background = EgoWarpBackground().project(frames, state, anchor)
    hw = state.meta.get("hw") or (frames.shape[-2], frames.shape[-1])
    support = TemporalBorderSupport().classify(corrupted_tracks, hw)
    y_corrupted, _ = composite_objects(
        background, frames, corrupted_tracks, corrupted_tracks, support
    )
    return y_corrupted


def evaluate_synthetic_case(
    case_index: int,
    dataset,
    auditor,
    projector: DynamicsProjector,
    settings: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    clip_index = case_index % len(dataset)
    sample = dataset[clip_index]
    return project_synthetic_sample(
        sample,
        auditor,
        projector,
        case_index,
        seed,
        settings,
        clip_index=clip_index,
    )


def project_synthetic_sample(
    sample: dict[str, Any],
    auditor,
    projector: DynamicsProjector,
    case_index: int,
    seed: int,
    settings: Mapping[str, Any],
    *,
    clip_index: int | None = None,
) -> dict[str, Any]:
    """对已加载的 clean sample 注入错误并生成 synthetic projection target。"""
    frames = sample["frames"]
    state = auditor.audit(sample)
    if not state.tracks:
        location = sample.get("sample_id") if clip_index is None else f"clip {clip_index}"
        raise ValueError(f"{location} 无可用轨迹，无法构造 synthetic case")

    corruption = CORRUPTIONS[case_index % len(CORRUPTIONS)]
    generator = torch.Generator().manual_seed(seed + case_index * 17)
    track_index = select_track_index(state.tracks, case_index, seed)
    corrupted_tracks = [_clone_track(track) for track in state.tracks]
    corrupt_track(corrupted_tracks[track_index], corruption, generator)

    state_corrupted = MotionState(
        u_static=state.u_static,
        u_ego=state.u_ego,
        static_mask=state.static_mask,
        flow_conf=state.flow_conf,
        depth=state.depth,
        tracks=corrupted_tracks,
        meta=dict(state.meta),
    )
    device = state.depth.device
    frames = frames.to(device)
    state_corrupted = state_corrupted.to(device)
    corrupted_tracks = state_corrupted.tracks
    y_corrupted = build_corrupted_input(frames, state_corrupted, corrupted_tracks)
    result = projector.project(frames, state_corrupted)

    prior_weight = float(settings.get("prior_weight", PRIOR_WEIGHT))
    before = float(result.energy_before["obj"]) + prior_weight * float(result.energy_before["prior"])
    after = float(result.energy_after["obj"]) + prior_weight * float(result.energy_after["prior"])
    return {
        "case_id": f"synthetic-{case_index:03d}",
        "case_index": case_index,
        "source": "synthetic",
        "clip_index": clip_index,
        "sample_id": sample["sample_id"],
        "track_index": track_index,
        "track_token": corrupted_tracks[track_index].instance_token,
        "corruption": corruption,
        "prior_weight": prior_weight,
        "energy_before": before,
        "energy_after": after,
        "energy_decreased": after < before - float(settings.get("energy_tolerance", 1e-6)),
        "eligible_fraction": float(result.diagnostics["eligible_fraction"]),
        "static_drift": float(auditor.static_drift_score(state)),
        "latent_flow": state.u_static.detach().cpu(),
        "flow_confidence": state.flow_conf.detach().cpu().unsqueeze(1),
        "y_corrupted": y_corrupted.detach().cpu(),
        "x_dagger": result.target.detach().cpu(),
        "mask": result.valid_mask.detach().cpu(),
    }
