"""PA1-BRANCH：结构对齐 common-prefix sibling candidate pilot。

该模块只在冻结的 Base SVD rollout 上构造候选，不读取 future GT、不写训练 cache、
不更新 LoRA，也不输出 preference winner。它首先复核 PA1-HORIZON 的 14-frame
选择，然后以一个 official Base trace 的共享前缀为 anchor，构造两组零均值、等范数
的反向扰动。独立 seed 只作为 distance calibration diagnostic，永远不进入 candidate
manifest 或 preference label。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..backbones import build_backbone
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data.physics_dpo_schema import (
    PhysicsDpoSchemaError,
    validate_candidates,
    validate_conditions,
)
from ..eval.independent_tracks import (
    CoTracker3IndependentEvaluator,
    IndependentTrackState,
    aggregate_dynamics,
    summarize_camera_compensated_dynamics,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video, write_video
from ..utils.viz import hstack_panels
from .physics_dpo_horizon import (
    METRIC_FIELDS,
    _dataset_for_horizon,
    _existing_video_path,
    _json_line,
    _load_condition_frame,
    _load_scene_split,
    _make_condition_record,
    _peak_memory_bytes,
    _reset_peak_memory,
    _save_tensor,
    _sync_cuda,
    _tensor_fingerprint,
    _validate_completed_run,
    _validate_pa0,
    fingerprint_denoising_trace,
    select_profile_conditions,
)
from .svd_conditioning_parity import _autocast, _base_model_fingerprint, _frames_to_tensor, trace_backbone_generation


SEQUENCE_TRACE_FIELDS = (
    "scheduler_timesteps",
    "scheduler_inputs",
    "scaled_model_inputs",
    "unet_inputs",
    "raw_model_outputs",
    "unconditional_raw_model_outputs",
    "conditional_raw_model_outputs",
    "cfg_outputs",
    "scheduler_step_outputs",
    "post_step_latents",
)
ALLOWED_FORK_FRACTIONS = (0.4, 0.6, 0.8)
REVIEW_VERDICTS = frozenset({"same_scene", "different_composition", "invalid", "uncertain"})


class BranchPilotError(RuntimeError):
    """PA1-BRANCH provenance、sampling 或结构门禁不满足。"""


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _maximum(values: Sequence[float]) -> float | None:
    return max(float(value) for value in values) if values else None


def resolve_fork_step(num_inference_steps: int, fork_fraction: float) -> int:
    """将预注册的 fraction 解释为已完成的 shared scheduler transitions 数。"""
    if int(num_inference_steps) <= 1:
        raise BranchPilotError("PA1-BRANCH 需要至少两个 denoising steps")
    if not any(math.isclose(float(fork_fraction), allowed, rel_tol=0.0, abs_tol=1.0e-9) for allowed in ALLOWED_FORK_FRACTIONS):
        raise BranchPilotError(f"fork_fraction 只能为 {ALLOWED_FORK_FRACTIONS}")
    step = int(round(int(num_inference_steps) * float(fork_fraction)))
    if step <= 0 or step >= int(num_inference_steps):
        raise BranchPilotError("fork step 必须严格位于完整 denoising chain 内部")
    return step


def _rms(value: torch.Tensor) -> float:
    tensor = value.detach().float()
    return float(tensor.square().mean().sqrt()) if int(tensor.numel()) else 0.0


def make_antithetic_perturbations(
    prefix_latent: torch.Tensor,
    *,
    sigma_at_fork: float,
    strength_rho: float,
    direction_seed: int,
) -> dict[str, dict[str, Any]]:
    """构造两组 permutation-distinct 的零均值正/负方向。

    第二方向是第一方向的固定随机 permutation，因此四条理论 perturbation 在浮点
    表示中严格等范数；实际 bf16 注入后的值也另行落盘并受阈值检查。
    """
    if prefix_latent.numel() < 2:
        raise BranchPilotError("prefix latent 太小，不能构造零均值方向")
    if not math.isfinite(float(sigma_at_fork)) or float(sigma_at_fork) <= 0.0:
        raise BranchPilotError("fork sigma 必须为有限正数")
    if not math.isfinite(float(strength_rho)) or float(strength_rho) <= 0.0:
        raise BranchPilotError("branch strength_rho 必须为有限正数")
    generator = torch.Generator(device="cpu").manual_seed(int(direction_seed))
    direction = torch.randn(tuple(prefix_latent.shape), generator=generator, dtype=torch.float32)
    direction = direction - direction.mean()
    norm = direction.square().mean().sqrt()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise BranchPilotError("无法归一化 antithetic direction")
    direction = direction / norm
    permutation = torch.randperm(direction.numel(), generator=generator)
    second = direction.reshape(-1)[permutation].reshape_as(direction)
    target_rms = float(strength_rho) * float(sigma_at_fork)
    signed = {
        "g0-positive": direction * target_rms,
        "g0-negative": -direction * target_rms,
        "g1-positive": second * target_rms,
        "g1-negative": -second * target_rms,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, delta in signed.items():
        result[name] = {
            "group_index": int(name[1]),
            "direction": "positive" if name.endswith("positive") else "negative",
            "theoretical_delta": delta.contiguous(),
            "theoretical_rms": _rms(delta),
            "theoretical_mean": float(delta.mean()),
            "direction_hash": _tensor_fingerprint(delta),
        }
    target_values = [float(row["theoretical_rms"]) for row in result.values()]
    if max(target_values) - min(target_values) > 1.0e-7 * max(target_values):
        raise BranchPilotError("antithetic perturbation 未保持理论等范数")
    return result


def calibrated_future_distance(
    *,
    candidate_distance: float | None,
    rerun_floor: float | None,
    independent_distance: float | None,
    minimum_ratio: float,
    maximum_ratio: float,
) -> dict[str, Any]:
    """检查候选未来距离位于 deterministic floor 与 independent-seed 距离之间。"""
    candidate = _finite(candidate_distance)
    floor = _finite(rerun_floor)
    independent = _finite(independent_distance)
    if candidate is None or floor is None or independent is None or independent <= floor:
        return {"valid": False, "passed": False, "reason": "distance_invalid", "ratio_to_independent": None}
    numerator = max(candidate - floor, 0.0)
    denominator = independent - floor
    ratio = numerator / denominator
    if candidate <= floor:
        return {"valid": True, "passed": False, "reason": "candidate_at_rerun_floor", "ratio_to_independent": ratio}
    if ratio < float(minimum_ratio):
        return {"valid": True, "passed": False, "reason": "candidate_indistinguishable", "ratio_to_independent": ratio}
    if ratio > float(maximum_ratio):
        return {"valid": True, "passed": False, "reason": "candidate_too_independent", "ratio_to_independent": ratio}
    return {"valid": True, "passed": True, "reason": None, "ratio_to_independent": ratio}


def track_correspondence(left: IndependentTrackState, right: IndependentTrackState, *, maximum_query_delta_px: float) -> dict[str, Any]:
    """同一固定 CoTracker grid 下的可见性相交/并集与 strata 一致性。"""
    if left.visibility.shape != right.visibility.shape or left.points.shape != right.points.shape:
        return {"valid": False, "reason": "track_shape_mismatch", "coverage": None, "label_agreement": None}
    if left.query_points.shape != right.query_points.shape:
        return {"valid": False, "reason": "query_shape_mismatch", "coverage": None, "label_agreement": None}
    query_delta = torch.linalg.vector_norm(left.query_points.float() - right.query_points.float(), dim=-1)
    max_delta = float(query_delta.max()) if int(query_delta.numel()) else math.inf
    if not math.isfinite(max_delta) or max_delta > float(maximum_query_delta_px):
        return {
            "valid": False,
            "reason": "query_grid_mismatch",
            "coverage": None,
            "label_agreement": None,
            "query_max_delta_px": max_delta,
        }
    union = left.visibility | right.visibility
    intersection = left.visibility & right.visibility
    coverage = float(intersection.float().sum() / union.float().sum()) if bool(union.any()) else 0.0
    if len(left.labels) != len(right.labels):
        return {"valid": False, "reason": "label_count_mismatch", "coverage": coverage, "label_agreement": None}
    agreement = float(sum(a == b for a, b in zip(left.labels, right.labels)) / len(left.labels)) if left.labels else 0.0
    return {
        "valid": True,
        "reason": None,
        "coverage": coverage,
        "label_agreement": agreement,
        "query_max_delta_px": max_delta,
        "intersection_visible": int(intersection.sum()),
        "union_visible": int(union.sum()),
    }


def choose_calibration_action(
    *,
    current_strength_name: str,
    all_distance_indistinguishable: bool,
    any_structure_mismatch: bool,
    any_other_failure: bool,
) -> str:
    """严格遵循 V2 的逐级最小强度、后调整 fork 的规则。"""
    if all_distance_indistinguishable and str(current_strength_name) == "small":
        return "increase_strength_to_medium"
    if all_distance_indistinguishable and str(current_strength_name) == "medium":
        return "increase_strength_to_large"
    if any_structure_mismatch:
        return "adjust_fork_to_0.8"
    if any_other_failure:
        return "inspect_common_prefix_before_renoise"
    return "human_review"


def _prefix_trace_fingerprint(trace: Mapping[str, Any], fork_step: int) -> dict[str, Any]:
    if int(fork_step) <= 0:
        raise BranchPilotError("prefix fingerprint 需要至少一步 shared prefix")
    missing = [name for name in ("condition_noise", "initial_video_latents", *SEQUENCE_TRACE_FIELDS) if name not in trace]
    if missing:
        raise BranchPilotError(f"official Base trace 缺少 prefix 字段: {missing}")
    fields: dict[str, Any] = {
        "condition_noise": _tensor_fingerprint(trace["condition_noise"]),
        "initial_video_latents": _tensor_fingerprint(trace["initial_video_latents"]),
    }
    for name in SEQUENCE_TRACE_FIELDS:
        values = trace[name]
        if not isinstance(values, Sequence) or len(values) < int(fork_step):
            raise BranchPilotError(f"official Base trace 的 {name} 不足 fork step")
        fields[name] = [_tensor_fingerprint(value) for value in values[:int(fork_step)]]
    prefix_latent = trace["post_step_latents"][int(fork_step) - 1]
    return {
        "fork_step": int(fork_step),
        "prefix_latent_hash": _tensor_fingerprint(prefix_latent),
        "prefix_trace_fields": fields,
        "prefix_trace_hash": sha256_json(fields),
    }


def _manual_state(
    backbone: SVDBackbone,
    cond_frame: torch.Tensor,
    *,
    seed: int,
    num_frames: int,
    num_inference_steps: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    """仅重建 official condition 与 scheduler；suffix 从现成 prefix latent 开始。"""
    pipe = backbone._generation_pipeline()
    pipe.set_progress_bar_config(disable=True)
    settings = backbone.generation_settings()
    if str(settings["protocol"]) != "svd_official_v1":
        raise BranchPilotError("common-prefix 只支持 svd_official_v1")
    device = torch.device(backbone.device)
    pipe._guidance_scale = float(settings["max_guidance_scale"])
    generator = torch.Generator(device=backbone.device).manual_seed(int(seed))
    with torch.no_grad(), _autocast(backbone):
        conditioning = backbone.build_official_generation_conditioning(
            cond_frame, generator=generator, num_frames=int(num_frames), height=int(height), width=int(width),
        )
        pipe.scheduler.set_timesteps(int(num_inference_steps), device=device)
        initial_latents = pipe.prepare_latents(
            1,
            int(num_frames),
            int(pipe.unet.config.in_channels),
            int(height),
            int(width),
            conditioning["image_embeds"].dtype,
            device,
            generator,
        )
    guidance = torch.linspace(
        float(settings["min_guidance_scale"]), float(settings["max_guidance_scale"]), int(num_frames),
        device=device, dtype=initial_latents.dtype,
    ).unsqueeze(0)
    guidance = guidance.repeat(1, 1)
    while guidance.ndim < initial_latents.ndim:
        guidance = guidance.unsqueeze(-1)
    return {
        "pipe": pipe,
        "conditioning": conditioning,
        "initial_latents": initial_latents,
        "timesteps": pipe.scheduler.timesteps.detach().clone(),
        "guidance": guidance,
        "do_cfg": bool(float(settings["max_guidance_scale"]) > 1.0),
        "num_frames": int(num_frames),
        "num_inference_steps": int(num_inference_steps),
        "decode_chunk_size": 4,
    }


def _run_suffix(
    backbone: SVDBackbone,
    state: Mapping[str, Any],
    prefix_latent: torch.Tensor,
    *,
    fork_step: int,
) -> dict[str, Any]:
    """从已完成的 prefix latent 恢复 Euler scheduler，并捕获 suffix 的逐步张量。"""
    pipe = state["pipe"]
    device = torch.device(backbone.device)
    pipe.scheduler.set_timesteps(int(state["num_inference_steps"]), device=device)
    if not hasattr(pipe.scheduler, "set_begin_index"):
        raise BranchPilotError("当前 scheduler 不支持从 common prefix 恢复")
    pipe.scheduler.set_begin_index(int(fork_step))
    timesteps = pipe.scheduler.timesteps
    if int(fork_step) >= len(timesteps):
        raise BranchPilotError("fork_step 超出 scheduler timesteps")
    latents = prefix_latent.detach().to(device=device, dtype=state["initial_latents"].dtype).contiguous()
    values: dict[str, list[torch.Tensor]] = {name: [] for name in SEQUENCE_TRACE_FIELDS}
    with torch.no_grad(), _autocast(backbone):
        for time_index, timestep in enumerate(timesteps[int(fork_step):], start=int(fork_step)):
            latent_model_input = torch.cat([latents] * 2) if bool(state["do_cfg"]) else latents
            scaled = pipe.scheduler.scale_model_input(latent_model_input, timestep)
            unet_input = torch.cat([scaled, state["conditioning"]["image_latents"]], dim=2)
            raw = pipe.unet(
                unet_input,
                timestep,
                encoder_hidden_states=state["conditioning"]["image_embeds"],
                added_time_ids=state["conditioning"]["added_time_ids"],
                return_dict=False,
            )[0]
            if bool(state["do_cfg"]):
                uncond, conditional = raw.chunk(2)
                cfg_output = uncond + state["guidance"] * (conditional - uncond)
            else:
                uncond = conditional = None
                cfg_output = raw
            stepped = pipe.scheduler.step(cfg_output, timestep, latents).prev_sample
            values["scheduler_timesteps"].append(timestep.detach().cpu().clone())
            values["scheduler_inputs"].append(latent_model_input.detach().cpu().clone())
            values["scaled_model_inputs"].append(scaled.detach().cpu().clone())
            values["unet_inputs"].append(unet_input.detach().cpu().clone())
            values["raw_model_outputs"].append(raw.detach().cpu().clone())
            if uncond is not None and conditional is not None:
                values["unconditional_raw_model_outputs"].append(uncond.detach().cpu().clone())
                values["conditional_raw_model_outputs"].append(conditional.detach().cpu().clone())
            values["cfg_outputs"].append(cfg_output.detach().cpu().clone())
            values["scheduler_step_outputs"].append(stepped.detach().cpu().clone())
            values["post_step_latents"].append(stepped.detach().cpu().clone())
            latents = stepped
        decoded = pipe.decode_latents(latents, int(state["num_frames"]), int(state["decode_chunk_size"]))
        pil_frames = pipe.video_processor.postprocess_video(video=decoded, output_type="pil")
    if not values["post_step_latents"]:
        raise BranchPilotError("common-prefix suffix 为空")
    return {
        **values,
        "final_latent": values["post_step_latents"][-1],
        "decoded_frames": _frames_to_tensor(pil_frames[0]),
        "suffix_start_step": int(fork_step),
        "suffix_end_step": len(timesteps),
    }


def _combine_prefix_and_suffix(base_trace: Mapping[str, Any], suffix: Mapping[str, Any], *, fork_step: int) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "condition_noise": base_trace["condition_noise"],
        "initial_video_latents": base_trace["initial_video_latents"],
    }
    for name in SEQUENCE_TRACE_FIELDS:
        prefix = list(base_trace[name][:int(fork_step)])
        continuation = list(suffix[name])
        trace[name] = prefix + continuation
    trace["final_latent"] = suffix["final_latent"]
    trace["decoded_frames"] = suffix["decoded_frames"]
    return trace


def _assert_manual_continuation_exact(official: Mapping[str, Any], reconstructed: Mapping[str, Any]) -> dict[str, Any]:
    official_hash = fingerprint_denoising_trace(official)
    reconstructed_hash = fingerprint_denoising_trace(reconstructed)
    if official_hash != reconstructed_hash:
        raise BranchPilotError("common-prefix continuation 未能 exact 重构 official Base trace")
    return official_hash


def _future_rgb_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 4 or left.shape[0] < 2:
        raise BranchPilotError("future RGB distance 需要同形状、至少两帧视频")
    return _rms(left[1:] - right[1:])


def _future_vae_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 5 or left.shape[1] < 2:
        raise BranchPilotError("future VAE distance 需要同形状、至少两帧 latent")
    return _rms(left[:, 1:] - right[:, 1:])


def _first_frame_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 4:
        raise BranchPilotError("first-frame comparison 需要同形状视频")
    diff = left[0].float() - right[0].float()
    return {"rgb_rms": _rms(diff), "rgb_mean_abs": float(diff.abs().mean())}


def _quality_diagnostics(frames: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(frames).all())
    saturation = float((frames.detach().float().abs() >= 0.995).float().mean()) if finite else math.inf
    return {
        "finite": finite,
        "minimum": float(frames.detach().float().min()) if finite else None,
        "maximum": float(frames.detach().float().max()) if finite else None,
        "saturation_fraction": saturation,
    }


def _score_with_state(
    evaluator: CoTracker3IndependentEvaluator,
    frames: torch.Tensor,
    *,
    device: str,
) -> tuple[dict[str, Any], IndependentTrackState]:
    _reset_peak_memory(device)
    started = time.perf_counter()
    state = evaluator.track(frames)
    _sync_cuda(device)
    seconds = time.perf_counter() - started
    dynamics = summarize_camera_compensated_dynamics(state)
    aggregate = aggregate_dynamics(dynamics)
    lengths = state.visibility.sum(dim=1).to(torch.float32)
    return {
        "valid": bool(state.valid) and aggregate is not None,
        "seconds": seconds,
        "peak_vram_bytes": _peak_memory_bytes(device),
        "query_count": int(state.visibility.shape[0]),
        "valid_track_count": int((lengths > 0).sum()),
        "median_track_length_frames": float(lengths.median()) if int(lengths.numel()) else None,
        "track_coverage": float(state.visibility.float().mean()) if int(state.visibility.numel()) else None,
        "dynamics": dynamics,
        "aggregate": aggregate,
        "provider_diagnostics": state.diagnostics,
    }, state


def _candidate_record(
    *,
    condition: Mapping[str, Any],
    candidate_id: str,
    role: str,
    branch_family: str,
    branch_direction: str,
    branch_strength: float,
    perturbation_rms: float,
    antithetic_group_id: str,
    generation_seed: int,
    fork_step: int,
    initial_latent_hash: str,
    prefix_latent_hash: str,
    prefix_trace_hash: str,
    artifact_dir: Path,
    work_dir: Path,
    fps: int,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "condition_id": condition["condition_id"],
        "scene_id": condition["scene_id"],
        "split": condition["split"],
        "candidate_role": role,
        "rgb_video_path": str((artifact_dir / "video.mp4").relative_to(work_dir)),
        "vae_latent_path": str((artifact_dir / "vae_latents.pt").relative_to(work_dir)),
        "diagnostics_path": str((artifact_dir / "trace.json").relative_to(work_dir)),
        "generation_protocol": condition["generation_protocol"],
        "base_model_fingerprint": condition["base_model_fingerprint"],
        "scheduler_fingerprint": condition["scheduler_fingerprint"],
        "initial_latent_hash": initial_latent_hash,
        "prefix_latent_hash": prefix_latent_hash,
        "prefix_trace_hash": prefix_trace_hash,
        "fork_step": int(fork_step),
        "branch_family": branch_family,
        "branch_direction": branch_direction,
        "branch_strength": float(branch_strength),
        "perturbation_rms": float(perturbation_rms),
        "antithetic_group_id": antithetic_group_id,
        "generation_seed": int(generation_seed),
        "guidance_schedule": [float(condition["guidance_schedule"][0]), float(condition["guidance_schedule"][1])],
        "num_frames": int(condition["num_frames"]),
        "fps": int(fps),
        "uses_future_gt": False,
        "git_commit": condition["git_commit"],
        "config_fingerprint": condition["config_fingerprint"],
    }


def _write_candidate_artifact(
    *,
    backbone: SVDBackbone,
    artifact_dir: Path,
    frames: torch.Tensor,
    trace: Mapping[str, Any],
    trace_payload: Mapping[str, Any],
    fps: int,
) -> tuple[Path, torch.Tensor, int]:
    artifact_dir.mkdir(parents=True, exist_ok=False)
    video_path = artifact_dir / "video.mp4"
    write_video(to_uint8_video(frames), str(video_path), fps=int(fps))
    video_artifact = _existing_video_path(video_path)
    _save_tensor(artifact_dir / "final_denoising_latent.pt", trace["final_latent"])
    vae_latents = backbone.encode(frames.unsqueeze(0)).detach().cpu()
    _save_tensor(artifact_dir / "vae_latents.pt", vae_latents)
    atomic_write_json(str(artifact_dir / "trace.json"), dict(trace_payload))
    paths = [video_artifact, artifact_dir / "final_denoising_latent.pt", artifact_dir / "vae_latents.pt", artifact_dir / "trace.json"]
    return video_artifact, vae_latents, sum(path.stat().st_size for path in paths)


def _write_independent_diagnostic(
    *,
    backbone: SVDBackbone,
    diagnostic_dir: Path,
    trace: Mapping[str, Any],
    seed: int,
    fps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    diagnostic_dir.mkdir(parents=True, exist_ok=False)
    frames = trace["decoded_frames"].detach().cpu()
    video_path = diagnostic_dir / "video.mp4"
    write_video(to_uint8_video(frames), str(video_path), fps=int(fps))
    video_artifact = _existing_video_path(video_path)
    _save_tensor(diagnostic_dir / "final_denoising_latent.pt", trace["final_latent"])
    vae_latents = backbone.encode(frames.unsqueeze(0)).detach().cpu()
    _save_tensor(diagnostic_dir / "vae_latents.pt", vae_latents)
    trace_hashes = fingerprint_denoising_trace(trace)
    payload = {
        "diagnostic_only": True,
        "reason": "future-distance upper calibration; excluded from candidate manifest and labels",
        "generation_seed": int(seed),
        "trace": trace_hashes,
    }
    atomic_write_json(str(diagnostic_dir / "trace.json"), payload)
    paths = [
        video_artifact,
        diagnostic_dir / "final_denoising_latent.pt",
        diagnostic_dir / "vae_latents.pt",
        diagnostic_dir / "trace.json",
    ]
    return frames, vae_latents, {
        "video_path": str(video_artifact),
        "trace_path": str(diagnostic_dir / "trace.json"),
        "trace": trace_hashes,
        "storage_bytes": sum(path.stat().st_size for path in paths),
    }


def _make_panel(
    ordered_frames: Sequence[torch.Tensor], labels: Sequence[str]) -> np.ndarray:
    if len(ordered_frames) != len(labels) or not ordered_frames:
        raise BranchPilotError("panel 需要等数量的视频与标签")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment guard
        raise BranchPilotError("PA1 panel 需要 OpenCV") from exc
    videos = [to_uint8_video(value) for value in ordered_frames]
    if len({video.shape[0] for video in videos}) != 1:
        raise BranchPilotError("panel 视频帧数不一致")
    output = []
    for time_index in range(videos[0].shape[0]):
        columns = []
        for video, label in zip(videos, labels):
            frame = video[time_index].copy()
            cv2.rectangle(frame, (0, 0), (72, 24), (0, 0, 0), -1)
            cv2.putText(frame, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            columns.append(frame)
        output.append(hstack_panels(*columns))
    return np.stack(output, axis=0)


def _review_cases(
    *,
    work_dir: Path,
    conditions: Sequence[Mapping[str, Any]],
    frames_by_candidate: Mapping[str, torch.Tensor],
    branch_groups: Sequence[Mapping[str, Any]],
    fps: int,
    review_seed: int,
) -> list[dict[str, Any]]:
    panel_dir = work_dir / "panels"
    panel_dir.mkdir(exist_ok=False)
    base_by_condition = {
        str(row["condition_id"]): str(row["candidate_id"])
        for row in branch_groups if str(row["candidate_role"]) == "base_guard"
    }
    sibling_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in branch_groups:
        if str(row["candidate_role"]) == "sibling":
            sibling_groups.setdefault(str(row["antithetic_group_id"]), []).append(row)
    cases: list[dict[str, Any]] = []
    for index, (group_id, members) in enumerate(sorted(sibling_groups.items())):
        if len(members) != 2:
            raise BranchPilotError("review group 必须恰含一对 antithetic siblings")
        condition_id = str(members[0]["condition_id"])
        base_id = base_by_condition.get(condition_id)
        if base_id is None:
            raise BranchPilotError("review group 缺少 Base guard")
        candidate_ids = [base_id, *sorted(str(row["candidate_id"]) for row in members)]
        rng = random.Random(int(review_seed) + index)
        rng.shuffle(candidate_ids)
        labels = [chr(ord("A") + offset) for offset in range(len(candidate_ids))]
        panel = _make_panel([frames_by_candidate[candidate_id] for candidate_id in candidate_ids], labels)
        case_id = f"branch-review-{index:02d}"
        panel_path = panel_dir / f"{case_id}.mp4"
        write_video(panel, str(panel_path), fps=int(fps))
        artifact = _existing_video_path(panel_path)
        cases.append({
            "case_id": case_id,
            "condition_id": condition_id,
            "antithetic_group_id": group_id,
            "panel_path": str(artifact.relative_to(work_dir)),
            "blind_columns": labels,
            "candidate_order": candidate_ids,
            "rubric": "三列是否保持同一驾驶场景布局、主体身份和条件首帧；两条 sibling 是否只是同场景的不同未来，而非不同构图或灾难性失真？",
        })
    return cases


def _write_review_materials(work_dir: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_json(str(work_dir / "review_cases.json"), list(cases))
    template = "".join(json.dumps({
        "case_id": row["case_id"],
        "verdict": "pending",
        "reviewer": "human",
        "notes": "",
        "rubric": row["rubric"],
    }, ensure_ascii=False, sort_keys=True) + "\n" for row in cases)
    atomic_write_text(str(work_dir / "reviews.template.jsonl"), template)
    atomic_write_text(
        str(work_dir / "REVIEW_README.md"),
        "# PA1-BRANCH 结构对齐盲审\n\n"
        "每个 `panels/*.mp4` 含同一 condition 的 Anchor 与一对 sibling，列顺序已固定随机化，"
        "不展示任何自动 metric 或偏好方向。复制 `reviews.template.jsonl` 为 `reviews.jsonl`，"
        "逐行填写：`same_scene`（布局/身份一致，未来仅有可接受分歧）、"
        "`different_composition`（构图或主体明显不同）、`invalid`（灾难性质量/首帧失真）或 `uncertain`。\n\n"
        "完成 8 个 case 后运行 `--aggregate-only`。该审查不要求选择运动或物理 winner。\n",
    )


def _review_summary(run_dir: Path, review_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cases_path = run_dir / "review_cases.json"
    if not cases_path.is_file():
        raise BranchPilotError("PA1-BRANCH 缺少 review_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise BranchPilotError("review_cases.json 必须为 list")
    expected = {str(row["case_id"]) for row in cases if isinstance(row, Mapping)}
    reviews_path = run_dir / "reviews.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if reviews_path.is_file():
        for raw in reviews_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            case_id = str(row.get("case_id", ""))
            if case_id in rows:
                raise BranchPilotError(f"reviews.jsonl 重复 case_id: {case_id}")
            if case_id not in expected:
                raise BranchPilotError(f"reviews.jsonl 含未知 case_id: {case_id}")
            if str(row.get("verdict")) not in REVIEW_VERDICTS:
                raise BranchPilotError(f"review verdict 非法: {row.get('verdict')!r}")
            rows[case_id] = row
    completed = [rows[case] for case in sorted(expected) if case in rows]
    decisive = [row for row in completed if str(row["verdict"]) != "uncertain"]
    same = sum(str(row["verdict"]) == "same_scene" for row in decisive)
    bad = sum(str(row["verdict"]) in {"different_composition", "invalid"} for row in decisive)
    rate = same / len(decisive) if decisive else None
    required = int(review_cfg["required_cases"])
    passed = bool(
        len(completed) >= required
        and rate is not None
        and rate >= float(review_cfg["minimum_same_scene_rate"])
        and bad <= int(review_cfg["maximum_bad_cases"])
    )
    return {
        "required_cases": required,
        "completed_cases": len(completed),
        "decisive_cases": len(decisive),
        "same_scene": same,
        "different_or_invalid": bad,
        "uncertain": sum(str(row["verdict"]) == "uncertain" for row in completed),
        "same_scene_rate": rate,
        "minimum_same_scene_rate": float(review_cfg["minimum_same_scene_rate"]),
        "maximum_bad_cases": int(review_cfg["maximum_bad_cases"]),
        "pass": passed,
        "status": "pass" if passed else "awaiting_reviews" if len(completed) < required else "rejected",
    }


def _clean_terminal_markers(run_dir: Path) -> None:
    for name in ("COMPLETE", "FAILED", "awaiting_reviews", "MACHINE_COMPLETE"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def aggregate_physics_dpo_branch_reviews(cfg: Any) -> dict[str, Any]:
    """只读 machine artifact 加人工 review；不使用 GPU，不重算 candidate。"""
    run_dir = Path(str(cfg.work_dir))
    machine_path = run_dir / "machine_summary.json"
    if not machine_path.is_file():
        raise BranchPilotError("--aggregate-only 需要已有 machine_summary.json")
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    if str(machine.get("status")) != "awaiting_reviews" or not bool(machine.get("machine_pass")):
        raise BranchPilotError("仅 machine-pass 的 awaiting_reviews run 可以聚合人工 review")
    review = _review_summary(run_dir, cfg.branch.review)
    status = "done" if bool(review["pass"]) else str(review["status"])
    summary = {
        **machine,
        "status": status,
        "human_review": review,
        "next_gate": "PA2-PAIR-03" if status == "done" else "PA1-BRANCH-02",
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    _clean_terminal_markers(run_dir)
    marker = "COMPLETE" if status in {"done", "rejected"} else "awaiting_reviews"
    atomic_write_text(str(run_dir / marker), sha256_json(summary) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": status, "ended_at": utc_now(), "exit_reason": "human_review"})
    atomic_write_json(str(manifest_path), manifest)
    return summary


def _load_horizon_provenance(branch: Any) -> dict[str, Any]:
    run_dir = Path(str(branch.horizon_run))
    summary = _validate_completed_run(run_dir, label="PA1 horizon run")
    if (
        str(summary.get("status")) != "done"
        or str(summary.get("task_id")) != "PA1-HORIZON-01"
        or int(summary.get("decision", {}).get("selected_num_frames", -1)) != int(branch.expected_num_frames)
        or str(summary.get("next_gate")) != "PA1-BRANCH-02"
    ):
        raise BranchPilotError("PA1-HORIZON provenance 未解锁 PA1-BRANCH")
    profile_path = run_dir / "profile.json"
    if not profile_path.is_file():
        raise BranchPilotError("PA1-HORIZON 缺少 profile.json")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected = str(branch.expected_horizon_profile_fingerprint)
    actual = str(summary.get("profile_fingerprint", ""))
    if actual != expected:
        raise BranchPilotError("PA1-HORIZON profile fingerprint 与预注册值不一致")
    return {
        "run_path": str(run_dir),
        "summary_sha256": file_fingerprint(str(run_dir / "summary.json")),
        "profile_sha256": file_fingerprint(str(profile_path)),
        "profile_fingerprint": actual,
        "selected_num_frames": int(branch.expected_num_frames),
        "decision": summary["decision"],
    }


def _validate_branch_config(cfg: Any) -> None:
    branch = cfg.branch
    if str(cfg.model.generation.protocol) != "svd_official_v1" or bool(cfg.model.lora.enable):
        raise BranchPilotError("PA1-BRANCH 只允许冻结 Base svd_official_v1")
    if int(branch.num_inference_steps) != 25:
        raise BranchPilotError("PA1-BRANCH 预注册为 25 inference steps")
    if int(branch.condition_count) != 4 or int(branch.sibling_count) != 4:
        raise BranchPilotError("PA1-BRANCH 只允许 4 conditions 且每 condition 4 siblings")
    if str(branch.family) != "common_prefix":
        raise BranchPilotError("第一轮 PA1-BRANCH 必须从 common_prefix family 开始")
    if int(cfg.data.num_frames) != int(branch.expected_num_frames) or int(cfg.model.num_frames) != int(branch.expected_num_frames):
        raise BranchPilotError("PA1-BRANCH data/model num_frames 必须等于 PA1-HORIZON 冻结值")
    resolve_fork_step(int(branch.num_inference_steps), float(branch.fork_fraction))
    if str(branch.strength_name) not in {"small", "medium", "large"}:
        raise BranchPilotError("strength_name 必须为 small/medium/large")
    if int(branch.independent_seed_offset) <= 0:
        raise BranchPilotError("independent_seed_offset 必须为正，防止与 Base seed 混淆")


def preflight_physics_dpo_branch(cfg: Any) -> dict[str, Any]:
    """只读 PA1-BRANCH 前置检查；不创建 run，不加载 SVD，不占 GPU。"""
    result: dict[str, Any] = {
        "task_id": str(cfg.branch.task_id),
        "status": "ready",
        "uses_gpu": False,
        "uses_future_gt": False,
        "blockers": [],
    }
    try:
        _validate_branch_config(cfg)
        result["horizon"] = _load_horizon_provenance(cfg.branch)
        split, split_provenance = _load_scene_split(cfg.branch)
        result["scene_split"] = split_provenance
        result["pa0_review"] = _validate_pa0(cfg.branch)
        selected = select_profile_conditions(
            split,
            partition=str(cfg.branch.condition_partition),
            condition_count=int(cfg.branch.condition_count),
            required_start_index=int(cfg.branch.required_start_index),
        )
        result["selected_conditions"] = selected
    except Exception as exc:
        result["status"] = "blocked"
        result["blockers"].append({"kind": "provenance", "error": repr(exc)})
        return result
    try:
        dataset = _dataset_for_horizon(cfg.data, num_frames=int(cfg.branch.expected_num_frames))
        records = {str(row["sample_id"]): dict(row) for row in dataset.clip_records}
        checks = []
        for row in result["selected_conditions"]:
            clip_id = str(row["clip_id"])
            first_token = str(row["sample_tokens"][0])
            if clip_id not in records or str(records[clip_id]["sample_tokens"][0]) != first_token:
                raise BranchPilotError(f"PA1-BRANCH condition 不匹配冻结 14-frame clip: {clip_id}")
            frame = _load_condition_frame(dataset, first_token)
            checks.append({"clip_id": clip_id, "condition_frame_sha256": _tensor_fingerprint(frame)})
        result["data"] = {"ready": True, "condition_checks": checks}
    except Exception as exc:
        result["blockers"].append({"kind": "nuscenes_trainval", "error": repr(exc)})
    evaluator = CoTracker3IndependentEvaluator(dict(cfg.branch.evaluator)).preflight()
    result["evaluator"] = evaluator
    if not bool(evaluator.get("available")):
        result["blockers"].append({"kind": "cotracker3", "reasons": evaluator.get("reasons", [])})
    if result["blockers"]:
        result["status"] = "blocked"
    return result


def _generate_condition_group(
    *,
    backbone: SVDBackbone,
    condition: Mapping[str, Any],
    condition_frame: torch.Tensor,
    branch: Any,
    work_dir: Path,
    generation_seed: int,
    direction_seed: int,
    fps: int,
    height: int,
    width: int,
    metrics: JsonlMetrics,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """一个 condition 的 Base guard、4 siblings 和 independent calibration diagnostic。"""
    fork_step = resolve_fork_step(int(branch.num_inference_steps), float(branch.fork_fraction))
    common = {
        "seed": int(generation_seed),
        "num_frames": int(condition["num_frames"]),
        "num_inference_steps": int(branch.num_inference_steps),
        "height": int(height),
        "width": int(width),
    }
    _reset_peak_memory(str(backbone.device))
    started = time.perf_counter()
    official = trace_backbone_generation(backbone, condition_frame.to(backbone.device), **common)
    _sync_cuda(str(backbone.device))
    base_seconds = time.perf_counter() - started
    base_peak = _peak_memory_bytes(str(backbone.device))
    official_hash = fingerprint_denoising_trace(official)
    if not bool(torch.isfinite(official["decoded_frames"]).all()):
        raise BranchPilotError("Base guard decoded frames 包含 NaN/Inf")

    _reset_peak_memory(str(backbone.device))
    rerun_started = time.perf_counter()
    rerun = trace_backbone_generation(backbone, condition_frame.to(backbone.device), **common)
    _sync_cuda(str(backbone.device))
    base_rerun_seconds = time.perf_counter() - rerun_started
    base_rerun_peak = _peak_memory_bytes(str(backbone.device))
    rerun_hash = fingerprint_denoising_trace(rerun)
    if official_hash != rerun_hash:
        raise BranchPilotError("PA1-BRANCH Base guard full denoising trace rerun 不 exact")

    independent_seed = int(generation_seed) + int(branch.independent_seed_offset)
    independent_trace = trace_backbone_generation(
        backbone,
        condition_frame.to(backbone.device),
        **{**common, "seed": independent_seed},
    )
    if not bool(torch.isfinite(independent_trace["decoded_frames"]).all()):
        raise BranchPilotError("independent calibration diagnostic 包含 NaN/Inf")

    prefix = _prefix_trace_fingerprint(official, fork_step)
    prefix_latent = official["post_step_latents"][fork_step - 1]
    state = _manual_state(backbone, condition_frame.to(backbone.device), **common)
    if not torch.equal(state["initial_latents"].detach().cpu(), official["initial_video_latents"]):
        raise BranchPilotError("manual continuation 的 initial latent 未匹配 official Base")
    if not torch.equal(state["conditioning"]["condition_noise"].detach().cpu(), official["condition_noise"]):
        raise BranchPilotError("manual continuation 的 condition noise 未匹配 official Base")
    manual_suffix = _run_suffix(backbone, state, prefix_latent, fork_step=fork_step)
    reconstructed_base = _combine_prefix_and_suffix(official, manual_suffix, fork_step=fork_step)
    manual_hash = _assert_manual_continuation_exact(official, reconstructed_base)

    root = work_dir / "candidates" / str(condition["condition_id"])
    base_dir = root / "base_guard"
    base_trace_payload = {
        "candidate_role": "base_guard",
        "generation_seed": int(generation_seed),
        "fork_fraction": float(branch.fork_fraction),
        "fork_step": fork_step,
        "official_trace": official_hash,
        "rerun_trace": rerun_hash,
        "manual_continuation_trace": manual_hash,
        "base_guard_exact": True,
        "prefix": prefix,
    }
    base_video, base_vae, base_storage = _write_candidate_artifact(
        backbone=backbone,
        artifact_dir=base_dir,
        frames=reconstructed_base["decoded_frames"],
        trace=reconstructed_base,
        trace_payload=base_trace_payload,
        fps=fps,
    )
    base_id = f"base-{condition['condition_id']}"
    base_record = _candidate_record(
        condition={**condition, "guidance_schedule": [
            float(backbone.generation_settings()["min_guidance_scale"]),
            float(backbone.generation_settings()["max_guidance_scale"]),
        ]},
        candidate_id=base_id,
        role="base_guard",
        branch_family="base_guard",
        branch_direction="base",
        branch_strength=0.0,
        perturbation_rms=0.0,
        antithetic_group_id=f"base-{condition['condition_id']}",
        generation_seed=int(generation_seed),
        fork_step=fork_step,
        initial_latent_hash=official_hash["initial_latent_hash"],
        prefix_latent_hash=prefix["prefix_latent_hash"],
        prefix_trace_hash=prefix["prefix_trace_hash"],
        artifact_dir=base_dir,
        work_dir=work_dir,
        fps=fps,
    )
    base_record.update({
        "rgb_video_path": str(base_video.relative_to(work_dir)),
        "score_path": str((base_dir / "score.json").relative_to(work_dir)),
        "generation_seconds": base_seconds,
        "generation_peak_vram_bytes": base_peak,
        "storage_bytes": int(base_storage),
    })

    independent_dir = work_dir / "diagnostic_independent" / str(condition["condition_id"])
    independent_frames, independent_vae, independent_artifact = _write_independent_diagnostic(
        backbone=backbone,
        diagnostic_dir=independent_dir,
        trace=independent_trace,
        seed=independent_seed,
        fps=fps,
    )

    candidates = [base_record]
    frames_by_candidate: dict[str, torch.Tensor] = {base_id: reconstructed_base["decoded_frames"].detach().cpu()}
    vae_by_candidate: dict[str, torch.Tensor] = {base_id: base_vae}
    perturbations = make_antithetic_perturbations(
        prefix_latent,
        sigma_at_fork=float(state["pipe"].scheduler.sigmas[fork_step]),
        strength_rho=float(branch.strength_rho),
        direction_seed=int(direction_seed),
    )
    actual_rms: list[float] = []
    actual_means: list[float] = []
    for branch_name, perturbation in sorted(perturbations.items()):
        delta = perturbation["theoretical_delta"].to(device=backbone.device, dtype=prefix_latent.dtype)
        branched_latent = (prefix_latent.to(backbone.device, dtype=prefix_latent.dtype) + delta).to(prefix_latent.dtype)
        actual = branched_latent.detach().cpu().float() - prefix_latent.detach().cpu().float()
        actual_rms.append(_rms(actual))
        actual_means.append(abs(float(actual.mean())))
        _reset_peak_memory(str(backbone.device))
        sibling_started = time.perf_counter()
        suffix = _run_suffix(backbone, state, branched_latent, fork_step=fork_step)
        _sync_cuda(str(backbone.device))
        sibling_seconds = time.perf_counter() - sibling_started
        sibling_peak = _peak_memory_bytes(str(backbone.device))
        full_trace = _combine_prefix_and_suffix(official, suffix, fork_step=fork_step)
        full_hash = fingerprint_denoising_trace(full_trace)
        if not bool(torch.isfinite(full_trace["decoded_frames"]).all()):
            raise BranchPilotError("sibling decoded frames 包含 NaN/Inf")
        group_index = int(perturbation["group_index"])
        direction = str(perturbation["direction"])
        candidate_id = f"sibling-{condition['condition_id']}-g{group_index}-{direction}"
        artifact_dir = root / f"g{group_index}-{direction}"
        payload = {
            "candidate_role": "sibling",
            "branch_family": "common_prefix",
            "generation_seed": int(generation_seed),
            "fork_fraction": float(branch.fork_fraction),
            "fork_step": fork_step,
            "prefix": prefix,
            "full_trace": full_hash,
            "perturbation": {
                "direction_hash": perturbation["direction_hash"],
                "theoretical_rms": perturbation["theoretical_rms"],
                "theoretical_mean": perturbation["theoretical_mean"],
                "actual_rms": actual_rms[-1],
                "actual_mean": float(actual.mean()),
                "sigma_at_fork": float(state["pipe"].scheduler.sigmas[fork_step]),
                "strength_rho": float(branch.strength_rho),
            },
        }
        sibling_video, vae_latents, storage = _write_candidate_artifact(
            backbone=backbone,
            artifact_dir=artifact_dir,
            frames=full_trace["decoded_frames"],
            trace=full_trace,
            trace_payload=payload,
            fps=fps,
        )
        record = _candidate_record(
            condition={**condition, "guidance_schedule": [
                float(backbone.generation_settings()["min_guidance_scale"]),
                float(backbone.generation_settings()["max_guidance_scale"]),
            ]},
            candidate_id=candidate_id,
            role="sibling",
            branch_family="common_prefix",
            branch_direction=direction,
            branch_strength=float(branch.strength_rho),
            perturbation_rms=float(perturbation["theoretical_rms"]),
            antithetic_group_id=f"antithetic-{condition['condition_id']}-g{group_index}",
            generation_seed=int(generation_seed),
            fork_step=fork_step,
            initial_latent_hash=official_hash["initial_latent_hash"],
            prefix_latent_hash=prefix["prefix_latent_hash"],
            prefix_trace_hash=prefix["prefix_trace_hash"],
            artifact_dir=artifact_dir,
            work_dir=work_dir,
            fps=fps,
        )
        record.update({
            "rgb_video_path": str(sibling_video.relative_to(work_dir)),
            "score_path": str((artifact_dir / "score.json").relative_to(work_dir)),
            "generation_seconds": sibling_seconds,
            "generation_peak_vram_bytes": sibling_peak,
            "storage_bytes": int(storage),
        })
        candidates.append(record)
        frames_by_candidate[candidate_id] = full_trace["decoded_frames"].detach().cpu()
        vae_by_candidate[candidate_id] = vae_latents
        metrics.append(len(candidates), {
            "event": "sibling_generated",
            "condition_id": condition["condition_id"],
            "candidate_id": candidate_id,
            "generation_seconds": sibling_seconds,
            "generation_peak_vram_bytes": sibling_peak,
            "theoretical_perturbation_rms": perturbation["theoretical_rms"],
            "actual_perturbation_rms": actual_rms[-1],
            "actual_perturbation_mean": float(actual.mean()),
        })
    perturbation_gap = (max(actual_rms) - min(actual_rms)) / max(max(actual_rms), 1.0e-12)
    details = {
        "condition_id": condition["condition_id"],
        "base_guard_exact": True,
        "manual_continuation_exact": True,
        "generation_seed": int(generation_seed),
        "independent_seed": independent_seed,
        "fork_step": fork_step,
        "fork_fraction": float(branch.fork_fraction),
        "sigma_at_fork": float(state["pipe"].scheduler.sigmas[fork_step]),
        "actual_perturbation_rms_relative_gap": perturbation_gap,
        "actual_perturbation_max_abs_mean": max(actual_means, default=math.inf),
        "independent_diagnostic": {
            "rgb_video_path": str(Path(independent_artifact["video_path"]).relative_to(work_dir)),
            "vae_latent_path": str((independent_dir / "vae_latents.pt").relative_to(work_dir)),
            "diagnostics_path": str((independent_dir / "trace.json").relative_to(work_dir)),
            "trace": independent_artifact["trace"],
            "storage_bytes": int(independent_artifact["storage_bytes"]),
        },
        "base_generation": {
            "seconds": base_seconds,
            "peak_vram_bytes": base_peak,
            "rerun_seconds": base_rerun_seconds,
            "rerun_peak_vram_bytes": base_rerun_peak,
        },
    }
    return candidates, frames_by_candidate, vae_by_candidate, {
        **details,
        "independent_frames": independent_frames,
        "independent_vae": independent_vae,
    }


def _candidate_feasibility(
    *,
    candidate: Mapping[str, Any],
    score: Mapping[str, Any],
    quality: Mapping[str, Any],
    base_score: Mapping[str, Any],
    distance: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = score.get("aggregate") if isinstance(score.get("aggregate"), Mapping) else None
    base_aggregate = base_score.get("aggregate") if isinstance(base_score.get("aggregate"), Mapping) else None
    checks: dict[str, bool] = {
        "uses_future_gt_false": candidate.get("uses_future_gt") is False,
        "finite": bool(quality.get("finite")),
        "score_valid": bool(score.get("valid")),
        "track_coverage": _finite(score.get("track_coverage")) is not None and float(score["track_coverage"]) >= float(thresholds["minimum_track_coverage"]),
        "median_track_length": _finite(score.get("median_track_length_frames")) is not None and float(score["median_track_length_frames"]) >= float(thresholds["minimum_median_track_length"]),
        "not_saturated": _finite(quality.get("saturation_fraction")) is not None and float(quality["saturation_fraction"]) <= float(thresholds["maximum_saturation_fraction"]),
    }
    if str(candidate["candidate_role"]) == "sibling":
        checks["first_frame"] = distance is not None and float(distance["first_frame"]["rgb_rms"]) <= float(thresholds["maximum_first_frame_rgb_rms"])
        checks["future_distance"] = distance is not None and bool(distance["future_vae_calibration"]["passed"])
        if aggregate is None or base_aggregate is None:
            checks["dynamic_degree"] = False
            checks["survival_noninferior"] = False
        else:
            velocity = _finite(aggregate.get("camera_compensated_image_plane_velocity_rms_px"))
            base_velocity = _finite(base_aggregate.get("camera_compensated_image_plane_velocity_rms_px"))
            survival = _finite(aggregate.get("survival_rate"))
            base_survival = _finite(base_aggregate.get("survival_rate"))
            ratio = velocity / max(base_velocity, 1.0e-8) if velocity is not None and base_velocity is not None else None
            checks["dynamic_degree"] = ratio is not None and float(thresholds["minimum_velocity_ratio_to_base"]) <= ratio <= float(thresholds["maximum_velocity_ratio_to_base"])
            checks["survival_noninferior"] = survival is not None and base_survival is not None and survival >= base_survival - float(thresholds["maximum_survival_drop"])
    return {"feasible": all(checks.values()), "checks": checks}


def _provisional_direction_relation(
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> str:
    """仅用于 sign-balance diagnostics，不是 preference label 或训练 scorer。"""
    pos, neg = positive.get("aggregate"), negative.get("aggregate")
    if not isinstance(pos, Mapping) or not isinstance(neg, Mapping):
        return "abstain"
    pa = _finite(pos.get("camera_compensated_image_plane_acceleration_rms_px"))
    na = _finite(neg.get("camera_compensated_image_plane_acceleration_rms_px"))
    ps = _finite(pos.get("survival_rate"))
    ns = _finite(neg.get("survival_rate"))
    pv = _finite(pos.get("camera_compensated_image_plane_velocity_rms_px"))
    nv = _finite(neg.get("camera_compensated_image_plane_velocity_rms_px"))
    if None in {pa, na, ps, ns, pv, nv}:
        return "abstain"
    assert pa is not None and na is not None and ps is not None and ns is not None and pv is not None and nv is not None
    margin = float(thresholds["provisional_acceleration_relative_margin"])
    motion_tol = float(thresholds["provisional_velocity_relative_tolerance"])
    survival_tol = float(thresholds["maximum_survival_drop"])
    if pa * (1.0 + margin) < na and ps >= ns - survival_tol and pv >= nv * (1.0 - motion_tol):
        return "positive"
    if na * (1.0 + margin) < pa and ns >= ps - survival_tol and nv >= pv * (1.0 - motion_tol):
        return "negative"
    return "abstain"


def _machine_summary(
    *,
    condition_reports: Sequence[Mapping[str, Any]],
    group_reports: Sequence[Mapping[str, Any]],
    branch: Any,
) -> dict[str, Any]:
    valid_conditions = sum(bool(row.get("passed")) for row in condition_reports)
    valid_groups = sum(bool(row.get("passed")) for row in group_reports)
    relations = [str(row.get("provisional_direction_relation")) for row in group_reports]
    positive = relations.count("positive")
    negative = relations.count("negative")
    decisive = positive + negative
    direction_fraction = max(positive, negative) / decisive if decisive else 0.0
    direction_balance = decisive == 0 or direction_fraction <= float(branch.thresholds.maximum_provisional_direction_fraction)
    checks = {
        "minimum_valid_conditions": valid_conditions >= int(branch.thresholds.minimum_valid_conditions),
        "minimum_valid_antithetic_groups": valid_groups >= int(branch.thresholds.minimum_valid_antithetic_groups),
        "provisional_direction_balance": direction_balance,
    }
    all_distance_indistinguishable = bool(group_reports) and all(bool(row.get("distance_indistinguishable")) for row in group_reports)
    any_structure_mismatch = any(bool(row.get("structure_mismatch")) for row in group_reports)
    action = choose_calibration_action(
        current_strength_name=str(branch.strength_name),
        all_distance_indistinguishable=all_distance_indistinguishable,
        any_structure_mismatch=any_structure_mismatch,
        any_other_failure=not all(checks.values()),
    )
    return {
        "machine_pass": all(checks.values()),
        "checks": checks,
        "valid_conditions": valid_conditions,
        "total_conditions": len(condition_reports),
        "valid_antithetic_groups": valid_groups,
        "total_antithetic_groups": len(group_reports),
        "branch_balance": {
            "provisional_positive": positive,
            "provisional_negative": negative,
            "provisional_abstain": relations.count("abstain"),
            "provisional_direction_fraction": direction_fraction if decisive else None,
            "maximum_provisional_direction_fraction": float(branch.thresholds.maximum_provisional_direction_fraction),
            "passed": direction_balance,
            "scope": "diagnostic only; not a pair label or training scorer",
        },
        "calibration_action": action,
    }


def run_physics_dpo_branch(cfg: Any) -> dict[str, Any]:
    """执行一次冻结配置的 PA1-BRANCH common-prefix pilot。"""
    _validate_branch_config(cfg)
    git = git_state(".")
    if git.get("dirty"):
        raise BranchPilotError("正式 PA1-BRANCH 拒绝在 dirty worktree 上运行")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"PA1-BRANCH run directory 已存在: {work_dir}")
    horizon = _load_horizon_provenance(cfg.branch)
    split_manifest, split_provenance = _load_scene_split(cfg.branch)
    pa0 = _validate_pa0(cfg.branch)
    selected = select_profile_conditions(
        split_manifest,
        partition=str(cfg.branch.condition_partition),
        condition_count=int(cfg.branch.condition_count),
        required_start_index=int(cfg.branch.required_start_index),
    )
    cfg_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=cfg_fp,
        cache_fingerprint="not-applicable:pa1-branch-common-prefix",
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split=str(cfg.branch.condition_partition),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.branch.task_id),
        "status": "running",
        "uses_future_gt": False,
        "training": False,
        "branch_family": str(cfg.branch.family),
        "horizon": horizon,
        "scene_split": split_provenance,
        "pa0_review": pa0,
        "condition_selection_rule": {
            "partition": str(cfg.branch.condition_partition),
            "condition_count": int(cfg.branch.condition_count),
            "required_start_index": int(cfg.branch.required_start_index),
            "ordering": "one start-index-matched clip per scene; ascending (scene_token, clip_id)",
        },
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    conditions_path = work_dir / "conditions.jsonl"
    candidates_path = work_dir / "candidate_manifest.jsonl"
    try:
        dataset = _dataset_for_horizon(cfg.data, num_frames=int(cfg.branch.expected_num_frames))
        by_clip = {str(row["sample_id"]): dict(row) for row in dataset.clip_records}
        selected_with_frame: list[dict[str, Any]] = []
        for row in selected:
            clip_id = str(row["clip_id"])
            first_token = str(row["sample_tokens"][0])
            if clip_id not in by_clip or str(by_clip[clip_id]["sample_tokens"][0]) != first_token:
                raise BranchPilotError(f"冻结 branch condition 与 14-frame dataset 不一致: {clip_id}")
            frame = _load_condition_frame(dataset, first_token)
            selected_with_frame.append({**row, "condition_frame": frame, "condition_frame_sha256": _tensor_fingerprint(frame)})

        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        backbone = build_backbone(cfg.model, load=True, device=str(cfg.device))
        if not isinstance(backbone, SVDBackbone):
            raise BranchPilotError("PA1-BRANCH 当前只支持 SVDBackbone")
        backbone.unet.eval()
        backbone.vae.eval()
        backbone.image_encoder.eval()
        metadata = backbone.generation_protocol_metadata()
        base_fp = _base_model_fingerprint(str(cfg.model.pretrained))
        manifest_data.update({
            "base_model_fingerprint": base_fp,
            "generation_protocol": metadata,
            "selected_source_conditions": [{key: value for key, value in row.items() if key != "condition_frame"} for row in selected_with_frame],
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)

        all_conditions: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        frames_by_candidate: dict[str, torch.Tensor] = {}
        vae_by_candidate: dict[str, torch.Tensor] = {}
        independent_by_condition: dict[str, dict[str, Any]] = {}
        condition_generation: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(selected_with_frame):
            condition = _make_condition_record(
                selected=row,
                split=str(cfg.branch.condition_partition),
                camera=str(cfg.data.cameras[0]),
                num_frames=int(cfg.branch.expected_num_frames),
                fps=int(metadata["fps_input"]),
                condition_frame_hash=str(row["condition_frame_sha256"]),
                scheduler_fingerprint=str(metadata["scheduler_config_fingerprint"]),
                base_model_fingerprint=base_fp,
                git_commit=str(git["commit"]),
                config_fingerprint_value=cfg_fp,
            )
            rows, candidate_frames, candidate_vaes, detail = _generate_condition_group(
                backbone=backbone,
                condition=condition,
                condition_frame=row["condition_frame"],
                branch=cfg.branch,
                work_dir=work_dir,
                generation_seed=int(cfg.branch.generation_seed_start) + index,
                direction_seed=int(cfg.branch.direction_seed_start) + index,
                fps=int(metadata["fps_input"]),
                height=int(cfg.data.height),
                width=int(cfg.data.width),
                metrics=metrics,
            )
            all_conditions.append(condition)
            all_candidates.extend(rows)
            frames_by_candidate.update(candidate_frames)
            vae_by_candidate.update(candidate_vaes)
            independent_by_condition[str(condition["condition_id"])] = {
                "frames": detail.pop("independent_frames"),
                "vae": detail.pop("independent_vae"),
                "detail": detail,
            }
            condition_generation[str(condition["condition_id"])] = detail
            _json_line(conditions_path, condition)
            for candidate in rows:
                _json_line(candidates_path, candidate)
            metrics.append(index, {
                "event": "condition_generated",
                "condition_id": condition["condition_id"],
                "base_guard_exact": True,
                "manual_continuation_exact": True,
                "actual_perturbation_rms_relative_gap": detail["actual_perturbation_rms_relative_gap"],
            })

        indexed_conditions = validate_conditions(all_conditions, split_manifest)
        validate_candidates(all_candidates, indexed_conditions, exact_sibling_count=int(cfg.branch.sibling_count))

        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _sync_cuda(str(cfg.device))
        evaluator = CoTracker3IndependentEvaluator(dict(cfg.branch.evaluator))
        evaluator_preflight = evaluator.preflight()
        if not bool(evaluator_preflight.get("available")):
            raise BranchPilotError("PA1-BRANCH independent CoTracker3 evaluator 不可用")
        evaluator._load()
        scores: dict[str, dict[str, Any]] = {}
        states: dict[str, IndependentTrackState] = {}
        qualities: dict[str, dict[str, Any]] = {}
        for candidate in all_candidates:
            candidate_id = str(candidate["candidate_id"])
            score, state = _score_with_state(evaluator, frames_by_candidate[candidate_id], device=str(cfg.device))
            quality = _quality_diagnostics(frames_by_candidate[candidate_id])
            scores[candidate_id], states[candidate_id], qualities[candidate_id] = score, state, quality
            score_path = work_dir / str(candidate["score_path"])
            atomic_write_json(str(score_path), {"score": score, "quality": quality})
            metrics.append(len(scores), {
                "event": "candidate_scored",
                "candidate_id": candidate_id,
                "condition_id": candidate["condition_id"],
                "score_valid": score["valid"],
                "score_seconds": score["seconds"],
                "score_peak_vram_bytes": score["peak_vram_bytes"],
                "track_coverage": score["track_coverage"],
            })

        by_condition: dict[str, list[dict[str, Any]]] = {}
        for candidate in all_candidates:
            by_condition.setdefault(str(candidate["condition_id"]), []).append(candidate)
        distance_rows: list[dict[str, Any]] = []
        correspondence_rows: list[dict[str, Any]] = []
        condition_reports: list[dict[str, Any]] = []
        group_reports: list[dict[str, Any]] = []
        candidate_feasibility: dict[str, dict[str, Any]] = {}
        for condition_id, candidates in sorted(by_condition.items()):
            base = next(row for row in candidates if str(row["candidate_role"]) == "base_guard")
            base_id = str(base["candidate_id"])
            independent = independent_by_condition[condition_id]
            base_independent = {
                "future_rgb_rms": _future_rgb_rms(frames_by_candidate[base_id], independent["frames"]),
                "future_vae_rms": _future_vae_rms(vae_by_candidate[base_id], independent["vae"]),
            }
            base_rerun_floor = {"future_rgb_rms": 0.0, "future_vae_rms": 0.0}
            candidate_feasibility[base_id] = _candidate_feasibility(
                candidate=base,
                score=scores[base_id],
                quality=qualities[base_id],
                base_score=scores[base_id],
                distance=None,
                thresholds=cfg.branch.thresholds,
            )
            sibling_distances: dict[str, dict[str, Any]] = {}
            for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"])):
                candidate_id = str(candidate["candidate_id"])
                if candidate_id == base_id:
                    continue
                pair_distance = {
                    "condition_id": condition_id,
                    "candidate_a": base_id,
                    "candidate_b": candidate_id,
                    "first_frame": _first_frame_metrics(frames_by_candidate[base_id], frames_by_candidate[candidate_id]),
                    "future_rgb_rms": _future_rgb_rms(frames_by_candidate[base_id], frames_by_candidate[candidate_id]),
                    "future_vae_rms": _future_vae_rms(vae_by_candidate[base_id], vae_by_candidate[candidate_id]),
                    "rerun_floor": base_rerun_floor,
                    "independent_seed_distance": base_independent,
                }
                pair_distance["future_rgb_calibration"] = calibrated_future_distance(
                    candidate_distance=pair_distance["future_rgb_rms"],
                    rerun_floor=base_rerun_floor["future_rgb_rms"],
                    independent_distance=base_independent["future_rgb_rms"],
                    minimum_ratio=float(cfg.branch.thresholds.minimum_future_distance_ratio_to_independent),
                    maximum_ratio=float(cfg.branch.thresholds.maximum_future_distance_ratio_to_independent),
                )
                pair_distance["future_vae_calibration"] = calibrated_future_distance(
                    candidate_distance=pair_distance["future_vae_rms"],
                    rerun_floor=base_rerun_floor["future_vae_rms"],
                    independent_distance=base_independent["future_vae_rms"],
                    minimum_ratio=float(cfg.branch.thresholds.minimum_future_distance_ratio_to_independent),
                    maximum_ratio=float(cfg.branch.thresholds.maximum_future_distance_ratio_to_independent),
                )
                sibling_distances[candidate_id] = pair_distance
                distance_rows.append(pair_distance)
                correspondence = track_correspondence(
                    states[base_id], states[candidate_id], maximum_query_delta_px=float(cfg.branch.thresholds.maximum_query_grid_delta_px),
                )
                correspondence_rows.append({"condition_id": condition_id, "candidate_a": base_id, "candidate_b": candidate_id, **correspondence})
                candidate_feasibility[candidate_id] = _candidate_feasibility(
                    candidate=candidate,
                    score=scores[candidate_id],
                    quality=qualities[candidate_id],
                    base_score=scores[base_id],
                    distance=pair_distance,
                    thresholds=cfg.branch.thresholds,
                )
            groups: dict[str, list[dict[str, Any]]] = {}
            for candidate in candidates:
                if str(candidate["candidate_role"]) == "sibling":
                    groups.setdefault(str(candidate["antithetic_group_id"]), []).append(candidate)
            group_passes = []
            for group_id, members in sorted(groups.items()):
                if len(members) != 2:
                    raise BranchPilotError("antithetic group 不是一正一负两个 sibling")
                positive = next(row for row in members if str(row["branch_direction"]) == "positive")
                negative = next(row for row in members if str(row["branch_direction"]) == "negative")
                positive_id, negative_id = str(positive["candidate_id"]), str(negative["candidate_id"])
                pair = {
                    "condition_id": condition_id,
                    "candidate_a": positive_id,
                    "candidate_b": negative_id,
                    "first_frame": _first_frame_metrics(frames_by_candidate[positive_id], frames_by_candidate[negative_id]),
                    "future_rgb_rms": _future_rgb_rms(frames_by_candidate[positive_id], frames_by_candidate[negative_id]),
                    "future_vae_rms": _future_vae_rms(vae_by_candidate[positive_id], vae_by_candidate[negative_id]),
                }
                distance_rows.append(pair)
                correspondence = track_correspondence(
                    states[positive_id], states[negative_id], maximum_query_delta_px=float(cfg.branch.thresholds.maximum_query_grid_delta_px),
                )
                correspondence_rows.append({"condition_id": condition_id, "candidate_a": positive_id, "candidate_b": negative_id, **correspondence})
                pos_distance = sibling_distances[positive_id]
                neg_distance = sibling_distances[negative_id]
                low, high = sorted((float(pos_distance["future_vae_rms"]), float(neg_distance["future_vae_rms"])))
                symmetry_ratio = low / max(high, 1.0e-12)
                direct_corr_ok = bool(correspondence.get("valid")) and float(correspondence.get("coverage") or 0.0) >= float(cfg.branch.thresholds.minimum_track_correspondence_coverage) and float(correspondence.get("label_agreement") or 0.0) >= float(cfg.branch.thresholds.minimum_subject_background_label_agreement)
                group_checks = {
                    "positive_feasible": bool(candidate_feasibility[positive_id]["feasible"]),
                    "negative_feasible": bool(candidate_feasibility[negative_id]["feasible"]),
                    "positive_base_correspondence": next(row for row in correspondence_rows if row["candidate_a"] == base_id and row["candidate_b"] == positive_id)["valid"] and float(next(row for row in correspondence_rows if row["candidate_a"] == base_id and row["candidate_b"] == positive_id)["coverage"] or 0.0) >= float(cfg.branch.thresholds.minimum_track_correspondence_coverage),
                    "negative_base_correspondence": next(row for row in correspondence_rows if row["candidate_a"] == base_id and row["candidate_b"] == negative_id)["valid"] and float(next(row for row in correspondence_rows if row["candidate_a"] == base_id and row["candidate_b"] == negative_id)["coverage"] or 0.0) >= float(cfg.branch.thresholds.minimum_track_correspondence_coverage),
                    "sibling_correspondence": direct_corr_ok,
                    "antithetic_distance_symmetry": symmetry_ratio >= float(cfg.branch.thresholds.minimum_antithetic_distance_symmetry_ratio),
                }
                report = {
                    "condition_id": condition_id,
                    "antithetic_group_id": group_id,
                    "positive_candidate_id": positive_id,
                    "negative_candidate_id": negative_id,
                    "checks": group_checks,
                    "passed": all(group_checks.values()),
                    "future_vae_distance_symmetry_ratio": symmetry_ratio,
                    "distance_indistinguishable": (
                        str(pos_distance["future_vae_calibration"]["reason"]) == "candidate_indistinguishable"
                        and str(neg_distance["future_vae_calibration"]["reason"]) == "candidate_indistinguishable"
                    ),
                    "structure_mismatch": not direct_corr_ok or float(pair["first_frame"]["rgb_rms"]) > float(cfg.branch.thresholds.maximum_first_frame_rgb_rms),
                    "provisional_direction_relation": _provisional_direction_relation(
                        scores[positive_id], scores[negative_id], thresholds=cfg.branch.thresholds,
                    ),
                }
                group_reports.append(report)
                group_passes.append(bool(report["passed"]))
            condition_reports.append({
                "condition_id": condition_id,
                "base_guard_exact": bool(condition_generation[condition_id]["base_guard_exact"]),
                "actual_perturbation_rms_relative_gap": condition_generation[condition_id]["actual_perturbation_rms_relative_gap"],
                "actual_perturbation_max_abs_mean": condition_generation[condition_id]["actual_perturbation_max_abs_mean"],
                "base_feasible": bool(candidate_feasibility[base_id]["feasible"]),
                "all_antithetic_groups_pass": bool(group_passes) and all(group_passes),
                "passed": bool(candidate_feasibility[base_id]["feasible"])
                and bool(group_passes)
                and all(group_passes)
                and float(condition_generation[condition_id]["actual_perturbation_rms_relative_gap"]) <= float(cfg.branch.thresholds.maximum_actual_perturbation_rms_relative_gap)
                and float(condition_generation[condition_id]["actual_perturbation_max_abs_mean"]) <= float(cfg.branch.thresholds.maximum_actual_perturbation_abs_mean),
            })

        for row in distance_rows:
            _json_line(work_dir / "pairwise_distances.jsonl", row)
        for row in correspondence_rows:
            _json_line(work_dir / "track_correspondence.jsonl", row)
        for candidate in all_candidates:
            candidate_id = str(candidate["candidate_id"])
            _json_line(work_dir / "candidate_diagnostics.jsonl", {
                "candidate_id": candidate_id,
                "condition_id": candidate["condition_id"],
                "feasibility": candidate_feasibility[candidate_id],
                "score": scores[candidate_id],
                "quality": qualities[candidate_id],
            })
        machine = _machine_summary(condition_reports=condition_reports, group_reports=group_reports, branch=cfg.branch)
        profile = {
            "task_id": str(cfg.branch.task_id),
            "family": str(cfg.branch.family),
            "fork_fraction": float(cfg.branch.fork_fraction),
            "fork_step": resolve_fork_step(int(cfg.branch.num_inference_steps), float(cfg.branch.fork_fraction)),
            "strength_name": str(cfg.branch.strength_name),
            "strength_rho": float(cfg.branch.strength_rho),
            "evaluator_preflight": evaluator_preflight,
            "conditions": condition_reports,
            "antithetic_groups": group_reports,
            "machine": machine,
        }
        atomic_write_json(str(work_dir / "profile.json"), profile)
        if bool(machine["machine_pass"]):
            review_cases = _review_cases(
                work_dir=work_dir,
                conditions=all_conditions,
                frames_by_candidate=frames_by_candidate,
                branch_groups=all_candidates,
                fps=int(metadata["fps_input"]),
                review_seed=int(cfg.branch.review.seed),
            )
            _write_review_materials(work_dir, review_cases)
            status = "awaiting_reviews"
            next_gate = "PA1-BRANCH-02 human review"
        else:
            status = "blocked"
            next_gate = "PA1-BRANCH-02 calibration"
        summary = {
            "status": status,
            "task_id": str(cfg.branch.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": cfg_fp,
            "scene_split_fingerprint": split_provenance["split_fingerprint"],
            "horizon_profile_fingerprint": horizon["profile_fingerprint"],
            "condition_count": len(all_conditions),
            "candidate_count": len(all_candidates),
            "machine": machine,
            "profile_fingerprint": sha256_json(profile),
            "next_gate": next_gate,
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "machine_summary.json"), summary)
        atomic_write_json(str(work_dir / "summary.json"), summary)
        _clean_terminal_markers(work_dir)
        marker = "awaiting_reviews" if status == "awaiting_reviews" else "COMPLETE"
        atomic_write_text(str(work_dir / marker), sha256_json(summary) + "\n")
        if status == "awaiting_reviews":
            atomic_write_text(str(work_dir / "MACHINE_COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({"status": status, "ended_at": utc_now(), "exit_reason": "human_review_required" if status == "awaiting_reviews" else str(machine["calibration_action"])})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {
            "status": "failed",
            "task_id": str(cfg.branch.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": cfg_fp,
            "error": repr(exc),
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "summary.json"), failure)
        atomic_write_text(str(work_dir / "FAILED"), sha256_json(failure) + "\n")
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PA1-BRANCH common-prefix sibling pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.aggregate_only:
        parser.error("--preflight 与 --aggregate-only 不能同时使用")
    cfg = load_config(args.config)
    if args.preflight:
        result = preflight_physics_dpo_branch(cfg)
    elif args.aggregate_only:
        result = aggregate_physics_dpo_branch_reviews(cfg)
    else:
        result = run_physics_dpo_branch(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
