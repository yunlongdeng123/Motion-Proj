"""PA1：冻结 Base guard 的 8/14 帧 horizon profile。

本模块只读取冻结 scene split 与每个 condition 的首帧 RGB。它不会读取 future
GT、不会构造 sibling candidate、不会写 cache，也不会更新任何模型参数。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from ..backbones import build_backbone
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..data.physics_dpo_schema import (
    PREFERENCE_SCHEMA_VERSION,
    PhysicsDpoSchemaError,
    make_condition_id,
    validate_candidates,
    validate_conditions,
    validate_scene_split,
)
from ..eval.independent_tracks import (
    CoTracker3IndependentEvaluator,
    aggregate_dynamics,
    summarize_camera_compensated_dynamics,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video, write_video
from .svd_conditioning_parity import _base_model_fingerprint, trace_backbone_generation


TRACE_FIELDS = (
    "condition_noise",
    "initial_video_latents",
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
    "final_latent",
    "decoded_frames",
)
METRIC_FIELDS = (
    "survival_rate",
    "camera_compensated_image_plane_velocity_rms_px",
    "camera_compensated_image_plane_acceleration_rms_px",
    "camera_compensated_image_plane_jerk_rms_px",
)


class HorizonProfileError(RuntimeError):
    """PA1 的 provenance、Base guard 或 profile 门禁不满足。"""


def _tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    # NumPy 不支持 CPU bfloat16；按连续 tensor 的底层字节哈希，保留 dtype/shape
    # 前缀即可避免与其他解释方式混淆。
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_line(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sync_cuda(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_bytes(device: str) -> int | None:
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _tensor_or_sequence_fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_fingerprint(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not all(isinstance(item, torch.Tensor) for item in value):
            raise HorizonProfileError("完整 denoising trace 含非 Tensor sequence")
        return [_tensor_fingerprint(item) for item in value]
    raise HorizonProfileError("完整 denoising trace 含未知字段类型")


def fingerprint_denoising_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """将 C0 已审计的 trace 压缩为逐字段、逐步可复核的不可变 hash。"""
    missing = [field for field in TRACE_FIELDS if field not in trace]
    if missing:
        raise HorizonProfileError(f"Base guard trace 缺少字段: {', '.join(missing)}")
    fields = {field: _tensor_or_sequence_fingerprint(trace[field]) for field in TRACE_FIELDS}
    return {
        "trace_fields": fields,
        "full_denoising_trace_fingerprint": sha256_json(fields),
        "step_count": len(fields["scheduler_timesteps"]),
        "initial_latent_hash": fields["initial_video_latents"],
        "final_latent_hash": fields["final_latent"],
        "decoded_frames_hash": fields["decoded_frames"],
    }


def select_profile_conditions(
    split_manifest: Mapping[str, Any],
    *,
    partition: str,
    condition_count: int,
    required_start_index: int,
) -> list[dict[str, Any]]:
    """每个 scene 只选首个 8-frame clip，保证同一首帧也可进入 14-frame profile。"""
    validate_scene_split(split_manifest)
    partitions = split_manifest.get("partitions")
    if not isinstance(partitions, Mapping) or partition not in partitions:
        raise HorizonProfileError(f"未知 PA1 partition: {partition}")
    source = partitions[partition]
    if not isinstance(source, Mapping):
        raise HorizonProfileError("PA1 partition 必须是 object")
    rows = source.get("clips")
    if not isinstance(rows, list):
        raise HorizonProfileError("PA1 partition 缺少 clips")
    by_scene: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        if int(raw.get("start_index", -1)) != int(required_start_index):
            continue
        scene_token = str(raw.get("scene_token", ""))
        if not scene_token:
            continue
        candidate = dict(raw)
        existing = by_scene.get(scene_token)
        if existing is None or str(candidate.get("clip_id", "")) < str(existing.get("clip_id", "")):
            by_scene[scene_token] = candidate
    selected = sorted(by_scene.values(), key=lambda row: (str(row["scene_token"]), str(row["clip_id"])))
    if int(condition_count) <= 0 or len(selected) < int(condition_count):
        raise HorizonProfileError("冻结 partition 中没有足够的跨 scene PA1 condition")
    return selected[:int(condition_count)]


def relative_delta(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    if not (math.isfinite(float(first)) and math.isfinite(float(second))):
        return None
    return abs(float(first) - float(second)) / max(abs(float(first)), abs(float(second)), 1.0e-8)


def compare_score_repeatability(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    maximum_relative_delta: float,
) -> dict[str, Any]:
    """invalid 始终保持 invalid，绝不将缺失 track 转成 0。"""
    first_aggregate = first.get("aggregate")
    second_aggregate = second.get("aggregate")
    if not isinstance(first_aggregate, Mapping) or not isinstance(second_aggregate, Mapping):
        return {"valid": False, "passed": False, "reason": "aggregate_invalid", "fields": {}}
    fields: dict[str, Any] = {}
    deltas: list[float] = []
    for field in METRIC_FIELDS:
        delta = relative_delta(first_aggregate.get(field), second_aggregate.get(field))
        fields[field] = {
            "first": first_aggregate.get(field),
            "second": second_aggregate.get(field),
            "relative_delta": delta,
        }
        if delta is None:
            return {"valid": False, "passed": False, "reason": f"metric_invalid:{field}", "fields": fields}
        deltas.append(delta)
    max_delta = max(deltas, default=math.inf)
    return {
        "valid": True,
        "passed": bool(max_delta <= float(maximum_relative_delta)),
        "reason": None if max_delta <= float(maximum_relative_delta) else "score_repeatability_exceeded",
        "maximum_relative_delta": max_delta,
        "threshold": float(maximum_relative_delta),
        "fields": fields,
    }


def decide_horizon(
    profiles: Mapping[int, Mapping[str, Any]],
    *,
    maximum_14_peak_vram_gb: float,
    maximum_14_generation_slowdown: float,
) -> dict[str, Any]:
    """实现 V2 §5.1 的唯一资源决策，不以 rollout 指标挑选 horizon。"""
    base = profiles.get(8)
    fourteen = profiles.get(14)
    if not isinstance(base, Mapping):
        return {"status": "blocked", "reason": "missing_8_frame_profile", "selected_num_frames": None}
    base_ok = bool(base.get("base_guard_exact")) and bool(base.get("score_valid")) and bool(base.get("score_repeatability_pass"))
    if str(base.get("status")) != "completed" or not base_ok:
        return {"status": "blocked", "reason": "8_frame_gate_failed", "selected_num_frames": None}
    if not isinstance(fourteen, Mapping):
        return {"status": "blocked", "reason": "missing_14_frame_profile", "selected_num_frames": None}
    if str(fourteen.get("status")) == "resource_rejected":
        return {
            "status": "done", "selected_num_frames": 8, "claim_scope": "short-horizon dynamics alignment",
            "reason": "14_frame_resource_rejected",
        }
    fourteen_ok = bool(fourteen.get("base_guard_exact")) and bool(fourteen.get("score_valid")) and bool(fourteen.get("score_repeatability_pass"))
    if str(fourteen.get("status")) != "completed" or not fourteen_ok:
        return {"status": "blocked", "reason": "14_frame_reliability_gate_failed", "selected_num_frames": None}
    base_seconds = float(base.get("generation_seconds_mean", math.inf))
    fourteen_seconds = float(fourteen.get("generation_seconds_mean", math.inf))
    peak_bytes = fourteen.get("generation_peak_vram_bytes_max")
    if not math.isfinite(base_seconds) or base_seconds <= 0.0 or not math.isfinite(fourteen_seconds):
        return {"status": "blocked", "reason": "generation_time_invalid", "selected_num_frames": None}
    if not isinstance(peak_bytes, int) or peak_bytes <= 0:
        return {"status": "blocked", "reason": "14_frame_vram_invalid", "selected_num_frames": None}
    peak_gb = float(peak_bytes) / float(1024**3)
    slowdown = fourteen_seconds / base_seconds
    if peak_gb <= float(maximum_14_peak_vram_gb) and slowdown <= float(maximum_14_generation_slowdown):
        return {
            "status": "done", "selected_num_frames": 14, "claim_scope": "short-horizon dynamics alignment",
            "reason": "14_frame_resource_gate_passed", "14_frame_peak_vram_gb": peak_gb,
            "14_to_8_generation_slowdown": slowdown,
        }
    return {
        "status": "done", "selected_num_frames": 8, "claim_scope": "short-horizon dynamics alignment",
        "reason": "14_frame_resource_gate_not_met", "14_frame_peak_vram_gb": peak_gb,
        "14_to_8_generation_slowdown": slowdown,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HorizonProfileError(f"{label} 缺少: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HorizonProfileError(f"{label} 必须是 object")
    return value


def _validate_completed_run(path: Path, *, label: str) -> dict[str, Any]:
    summary = _load_json(path / "summary.json", label=f"{label}.summary")
    complete = path / "COMPLETE"
    if not complete.is_file() or complete.read_text(encoding="utf-8").strip() != sha256_json(summary):
        raise HorizonProfileError(f"{label} 缺少或不匹配 COMPLETE")
    return summary


def _validate_pa0(cfg: Any) -> dict[str, Any]:
    path = Path(str(cfg.pa0_review_run))
    summary = _validate_completed_run(path, label="PA0 review run")
    if (
        summary.get("status") != "done"
        or not bool(summary.get("p0_pass"))
        or not bool(summary.get("e0_pass"))
        or str(summary.get("decision_fingerprint")) != str(cfg.expected_pa0_decision_fingerprint)
    ):
        raise HorizonProfileError("PA0 review decision 不满足 PA1 前置门禁")
    return {"path": str(path), "summary_sha256": file_fingerprint(str(path / "summary.json")), **summary}


def _load_scene_split(cfg: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    run_path = Path(str(cfg.scene_split_run))
    summary = _validate_completed_run(run_path, label="scene split run")
    if summary.get("status") != "done":
        raise HorizonProfileError("scene split run 未完成")
    manifest_path = run_path / "scene_split_manifest.json"
    split = _load_json(manifest_path, label="scene split manifest")
    validate_scene_split(split)
    expected = str(cfg.expected_scene_split_fingerprint)
    if str(split.get("split_fingerprint")) != expected or str(summary.get("split_fingerprint")) != expected:
        raise HorizonProfileError("scene split fingerprint 与预注册值不一致")
    return split, {
        "run_path": str(run_path),
        "summary_sha256": file_fingerprint(str(run_path / "summary.json")),
        "manifest_sha256": file_fingerprint(str(manifest_path)),
        "split_fingerprint": expected,
    }


def _dataset_for_horizon(data_cfg: Any, *, num_frames: int) -> NuScenesFutureVideoDataset:
    copied = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(data_cfg, resolve=True)))
    copied.num_frames = int(num_frames)
    metadata = Path(str(copied.dataroot)) / str(copied.version)
    if not metadata.is_dir():
        raise HorizonProfileError(
            f"PA1 需要冻结的 nuScenes {copied.version} metadata: {metadata}; "
            "v1.0-mini 不能替代 trainval scene split"
        )
    return NuScenesFutureVideoDataset(copied)


def _record_by_sample_id(dataset: NuScenesFutureVideoDataset) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): dict(row) for row in dataset.clip_records}


def preflight_physics_dpo_horizon(cfg: Any) -> dict[str, Any]:
    """只读检查 PA1 所需 provenance、数据、权重与 evaluator，不创建正式 run。"""
    result: dict[str, Any] = {
        "task_id": str(cfg.horizon.task_id),
        "status": "ready",
        "uses_gpu": False,
        "uses_future_gt": False,
        "blockers": [],
    }
    try:
        split_manifest, split_provenance = _load_scene_split(cfg.horizon)
        pa0_provenance = _validate_pa0(cfg.horizon)
        selected = select_profile_conditions(
            split_manifest,
            partition=str(cfg.horizon.condition_partition),
            condition_count=int(cfg.horizon.condition_count),
            required_start_index=int(cfg.horizon.required_start_index),
        )
        result.update({"scene_split": split_provenance, "pa0_review": pa0_provenance, "selected_conditions": selected})
    except Exception as exc:
        result["blockers"].append({"kind": "provenance", "error": repr(exc)})
        result["status"] = "blocked"
        return result
    try:
        datasets = {frames: _dataset_for_horizon(cfg.data, num_frames=frames) for frames in (8, 14)}
        records = {frames: _record_by_sample_id(dataset) for frames, dataset in datasets.items()}
        data_checks = []
        for selected in result["selected_conditions"]:
            clip_id = str(selected["clip_id"])
            first_token = str(selected["sample_tokens"][0])
            matched = all(
                clip_id in records[frames] and str(records[frames][clip_id]["sample_tokens"][0]) == first_token
                for frames in (8, 14)
            )
            if not matched:
                raise HorizonProfileError(f"condition {clip_id} 不能共享 8/14-frame 首帧")
            frame = _load_condition_frame(datasets[8], first_token)
            data_checks.append({"clip_id": clip_id, "condition_frame_sha256": _tensor_fingerprint(frame)})
        result["data"] = {"ready": True, "condition_checks": data_checks}
    except Exception as exc:
        result["blockers"].append({"kind": "nuscenes_trainval", "error": repr(exc)})
    evaluator_preflight = CoTracker3IndependentEvaluator(dict(cfg.horizon.evaluator)).preflight()
    result["evaluator"] = evaluator_preflight
    if not bool(evaluator_preflight.get("available")):
        result["blockers"].append({"kind": "cotracker3", "reasons": evaluator_preflight.get("reasons", [])})
    if result["blockers"]:
        result["status"] = "blocked"
    return result


def _load_condition_frame(dataset: NuScenesFutureVideoDataset, sample_token: str) -> torch.Tensor:
    """只读取 conditioning sample 的 CAM_FRONT RGB，绝不读取 future frame、ego 或 box。"""
    from PIL import Image

    sample = dataset.nusc.get("sample", sample_token)
    camera_token = sample["data"][dataset.camera]
    image_path = dataset.nusc.get_sample_data_path(camera_token)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((dataset.W, dataset.H), Image.BILINEAR)
        array = np.asarray(rgb).copy()
    return torch.from_numpy(array).float().permute(2, 0, 1).div(127.5).sub(1.0)


def _make_condition_record(
    *,
    selected: Mapping[str, Any],
    split: str,
    camera: str,
    num_frames: int,
    fps: int,
    condition_frame_hash: str,
    scheduler_fingerprint: str,
    base_model_fingerprint: str,
    git_commit: str,
    config_fingerprint_value: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "scene_id": str(selected["scene_name"]),
        "scene_token": str(selected["scene_token"]),
        "clip_id": str(selected["clip_id"]),
        "split": split,
        "camera": str(camera),
        "conditioning_frame": 0,
        "condition_frame_sha256": condition_frame_hash,
        "num_frames": int(num_frames),
        "fps": int(fps),
        "generation_protocol": "svd_official_v1",
        "scheduler_fingerprint": scheduler_fingerprint,
        "base_model_fingerprint": base_model_fingerprint,
        "uses_future_gt": False,
        "git_commit": git_commit,
        "config_fingerprint": config_fingerprint_value,
    }
    record["condition_id"] = make_condition_id(record)
    return record


def _save_tensor(path: Path, value: torch.Tensor) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value.detach().cpu(), temporary)
    os.replace(temporary, path)


def _existing_video_path(path: Path) -> Path:
    if path.is_file():
        return path
    fallback = path.with_suffix(".npy")
    if fallback.is_file():
        return fallback
    raise HorizonProfileError(f"Base guard video writer 未产生 artifact: {path}")


def _score_video(
    evaluator: CoTracker3IndependentEvaluator,
    frames: torch.Tensor,
    *,
    device: str,
) -> dict[str, Any]:
    _reset_peak_memory(device)
    started = time.perf_counter()
    state = evaluator.track(frames)
    _sync_cuda(device)
    seconds = time.perf_counter() - started
    dynamics = summarize_camera_compensated_dynamics(state)
    aggregate = aggregate_dynamics(dynamics)
    visible_lengths = state.visibility.sum(dim=1).to(torch.float32)
    valid = bool(state.valid) and aggregate is not None
    return {
        "valid": valid,
        "seconds": seconds,
        "peak_vram_bytes": _peak_memory_bytes(device),
        "query_count": int(state.visibility.shape[0]),
        "valid_track_count": int((visible_lengths > 0).sum()),
        "median_track_length_frames": float(visible_lengths.median()) if int(visible_lengths.numel()) else None,
        "track_coverage": float(state.visibility.float().mean()) if int(state.visibility.numel()) else None,
        "dynamics": dynamics,
        "aggregate": aggregate,
        "provider_diagnostics": state.diagnostics,
    }


def _resource_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _profile_one_base_guard(
    *,
    backbone: SVDBackbone,
    condition: Mapping[str, Any],
    condition_frame: torch.Tensor,
    generation_seed: int,
    num_frames: int,
    num_inference_steps: int,
    height: int,
    width: int,
    output_dir: Path,
) -> tuple[dict[str, Any], torch.Tensor]:
    """生成一个 Base guard 和 exact rerun；评分在释放 SVD 后执行。"""
    case_id = f"{condition['condition_id']}-s{generation_seed}"
    case_dir = output_dir / f"f{num_frames}" / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "seed": int(generation_seed),
        "num_frames": int(num_frames),
        "num_inference_steps": int(num_inference_steps),
        "height": int(height),
        "width": int(width),
    }
    _reset_peak_memory(str(backbone.device))
    started = time.perf_counter()
    trace = trace_backbone_generation(backbone, condition_frame.to(backbone.device), **common)
    _sync_cuda(str(backbone.device))
    generation_seconds = time.perf_counter() - started
    trace_hashes = fingerprint_denoising_trace(trace)
    frames = trace["decoded_frames"].detach().cpu()
    if not bool(torch.isfinite(frames).all()):
        raise HorizonProfileError("Base guard decoded RGB 包含 NaN/Inf")
    generation_peak = _peak_memory_bytes(str(backbone.device))

    _reset_peak_memory(str(backbone.device))
    rerun_started = time.perf_counter()
    rerun_trace = trace_backbone_generation(backbone, condition_frame.to(backbone.device), **common)
    _sync_cuda(str(backbone.device))
    rerun_seconds = time.perf_counter() - rerun_started
    rerun_hashes = fingerprint_denoising_trace(rerun_trace)
    rerun_peak = _peak_memory_bytes(str(backbone.device))
    rerun_exact = trace_hashes == rerun_hashes
    if not rerun_exact:
        raise HorizonProfileError("Base guard rerun 的完整 denoising trace 不 exact")

    video_path = case_dir / "base_guard.mp4"
    write_video(to_uint8_video(frames), str(video_path), fps=int(backbone.generation_settings()["fps"]))
    video_artifact = _existing_video_path(video_path)
    _save_tensor(case_dir / "final_denoising_latent.pt", trace["final_latent"])
    vae_latents = backbone.encode(frames.unsqueeze(0)).detach().cpu()
    _save_tensor(case_dir / "vae_latents.pt", vae_latents)
    artifact_paths = [video_artifact, case_dir / "final_denoising_latent.pt", case_dir / "vae_latents.pt"]
    trace_payload = {
        "case_id": case_id,
        "condition_id": condition["condition_id"],
        "generation_seed": int(generation_seed),
        "num_frames": int(num_frames),
        "num_inference_steps": int(num_inference_steps),
        "trace": trace_hashes,
        "rerun_trace": rerun_hashes,
        "rerun_exact": rerun_exact,
    }
    atomic_write_json(str(case_dir / "trace.json"), trace_payload)
    artifact_paths.append(case_dir / "trace.json")
    storage_bytes = sum(path.stat().st_size for path in artifact_paths)
    candidate = {
        "candidate_id": f"base-{condition['condition_id']}",
        "condition_id": condition["condition_id"],
        "scene_id": condition["scene_id"],
        "split": condition["split"],
        "candidate_role": "base_guard",
        "rgb_video_path": str(video_artifact.relative_to(output_dir.parent)),
        "vae_latent_path": str((case_dir / "vae_latents.pt").relative_to(output_dir.parent)),
        "diagnostics_path": str((case_dir / "trace.json").relative_to(output_dir.parent)),
        "generation_protocol": condition["generation_protocol"],
        "base_model_fingerprint": condition["base_model_fingerprint"],
        "scheduler_fingerprint": condition["scheduler_fingerprint"],
        "initial_latent_hash": trace_hashes["initial_latent_hash"],
        "prefix_latent_hash": trace_hashes["initial_latent_hash"],
        "prefix_trace_hash": trace_hashes["full_denoising_trace_fingerprint"],
        "fork_step": 0,
        "branch_family": "base_guard",
        "branch_direction": "base",
        "branch_strength": 0.0,
        "perturbation_rms": 0.0,
        "antithetic_group_id": f"base-{condition['condition_id']}",
        "generation_seed": int(generation_seed),
        "guidance_schedule": [
            float(backbone.generation_settings()["min_guidance_scale"]),
            float(backbone.generation_settings()["max_guidance_scale"]),
        ],
        "num_frames": int(num_frames),
        "fps": int(backbone.generation_settings()["fps"]),
        "uses_future_gt": False,
        "git_commit": condition["git_commit"],
        "config_fingerprint": condition["config_fingerprint"],
    }
    return {
        "case_id": case_id,
        "condition": dict(condition),
        "candidate": candidate,
        "generation_seed": int(generation_seed),
        "generation_seconds": generation_seconds,
        "generation_peak_vram_bytes": generation_peak,
        "rerun_seconds": rerun_seconds,
        "rerun_peak_vram_bytes": rerun_peak,
        "base_guard_exact": rerun_exact,
        "trace": trace_hashes,
        "storage_bytes": int(storage_bytes),
        "artifact_root": str(case_dir.relative_to(output_dir.parent)),
    }, frames


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _maximum_int(values: Sequence[int | None]) -> int | None:
    available = [int(value) for value in values if isinstance(value, int)]
    return max(available) if available else None


def run_physics_dpo_horizon(cfg: Any) -> dict[str, Any]:
    """执行 PA1-HORIZON-01；逻辑 blocked 会落盘并正常返回。"""
    horizon = cfg.horizon
    if not bool(horizon.base_guard_only):
        raise HorizonProfileError("PA1-HORIZON 必须 base_guard_only=true")
    if str(cfg.model.generation.protocol) != "svd_official_v1" or bool(cfg.model.lora.enable):
        raise HorizonProfileError("PA1 只允许冻结 Base 的 svd_official_v1")
    frame_counts = [int(value) for value in horizon.frame_counts]
    if sorted(frame_counts) != [8, 14] or len(set(frame_counts)) != 2:
        raise HorizonProfileError("PA1 必须恰好 profile 8 与 14 frames")
    if int(horizon.num_inference_steps) != 25:
        raise HorizonProfileError("PA1 预注册为 25 inference steps")
    git = git_state(".")
    if git.get("dirty"):
        raise HorizonProfileError("正式 PA1-HORIZON 拒绝在 dirty worktree 上运行")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"PA1 run directory 已存在: {work_dir}")

    split_manifest, split_provenance = _load_scene_split(horizon)
    pa0_provenance = _validate_pa0(horizon)
    selected = select_profile_conditions(
        split_manifest,
        partition=str(horizon.condition_partition),
        condition_count=int(horizon.condition_count),
        required_start_index=int(horizon.required_start_index),
    )
    config_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "base_guards").mkdir()
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=config_fp,
        cache_fingerprint="not-applicable:pa1-base-guard-profile",
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split=str(horizon.condition_partition),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(horizon.task_id),
        "uses_future_gt": False,
        "base_guard_only": True,
        "scene_split": split_provenance,
        "pa0_review": pa0_provenance,
        "condition_selection_rule": {
            "partition": str(horizon.condition_partition),
            "condition_count": int(horizon.condition_count),
            "required_start_index": int(horizon.required_start_index),
            "ordering": "one start-index-matched clip per scene; ascending (scene_token, clip_id)",
        },
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    conditions_path = work_dir / "conditions.jsonl"
    candidates_path = work_dir / "base_guard_manifest.jsonl"
    try:
        dataset8 = _dataset_for_horizon(cfg.data, num_frames=8)
        dataset14 = _dataset_for_horizon(cfg.data, num_frames=14)
        records8 = _record_by_sample_id(dataset8)
        records14 = _record_by_sample_id(dataset14)
        selected_with_frame: list[dict[str, Any]] = []
        for row in selected:
            sample_id = str(row["clip_id"])
            first_token = str(row["sample_tokens"][0])
            if sample_id not in records8 or sample_id not in records14:
                raise HorizonProfileError("冻结 PA1 condition 不能同时构成 8/14-frame window")
            if str(records8[sample_id]["sample_tokens"][0]) != first_token or str(records14[sample_id]["sample_tokens"][0]) != first_token:
                raise HorizonProfileError("8/14 profile condition 的首帧 token 不一致")
            condition_frame = _load_condition_frame(dataset8, first_token)
            selected_with_frame.append({**row, "condition_frame": condition_frame, "condition_frame_sha256": _tensor_fingerprint(condition_frame)})

        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        backbone = build_backbone(cfg.model, load=True, device=str(cfg.device))
        if not isinstance(backbone, SVDBackbone):
            raise HorizonProfileError("PA1 当前只支持 SVDBackbone")
        backbone.unet.eval()
        backbone.vae.eval()
        backbone.image_encoder.eval()
        generation_metadata = backbone.generation_protocol_metadata()
        base_model_fp = _base_model_fingerprint(str(cfg.model.pretrained))
        manifest_data.update({
            "base_model_fingerprint": base_model_fp,
            "generation_protocol": generation_metadata,
            "selected_source_conditions": [{key: value for key, value in row.items() if key != "condition_frame"} for row in selected_with_frame],
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)

        all_conditions: list[dict[str, Any]] = []
        generated_cases: dict[int, list[dict[str, Any]]] = {8: [], 14: []}
        frames_by_case: dict[str, torch.Tensor] = {}
        for frame_count in frame_counts:
            for condition_index, selected_row in enumerate(selected_with_frame):
                condition = _make_condition_record(
                    selected=selected_row,
                    split=str(horizon.condition_partition),
                    camera=str(cfg.data.cameras[0]),
                    num_frames=frame_count,
                    fps=int(generation_metadata["fps_input"]),
                    condition_frame_hash=str(selected_row["condition_frame_sha256"]),
                    scheduler_fingerprint=str(generation_metadata["scheduler_config_fingerprint"]),
                    base_model_fingerprint=base_model_fp,
                    git_commit=str(git["commit"]),
                    config_fingerprint_value=config_fp,
                )
                all_conditions.append(condition)
                try:
                    case, frames = _profile_one_base_guard(
                        backbone=backbone,
                        condition=condition,
                        condition_frame=selected_row["condition_frame"],
                        generation_seed=int(horizon.generation_seed_start) + condition_index,
                        num_frames=frame_count,
                        num_inference_steps=int(horizon.num_inference_steps),
                        height=int(cfg.data.height),
                        width=int(cfg.data.width),
                        output_dir=work_dir / "base_guards",
                    )
                except Exception as exc:
                    if frame_count == 14 and _resource_error(exc):
                        generated_cases[frame_count].append({"status": "resource_rejected", "error": repr(exc), "condition": condition})
                        metrics.append(frame_count, {"event": "generation_resource_rejected", "num_frames": frame_count, "condition_id": condition["condition_id"], "error": repr(exc)})
                        break
                    raise
                generated_cases[frame_count].append({"status": "completed", **case})
                frames_by_case[str(case["case_id"])] = frames
                _json_line(conditions_path, condition)
                _json_line(candidates_path, case["candidate"])
                metrics.append(frame_count, {
                    "event": "base_guard_generated", "num_frames": frame_count, "condition_id": condition["condition_id"],
                    "case_id": case["case_id"], "generation_seconds": case["generation_seconds"],
                    "generation_peak_vram_bytes": case["generation_peak_vram_bytes"], "storage_bytes": case["storage_bytes"],
                    "base_guard_exact": case["base_guard_exact"],
                })

        # schema 只接受实际成功生成的 Base guards；14-frame OOM 不伪造 candidate record。
        successful_conditions = [case["condition"] for rows in generated_cases.values() for case in rows if case.get("status") == "completed"]
        successful_candidates = [case["candidate"] for rows in generated_cases.values() for case in rows if case.get("status") == "completed"]
        indexed_conditions = validate_conditions(successful_conditions, split_manifest)
        validate_candidates(successful_candidates, indexed_conditions, exact_sibling_count=0)

        # 生成与评分分阶段执行，避免两个冻结模型同时占用单卡显存。
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _sync_cuda(str(cfg.device))
        evaluator = CoTracker3IndependentEvaluator(dict(horizon.evaluator))
        evaluator_preflight = evaluator.preflight()
        if not bool(evaluator_preflight.get("available")):
            raise HorizonProfileError("PA1 独立 CoTracker3 evaluator 不可用")
        evaluator._load()
        profile_by_frames: dict[int, dict[str, Any]] = {}
        for frame_count in frame_counts:
            rows = generated_cases[frame_count]
            if rows and str(rows[0].get("status")) == "resource_rejected":
                profile_by_frames[frame_count] = {
                    "status": "resource_rejected", "base_guard_exact": False, "score_valid": False,
                    "score_repeatability_pass": False, "reason": rows[0]["error"],
                }
                continue
            completed = [row for row in rows if row.get("status") == "completed"]
            if len(completed) != int(horizon.condition_count):
                profile_by_frames[frame_count] = {
                    "status": "failed", "base_guard_exact": False, "score_valid": False,
                    "score_repeatability_pass": False, "reason": "incomplete_base_guard_generation",
                }
                continue
            score_rows = []
            for case in completed:
                first = _score_video(evaluator, frames_by_case[str(case["case_id"])], device=str(cfg.device))
                second = _score_video(evaluator, frames_by_case[str(case["case_id"])], device=str(cfg.device))
                repeatability = compare_score_repeatability(
                    first, second, maximum_relative_delta=float(horizon.thresholds.maximum_score_repeatability_relative_delta),
                )
                score = {"first": first, "rerun": second, "repeatability": repeatability}
                case["score"] = score
                score_rows.append(score)
                trace_path = work_dir / case["artifact_root"] / "score.json"
                atomic_write_json(str(trace_path), score)
                case["storage_bytes_with_score"] = int(case["storage_bytes"]) + int(trace_path.stat().st_size)
                metrics.append(frame_count, {
                    "event": "base_guard_scored", "num_frames": frame_count, "condition_id": case["condition"]["condition_id"],
                    "case_id": case["case_id"], "score_seconds": first["seconds"],
                    "score_peak_vram_bytes": first["peak_vram_bytes"], "score_valid": first["valid"],
                    "score_repeatability_pass": repeatability["passed"],
                    "score_repeatability_max_relative_delta": repeatability.get("maximum_relative_delta"),
                })
            profile_by_frames[frame_count] = {
                "status": "completed",
                "condition_count": len(completed),
                "base_guard_exact": all(bool(row["base_guard_exact"]) for row in completed),
                "score_valid": all(bool(score["first"]["valid"]) and bool(score["rerun"]["valid"]) for score in score_rows),
                "score_repeatability_pass": all(bool(score["repeatability"]["passed"]) for score in score_rows),
                "generation_seconds_mean": _mean([float(row["generation_seconds"]) for row in completed]),
                "generation_seconds_max": max(float(row["generation_seconds"]) for row in completed),
                "generation_peak_vram_bytes_max": _maximum_int([row["generation_peak_vram_bytes"] for row in completed]),
                "score_seconds_mean": _mean([float(score["first"]["seconds"]) for score in score_rows]),
                "score_peak_vram_bytes_max": _maximum_int([score["first"]["peak_vram_bytes"] for score in score_rows]),
                "median_track_length_frames_mean": _mean([float(score["first"]["median_track_length_frames"]) for score in score_rows if score["first"]["median_track_length_frames"] is not None]),
                "track_coverage_mean": _mean([float(score["first"]["track_coverage"]) for score in score_rows if score["first"]["track_coverage"] is not None]),
                "storage_bytes_per_video_mean": _mean([float(row["storage_bytes_with_score"]) for row in completed]),
                "cases": completed,
            }

        decision = decide_horizon(
            profile_by_frames,
            maximum_14_peak_vram_gb=float(horizon.thresholds.maximum_14_peak_vram_gb),
            maximum_14_generation_slowdown=float(horizon.thresholds.maximum_14_generation_slowdown),
        )
        profile = {
            "task_id": str(horizon.task_id),
            "evaluator_preflight": evaluator_preflight,
            "profiles": {str(key): value for key, value in profile_by_frames.items()},
            "decision": decision,
        }
        atomic_write_json(str(work_dir / "profile.json"), profile)
        status = str(decision["status"])
        summary = {
            "status": status,
            "task_id": str(horizon.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": config_fp,
            "scene_split_fingerprint": str(split_manifest["split_fingerprint"]),
            "condition_count": int(horizon.condition_count),
            "frame_counts": frame_counts,
            "decision": decision,
            "profile_fingerprint": sha256_json(profile),
            "next_gate": "PA1-BRANCH-02" if status == "done" else None,
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({"status": "completed" if status == "done" else "failed", "ended_at": utc_now(), "exit_reason": decision["reason"]})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {"status": "failed", "task_id": str(horizon.task_id), "run_id": str(cfg.run_id), "error": repr(exc)}
        atomic_write_json(str(work_dir / "summary.json"), failure)
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="profile SAP-DPO 8/14-frame Base guards")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true", help="只读检查数据与 evaluator，不创建正式 run")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, list(args.overrides))
    result = preflight_physics_dpo_horizon(cfg) if args.preflight else run_physics_dpo_horizon(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
