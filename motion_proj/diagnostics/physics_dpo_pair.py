"""PA2：结构对齐 preference pair 的单卡合法性研究。

本模块只生成和审计真实 SVD rollout。P1 common-prefix sibling 是唯一可能进入
训练 schema 的 candidate family；P0 independent-seed 和 P2 Base re-noise 仅作为
pair-construction 对照，永远不被静默混入训练数据。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..auditor.generated_tracks import RAFTChainGeneratedTrackProvider
from ..backbones import build_backbone
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data.physics_dpo_schema import (
    validate_candidates,
    validate_conditions,
    validate_preferences,
    validate_segments,
)
from ..diagnostics.physics_dpo_branch import (
    _first_frame_metrics,
    _generate_condition_group,
    _load_horizon_provenance,
    _load_scene_split,
    _quality_diagnostics,
    _make_panel,
    _rms,
    _score_with_state,
    _sync_cuda,
    _tensor_fingerprint,
    _validate_pa0,
    _write_candidate_artifact,
    resolve_fork_step,
    trace_backbone_generation_with_fork_perturbation,
    verify_shared_prefix_before_callback_injection,
)
from ..diagnostics.physics_dpo_horizon import (
    _dataset_for_horizon,
    _existing_video_path,
    _json_line,
    _load_condition_frame,
    _make_condition_record,
    _peak_memory_bytes,
    _reset_peak_memory,
    fingerprint_denoising_trace,
    select_profile_conditions,
)
from ..diagnostics.evaluator_validity import _draw_overlay
from ..eval.independent_tracks import CoTracker3IndependentEvaluator
from ..diagnostics.projector_validity import PRIMARY_STRATA, build_candidate_tracks
from ..diagnostics.svd_conditioning_parity import _base_model_fingerprint, trace_backbone_generation
from ..preference.pair_scoring import (
    DECISIVE_LABELS,
    candidate_feasibility,
    decide_global_pair,
    decide_segments,
    select_condition_pair,
    wilson_lower_bound,
)
from ..preference.review import (
    PreferenceReviewError,
    review_summary,
    select_review_pairs,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import write_video


class PairPilotError(RuntimeError):
    """PA2 provenance、candidate 或 review 门槛失败。"""


CONSTRUCTORS = ("P0-independent", "P1-common-prefix", "P2-base-renoise")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairPilotError(f"{label} 必须是 object")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PairPilotError(f"缺少 {label}: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise PairPilotError(f"{label} 必须是 object")
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PairPilotError(f"JSONL row 不是 object: {path}")
            rows.append(value)
    return rows


def _track_label(track: Any) -> str:
    return str(track.category).rsplit("/", 1)[-1]


def _punc_score(
    frames: torch.Tensor,
    provider: RAFTChainGeneratedTrackProvider,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """把 P-UNC 的 point-track 修正转成无 GT 的 normalized projection energy。"""
    state = provider.track(frames)
    quality = _quality_diagnostics(frames)
    frame_count = int(frames.shape[0])
    if state.uses_future_gt:
        raise PairPilotError("PA2 P-UNC scorer 检测到 future GT")
    if not state.tracks:
        return {
            "valid": False,
            "uses_future_gt": False,
            "frame_count": frame_count,
            "projection_energy": None,
            "projection_points": 0,
            "projection_energy_by_frame": [None] * frame_count,
            "projection_points_by_frame": [0] * frame_count,
            "track_coverage": 0.0,
            "track_coverage_by_frame": [0.0] * frame_count,
            "survival_rate": 0.0,
            "survival_by_frame": [0.0] * frame_count,
            "motion_magnitude": None,
            "motion_magnitude_by_frame": [None] * frame_count,
            "net_displacement": None,
            "primary_track_count": 0,
            "median_track_length_frames": None,
            "scorer_confidence": 0.0,
            "punc_invariants": {"frame0_correction_max_px": None, "visibility_changed_count": 0},
            "provider_diagnostics": state.diagnostics,
        }, quality
    punc = build_candidate_tracks(state.tracks, state.confidence, (int(frames.shape[-2]), int(frames.shape[-1])), settings)["P-UNC"]
    energy_sum = [0.0] * frame_count
    energy_count = [0] * frame_count
    coverage_count = [0] * frame_count
    motion_sum = [0.0] * frame_count
    motion_count = [0] * frame_count
    displacement: list[float] = []
    confidences: list[float] = []
    primary_tracks = 0
    frame0_corrections: list[float] = []
    visibility_changed = 0
    constrained = _mapping(settings["constrained"], label="pair.punc.constrained")
    uncertainty_floor = float(constrained["uncertainty_floor_px"])
    uncertainty_scale = float(constrained["uncertainty_confidence_scale_px"])
    for index, (original, projected) in enumerate(zip(state.tracks, punc.tracks)):
        # RAFT provider 可在 CUDA 上保留原轨迹，而 projector validity 会克隆到 CPU。
        # scorer 是离线诊断，统一到 CPU 后再组合 mask，避免隐式跨设备并保持可复现。
        original_center = original.center.detach().cpu().float()
        projected_center = projected.center.detach().cpu().float()
        present = original.present.detach().cpu().bool()
        projected_present = projected.present.detach().cpu().bool()
        visibility_changed += int((present != projected_present).sum())
        for time in range(frame_count):
            coverage_count[time] += int(present[time])
        visible = torch.nonzero(present, as_tuple=False).flatten()
        if int(visible.numel()) >= 2:
            first, last = int(visible[0]), int(visible[-1])
            displacement.append(float(torch.linalg.vector_norm(original_center[last] - original_center[first])))
        for time in range(1, frame_count):
            if bool(present[time - 1] & present[time]):
                motion_sum[time] += float(torch.linalg.vector_norm(original_center[time] - original_center[time - 1]))
                motion_count[time] += 1
        uncertainty = punc.uncertainty[index].detach().cpu().float().clamp_min(1.0e-8)
        if uncertainty_scale > 0:
            inferred_confidence = (1.0 - (uncertainty - uncertainty_floor) / uncertainty_scale).clamp(0.0, 1.0)
            confidences.extend(float(value) for value in inferred_confidence[present])
        label = _track_label(original)
        if label not in PRIMARY_STRATA:
            continue
        primary_tracks += 1
        corrected = punc.corrected[index].detach().cpu().bool() & present & projected_present
        valid = corrected & torch.isfinite(original_center).all(dim=-1) & torch.isfinite(projected_center).all(dim=-1)
        if bool(valid.any()):
            delta = torch.linalg.vector_norm(projected_center[valid] - original_center[valid], dim=-1)
            normalized_squared = (delta / uncertainty[valid]).square()
            for time, value in zip(torch.nonzero(valid, as_tuple=False).flatten().tolist(), normalized_squared.tolist()):
                energy_sum[time] += float(value)
                energy_count[time] += 1
        if bool(present[0] & projected_present[0]):
            frame0_corrections.append(float(torch.linalg.vector_norm(projected_center[0] - original_center[0])))
    total_points = sum(energy_count)
    projection_by_frame = [energy_sum[index] / energy_count[index] if energy_count[index] else None for index in range(frame_count)]
    track_count = len(state.tracks)
    coverage = [count / track_count for count in coverage_count]
    motion = [motion_sum[index] / motion_count[index] if motion_count[index] else None for index in range(frame_count)]
    lengths = [int(track.present.sum()) for track in state.tracks]
    return {
        "valid": bool(total_points > 0 and state.tracks),
        "uses_future_gt": False,
        "frame_count": frame_count,
        "projection_energy": sum(energy_sum) / total_points if total_points else None,
        "projection_points": total_points,
        "projection_energy_by_frame": projection_by_frame,
        "projection_points_by_frame": energy_count,
        "track_coverage": sum(coverage) / len(coverage) if coverage else 0.0,
        "track_coverage_by_frame": coverage,
        "survival_rate": coverage[-1] if coverage else 0.0,
        "survival_by_frame": coverage,
        "motion_magnitude": sum(value for value in motion if value is not None) / max(sum(value is not None for value in motion), 1),
        "motion_magnitude_by_frame": motion,
        "net_displacement": sum(displacement) / len(displacement) if displacement else None,
        "primary_track_count": primary_tracks,
        "median_track_length_frames": float(torch.tensor(lengths, dtype=torch.float32).median()) if lengths else None,
        "scorer_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "punc_invariants": {
            "frame0_correction_max_px": max(frame0_corrections) if frame0_corrections else 0.0,
            "visibility_changed_count": visibility_changed,
        },
        "provider_diagnostics": state.diagnostics,
    }, quality


def make_renoise_delta(
    base_final_latent: torch.Tensor,
    prefix_latent: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """由 Base final latent 重新前向加噪，返回 callback 边界应注入的 delta。"""
    if tuple(base_final_latent.shape) != tuple(prefix_latent.shape):
        raise PairPilotError("Base final latent 与 fork prefix latent shape 不一致")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise PairPilotError("re-noise sigma 必须是有限正数")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    epsilon = torch.randn(tuple(base_final_latent.shape), generator=generator, dtype=torch.float32)
    epsilon = epsilon - epsilon.mean()
    epsilon = epsilon / max(_rms(epsilon), 1.0e-12)
    renoised = base_final_latent.detach().cpu().float() + float(sigma) * epsilon
    delta = renoised - prefix_latent.detach().cpu().float()
    return delta.to(dtype=prefix_latent.dtype), {
        "epsilon_rms": _rms(epsilon),
        "epsilon_mean": float(epsilon.mean()),
        "injected_delta_rms": _rms(delta),
        "injected_delta_mean": float(delta.mean()),
    }


def _validate_pa1(cfg: Any) -> dict[str, Any]:
    run_dir = Path(str(cfg.pair.pa1_branch_run))
    summary = _read_json(run_dir / "summary.json", label="PA1 branch summary")
    manifest = _read_json(run_dir / "manifest.json", label="PA1 branch manifest")
    if summary.get("status") != "done" or summary.get("task_id") != "PA1-BRANCH-02":
        raise PairPilotError("PA2 需要已完成的 PA1-BRANCH-02")
    machine = _mapping(summary.get("machine"), label="PA1 machine")
    human = _mapping(summary.get("human_review"), label="PA1 human review")
    if not bool(machine.get("machine_pass")) or not bool(human.get("pass")):
        raise PairPilotError("PA1 machine/human gate 未同时通过")
    if summary.get("next_gate") != "PA2-PAIR-03" or manifest.get("status") != "done":
        raise PairPilotError("PA1 provenance 未解锁 PA2")
    return {
        "run_path": str(run_dir),
        "summary_sha256": file_fingerprint(str(run_dir / "summary.json")),
        "config_fingerprint": summary.get("config_fingerprint"),
        "profile_fingerprint": summary.get("profile_fingerprint"),
        "human_review": human,
    }


def _validate_pair_config(cfg: Any) -> None:
    pair = cfg.pair
    mode = str(pair.mode)
    if mode not in {"formal", "extension", "smoke"}:
        raise PairPilotError(f"未知 PA2 mode: {mode}")
    if str(cfg.model.generation.protocol) != "svd_official_v1" or bool(cfg.model.lora.enable):
        raise PairPilotError("PA2 只允许冻结 Base svd_official_v1")
    if int(pair.num_inference_steps) != 25:
        raise PairPilotError("PA2 只允许 25 steps")
    expected = {
        "formal": {"smoke": False, "offset": 0, "count": 64, "minimum": 48, "reviews": 48},
        "extension": {"smoke": False, "offset": 64, "count": 56, "minimum": 0, "reviews": 0},
        "smoke": {"smoke": True, "offset": 0, "count": 1, "minimum": 0, "reviews": 1},
    }[mode]
    observed = {
        "smoke": bool(pair.smoke), "offset": int(pair.condition_offset),
        "count": int(pair.condition_count), "minimum": int(pair.minimum_valid_pairs),
        "reviews": int(pair.review.required_cases),
    }
    if observed != expected:
        raise PairPilotError(f"PA2 {mode} 规模未按预注册值冻结: observed={observed}, expected={expected}")
    if int(cfg.data.num_frames) != 14 or int(cfg.model.num_frames) != 14:
        raise PairPilotError("PA2 必须使用 PA1 冻结的 14 frames")
    if str(pair.condition_partition) != "preference_train":
        raise PairPilotError("PA2 只能从 preference_train 生成候选")
    if not math.isclose(float(pair.fork_fraction), 0.6, rel_tol=0.0, abs_tol=1.0e-12):
        raise PairPilotError("PA2 common-prefix 必须复用 PA1 冻结 fork=0.6")
    if not math.isclose(float(pair.strength_rho), 0.04, rel_tol=0.0, abs_tol=1.0e-12):
        raise PairPilotError("PA2 common-prefix 必须复用 PA1 冻结 large rho=0.04")
    if int(pair.renoise.candidate_count) != 2:
        raise PairPilotError("P2 Base re-noise 对照必须恰好生成两个候选")
    if int(pair.review.required_cases) != int(pair.review.p1_cases) + int(pair.review.p0_cases) + int(pair.review.p2_cases):
        raise PairPilotError("PA2 review strata 数必须恰好等于 required_cases")
    if mode == "formal" and (int(pair.review.required_cases) != 48 or int(pair.minimum_valid_pairs) != 48):
        raise PairPilotError("PA2 formal 必须使用 48 valid pairs 与 48-case review")


def _select_pair_conditions(split: Mapping[str, Any], pair: Any) -> list[dict[str, Any]]:
    offset = int(pair.condition_offset)
    count = int(pair.condition_count)
    selected = select_profile_conditions(
        split, partition=str(pair.condition_partition), condition_count=offset + count,
        required_start_index=int(pair.required_start_index),
    )
    result = [dict(row) for row in selected[offset:offset + count]]
    if len(result) != count:
        raise PairPilotError("PA2 condition offset/count 无法从冻结 split 满足")
    return result


def preflight_physics_dpo_pair(cfg: Any) -> dict[str, Any]:
    """只读 PA2 前置检查，不加载 SVD/RAFT，不创建 run。"""
    result: dict[str, Any] = {
        "task_id": str(cfg.pair.task_id), "status": "ready", "uses_gpu": False,
        "uses_future_gt": False, "blockers": [],
    }
    try:
        _validate_pair_config(cfg)
        result["pa1"] = _validate_pa1(cfg)
        result["horizon"] = _load_horizon_provenance(cfg.pair)
        split, split_provenance = _load_scene_split(cfg.pair)
        result["scene_split"] = split_provenance
        result["pa0"] = _validate_pa0(cfg.pair)
        result["selected_conditions"] = _select_pair_conditions(split, cfg.pair)
        if str(cfg.pair.mode) == "formal":
            evaluator = CoTracker3IndependentEvaluator(dict(cfg.branch.evaluator))
            result["independent_evaluator"] = evaluator.preflight()
            if not bool(result["independent_evaluator"].get("available")):
                raise PairPilotError("PA2 formal review 的 CoTracker3 不可用")
            if not Path("docs/PA2_PAIR_HUMAN_REVIEW_PROMPT.md").is_file():
                raise PairPilotError("PA2 formal 缺少完整人工评测提示词")
    except Exception as exc:
        result["status"] = "blocked"
        result["blockers"].append({"kind": "provenance", "error": repr(exc)})
        return result
    try:
        dataset = _dataset_for_horizon(cfg.data, num_frames=14)
        records = {str(row["sample_id"]): dict(row) for row in dataset.clip_records}
        checks = []
        for selected in result["selected_conditions"]:
            clip_id, token = str(selected["clip_id"]), str(selected["sample_tokens"][0])
            if clip_id not in records or str(records[clip_id]["sample_tokens"][0]) != token:
                raise PairPilotError(f"PA2 condition 不匹配冻结 dataset: {clip_id}")
            frame = _load_condition_frame(dataset, token)
            checks.append({"clip_id": clip_id, "condition_frame_sha256": _tensor_fingerprint(frame)})
        result["data"] = {"ready": True, "condition_checks": checks}
    except Exception as exc:
        result["status"] = "blocked"
        result["blockers"].append({"kind": "dataset", "error": repr(exc)})
    return result


def _audit_candidate_record(
    *,
    candidate_id: str,
    constructor: str,
    condition: Mapping[str, Any],
    artifact_dir: Path,
    work_dir: Path,
    generation_seed: int,
    trace: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "audit_candidate_id": candidate_id,
        "constructor": constructor,
        "condition_id": condition["condition_id"],
        "scene_id": condition["scene_id"],
        "split": condition["split"],
        "rgb_video_path": str((artifact_dir / "video.mp4").relative_to(work_dir)),
        "vae_latent_path": str((artifact_dir / "vae_latents.pt").relative_to(work_dir)),
        "diagnostics_path": str((artifact_dir / "trace.json").relative_to(work_dir)),
        "score_path": str((artifact_dir / "score.json").relative_to(work_dir)),
        "generation_seed": int(generation_seed),
        "trace": dict(trace),
        "uses_future_gt": False,
        "metadata": dict(metadata),
    }


def _generate_renoise_candidates(
    *,
    backbone: SVDBackbone,
    condition: Mapping[str, Any],
    condition_frame: torch.Tensor,
    base_record: Mapping[str, Any],
    generation_seed: int,
    work_dir: Path,
    cfg: Any,
    fps: int,
    height: int,
    width: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """P2 对照：从 exact Base final latent 前向加噪后走同一 official suffix。"""
    common = {
        "seed": int(generation_seed), "num_frames": int(condition["num_frames"]),
        "num_inference_steps": int(cfg.pair.num_inference_steps), "height": int(height), "width": int(width),
    }
    base_trace = trace_backbone_generation(backbone, condition_frame.to(backbone.device), **common)
    observed_hash = fingerprint_denoising_trace(base_trace)
    trace_payload = _read_json(work_dir / str(base_record["diagnostics_path"]), label="P1 Base trace")
    if observed_hash != trace_payload.get("official_trace"):
        raise PairPilotError("P2 re-noise 前的 Base rerun 未 exact 匹配 P1 Base guard")
    fork_step = resolve_fork_step(int(cfg.pair.num_inference_steps), float(cfg.pair.fork_fraction))
    prefix = base_trace["post_step_latents"][fork_step - 1]
    pipe = backbone._generation_pipeline()
    pipe.scheduler.set_timesteps(int(cfg.pair.num_inference_steps), device=torch.device(backbone.device))
    sigma = float(pipe.scheduler.sigmas[fork_step].detach().float().cpu()) * float(cfg.pair.renoise.sigma_multiplier)
    root = work_dir / "constructor_baselines" / str(condition["condition_id"])
    records: list[dict[str, Any]] = []
    frames_by_id: dict[str, torch.Tensor] = {}
    vae_by_id: dict[str, torch.Tensor] = {}
    for index in range(int(cfg.pair.renoise.candidate_count)):
        seed = int(cfg.pair.renoise.noise_seed_start) + int(generation_seed) * 10 + index
        delta, noise_info = make_renoise_delta(base_trace["final_latent"], prefix, sigma=sigma, seed=seed)
        _reset_peak_memory(str(backbone.device))
        started = time.perf_counter()
        trace, injection = trace_backbone_generation_with_fork_perturbation(
            backbone, condition_frame.to(backbone.device), **common, fork_step=fork_step, perturbation=delta,
        )
        _sync_cuda(str(backbone.device))
        shared_prefix = verify_shared_prefix_before_callback_injection(base_trace, trace, injection, fork_step=fork_step)
        if not bool(torch.isfinite(trace["decoded_frames"]).all()):
            raise PairPilotError("P2 Base re-noise decoded frames 包含 NaN/Inf")
        candidate_id = f"renoise-{condition['condition_id']}-r{index}"
        artifact_dir = root / f"renoise-{index}"
        hashes = fingerprint_denoising_trace(trace)
        payload = {
            "constructor": "P2-base-renoise", "candidate_role": "audit_only", "generation_seed": int(generation_seed),
            "fork_step": fork_step, "sigma": sigma, "base_trace": observed_hash,
            "shared_prefix_verification": shared_prefix, "renoise": noise_info, "full_trace": hashes,
        }
        video, vae, storage = _write_candidate_artifact(
            backbone=backbone, artifact_dir=artifact_dir, frames=trace["decoded_frames"], trace=trace,
            trace_payload=payload, fps=int(fps),
        )
        record = _audit_candidate_record(
            candidate_id=candidate_id, constructor="P2-base-renoise", condition=condition, artifact_dir=artifact_dir,
            work_dir=work_dir, generation_seed=generation_seed, trace=hashes,
            metadata={
                "fork_step": fork_step, "sigma": sigma, "shared_prefix": shared_prefix,
                "renoise": noise_info, "generation_seconds": time.perf_counter() - started,
                "generation_peak_vram_bytes": _peak_memory_bytes(str(backbone.device)), "storage_bytes": int(storage),
                "video_path": str(video.relative_to(work_dir)),
            },
        )
        records.append(record)
        frames_by_id[candidate_id] = trace["decoded_frames"].detach().cpu()
        vae_by_id[candidate_id] = vae
    return records, frames_by_id, vae_by_id


def _p0_independent_record(
    *,
    condition: Mapping[str, Any],
    work_dir: Path,
    generation_detail: Mapping[str, Any],
) -> dict[str, Any]:
    independent = _mapping(generation_detail["independent_diagnostic"], label="P0 independent diagnostic")
    candidate_id = f"independent-{condition['condition_id']}"
    video_path = Path(str(independent["rgb_video_path"]))
    artifact_dir = work_dir / video_path.parent
    return _audit_candidate_record(
        candidate_id=candidate_id,
        constructor="P0-independent",
        condition=condition,
        artifact_dir=artifact_dir,
        work_dir=work_dir,
        generation_seed=int(generation_detail["independent_seed"]),
        trace=_mapping(independent["trace"], label="P0 independent trace"),
        metadata={"diagnostic_promoted_to_audit_only_baseline": True, "source": "P1 independent distance calibration"},
    )


def _pair_record(
    *,
    pair_id: str,
    constructor: str,
    condition: Mapping[str, Any],
    candidate_a: str,
    candidate_b: str,
    score_a: Mapping[str, Any],
    score_b: Mapping[str, Any],
    feasibility_a: Mapping[str, Any],
    feasibility_b: Mapping[str, Any],
    quality_a: Mapping[str, Any],
    quality_b: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    frame_alignment_pass: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decision = decide_global_pair(
        candidate_a=candidate_a, candidate_b=candidate_b, score_a=score_a, score_b=score_b,
        feasibility_a=feasibility_a, feasibility_b=feasibility_b, quality_a=quality_a,
        quality_b=quality_b, thresholds=thresholds,
    )
    segments = decide_segments(
        pair_id=pair_id, candidate_a=candidate_a, candidate_b=candidate_b, score_a=score_a,
        score_b=score_b, thresholds=thresholds, frame_alignment_pass=frame_alignment_pass,
    )
    return {
        "pair_id": pair_id, "constructor": constructor, "condition_id": condition["condition_id"],
        "split": condition["split"], "frame_alignment_pass": bool(frame_alignment_pass), **decision,
    }, segments


def _core_preference(pair: Mapping[str, Any], *, scorer_fingerprint: str) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"], "condition_id": pair["condition_id"], "candidate_a": pair["candidate_a"],
        "candidate_b": pair["candidate_b"], "split": pair["split"], "global_label": pair["global_label"],
        "winner_candidate_id": pair["winner_candidate_id"], "loser_candidate_id": pair["loser_candidate_id"],
        "feasibility_a": pair["feasibility_a"], "feasibility_b": pair["feasibility_b"],
        "physics_components": pair["physics_components"], "quality_components": pair["quality_components"],
        "preference_margin": pair["preference_margin"], "pair_confidence": pair["pair_confidence"],
        "scorer_fingerprint": scorer_fingerprint, "human_review_id": None,
        "abstain_reason": pair["abstain_reason"], "uses_future_gt": False,
    }


def _write_constructor_pair(path: Path, row: Mapping[str, Any]) -> None:
    _json_line(path, row)


def _constructor_coverage(
    constructor_summary: Mapping[str, Mapping[str, Any]],
    condition_count: int,
) -> dict[str, Any]:
    """核对每个 condition 的 P0:raw-P1:P2 pair 数严格为 1:2:1。"""
    expected = {
        "P0-independent": int(condition_count),
        "P1-common-prefix": 2 * int(condition_count),
        "P2-base-renoise": int(condition_count),
    }
    actual = {
        name: int(constructor_summary.get(name, {}).get("pair_count", 0))
        for name in CONSTRUCTORS
    }
    return {
        "expected_pair_counts": expected,
        "actual_pair_counts": actual,
        "pass": actual == expected,
    }


def _machine_status(*, mode: str, checks: Mapping[str, Any]) -> str:
    if not all(bool(value) for value in checks.values()):
        return "blocked"
    return "awaiting_reviews" if mode == "formal" else "done"


def _next_gate(*, mode: str, status: str) -> str:
    if mode == "formal" and status == "awaiting_reviews":
        return "PA2 human review"
    if mode == "smoke" and status == "done":
        return "PA2 formal"
    if mode == "extension" and status == "done":
        return "PA2 expanded merge"
    return "PA2-PAIR-03"


def _blind_segment_label(
    segment_label: str,
    *,
    pair: Mapping[str, Any],
    blind_mapping: Mapping[str, str],
) -> str:
    if segment_label not in {"a_wins", "b_wins"}:
        return segment_label
    winner = pair["candidate_a"] if segment_label == "a_wins" else pair["candidate_b"]
    side = next((label for label, candidate_id in blind_mapping.items() if candidate_id == winner), None)
    if side is None:
        raise PairPilotError("local segment winner 不在 blind mapping")
    return f"{side.lower()}_better"


def _overlay_tensor(frames: torch.Tensor, state: Any) -> torch.Tensor:
    overlay = _draw_overlay(frames, state)
    return torch.from_numpy(overlay).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def _write_pair_review_materials(
    *,
    work_dir: Path,
    cfg: Any,
    frames_by_candidate: Mapping[str, torch.Tensor],
    core_preferences: Sequence[Mapping[str, Any]],
    core_segments: Sequence[Mapping[str, Any]],
    constructor_pairs: Sequence[Mapping[str, Any]],
    constructor_segments: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    fps: int,
) -> dict[str, Any]:
    """生成 48-case Stage A blind video 与 Stage B 独立 CoTracker adjudication。"""
    review_cfg = cfg.pair.review
    by_pair = {str(row["pair_id"]): dict(row) for row in constructor_pairs}
    p1_pool = [
        by_pair[str(row["pair_id"])] for row in core_preferences
        if str(row.get("global_label")) in DECISIVE_LABELS and str(row["pair_id"]) in by_pair
    ]
    p0_pool = [dict(row) for row in constructor_pairs if str(row.get("constructor")) == "P0-independent"]
    p2_pool = [dict(row) for row in constructor_pairs if str(row.get("constructor")) == "P2-base-renoise"]
    selected = [
        *select_review_pairs(p1_pool, required=int(review_cfg.p1_cases), seed=int(review_cfg.seed)),
        *select_review_pairs(p0_pool, required=int(review_cfg.p0_cases), seed=int(review_cfg.seed) + 1000),
        *select_review_pairs(p2_pool, required=int(review_cfg.p2_cases), seed=int(review_cfg.seed) + 2000),
    ]
    random.Random(int(review_cfg.seed) + 3000).shuffle(selected)
    if len(selected) != int(review_cfg.required_cases):
        raise PairPilotError("PA2 review 抽样数量不等于 required_cases")

    stage_a_dir = work_dir / "review" / "stage_a_blind"
    stage_b_overlay_dir = work_dir / "review" / "stage_b_overlay"
    stage_b_diagnostic_dir = work_dir / "review" / "stage_b_diagnostics"
    stage_a_dir.mkdir(parents=True, exist_ok=False)
    stage_b_overlay_dir.mkdir(parents=True, exist_ok=False)
    stage_b_diagnostic_dir.mkdir(parents=True, exist_ok=False)
    evaluator = CoTracker3IndependentEvaluator(dict(cfg.branch.evaluator))
    evaluator_preflight = evaluator.preflight()
    if not bool(evaluator_preflight.get("available")):
        raise PairPilotError(f"PA2 review 需要已通过 E0 的 CoTracker3: {evaluator_preflight}")
    independent_cache: dict[str, tuple[dict[str, Any], Any]] = {}
    segment_rows = [*core_segments, *constructor_segments]
    segments_by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for row in segment_rows:
        segments_by_pair.setdefault(str(row["pair_id"]), []).append(row)
    public_cases: list[dict[str, Any]] = []
    private_cases: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(selected):
        pair_id = str(pair["pair_id"])
        candidates = [str(pair["candidate_a"]), str(pair["candidate_b"])]
        random.Random(int(review_cfg.seed) + 4000 + index).shuffle(candidates)
        blind_mapping = {"A": candidates[0], "B": candidates[1]}
        case_id = f"pa2-pair-review-{index:03d}"
        stage_a = _make_panel([frames_by_candidate[candidates[0]], frames_by_candidate[candidates[1]]], ["A", "B"])
        stage_a_path = stage_a_dir / f"{case_id}.mp4"
        write_video(stage_a, str(stage_a_path), fps=int(fps))
        stage_a_artifact = _existing_video_path(stage_a_path)

        overlay_frames: list[torch.Tensor] = []
        independent_public: dict[str, Any] = {}
        for side, candidate_id in blind_mapping.items():
            if candidate_id not in independent_cache:
                independent_cache[candidate_id] = _score_with_state(
                    evaluator, frames_by_candidate[candidate_id], device=str(cfg.branch.evaluator.device),
                )
            independent_score, independent_state = independent_cache[candidate_id]
            overlay_frames.append(_overlay_tensor(frames_by_candidate[candidate_id], independent_state))
            independent_public[side] = independent_score
        stage_b = _make_panel(overlay_frames, ["A", "B"])
        stage_b_path = stage_b_overlay_dir / f"{case_id}.mp4"
        write_video(stage_b, str(stage_b_path), fps=int(fps))
        stage_b_artifact = _existing_video_path(stage_b_path)

        local = []
        seen_segments: set[tuple[int, int]] = set()
        for segment in sorted(segments_by_pair.get(pair_id, []), key=lambda row: (int(row["start_frame"]), int(row["end_frame"]))):
            window = (int(segment["start_frame"]), int(segment["end_frame"]))
            if window in seen_segments:
                continue
            seen_segments.add(window)
            local.append({
                "start_frame": window[0], "end_frame": window[1],
                "label": _blind_segment_label(str(segment["label"]), pair=pair, blind_mapping=blind_mapping),
                "confidence": segment.get("confidence"), "frame_alignment_pass": segment.get("frame_alignment_pass"),
            })
        diagnostic = {
            "case_id": case_id,
            "evidence_order": ["A", "B"],
            "independent_cotracker3": independent_public,
            "punc_pair_scorer": {side: dict(scores[candidate_id]) for side, candidate_id in blind_mapping.items()},
            "local_segment_labels": local,
            "interpretation": {
                "punc_projection_energy": "越低表示该视频轨迹需要的置信度归一化 P-UNC 修正越小；不是人工 verdict。",
                "cotracker3": "独立 evaluator，仅显示 track/coverage/dynamics，不产生 pair winner。",
            },
        }
        diagnostic_path = stage_b_diagnostic_dir / f"{case_id}.json"
        atomic_write_json(str(diagnostic_path), diagnostic)
        public_cases.append({
            "case_id": case_id,
            "stage_a_video": str(stage_a_artifact.relative_to(work_dir)),
            "stage_b_overlay_video": str(stage_b_artifact.relative_to(work_dir)),
            "stage_b_diagnostics": str(diagnostic_path.relative_to(work_dir)),
            "blind_sides": ["A", "B"],
        })
        private_cases.append({
            "case_id": case_id, "pair_id": pair_id, "condition_id": pair["condition_id"],
            "constructor": pair["constructor"], "blind_mapping": blind_mapping,
            "machine_global_label": pair["global_label"], "machine_winner_id": pair.get("winner_candidate_id"),
            "machine_loser_id": pair.get("loser_candidate_id"),
            "review_margin_bucket": pair["review_margin_bucket"],
            "review_machine_stratum": pair["review_machine_stratum"],
        })
        template_rows.append({
            "case_id": case_id,
            "stage_a_verdict": "pending",
            "stage_a_motion_plausibility": {"a": "pending", "b": "pending"},
            "stage_a_visual_quality": {"a": "pending", "b": "pending"},
            "stage_a_motion_amount": "pending",
            "stage_a_identity_consistency": {"a": "pending", "b": "pending"},
            "stage_a_failure_reasons": {"a": [], "b": []},
            "stage_b_verdict": "pending",
            "stage_b_change_reason": "",
            "low_motion_collapse": {"a": None, "b": None},
            "catastrophic_quality_failure": {"a": None, "b": None},
            "reviewer": "pending",
            "notes": "",
        })
    atomic_write_json(str(work_dir / "review_cases.json"), public_cases)
    atomic_write_json(str(work_dir / "review_cases.private.json"), private_cases)
    atomic_write_text(
        str(work_dir / "reviews.template.jsonl"),
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in template_rows),
    )
    prompt_path = Path("docs/PA2_PAIR_HUMAN_REVIEW_PROMPT.md")
    if not prompt_path.is_file():
        raise PairPilotError("缺少 PA2 完整人工评测提示词文档")
    prompt = prompt_path.read_text(encoding="utf-8")
    atomic_write_text(str(work_dir / "REVIEW_PROMPT.md"), prompt)
    atomic_write_text(str(work_dir / "REVIEW_README.md"), prompt)
    return {
        "required_cases": len(public_cases),
        "p1_cases": sum(row["constructor"] == "P1-common-prefix" for row in private_cases),
        "p0_cases": sum(row["constructor"] == "P0-independent" for row in private_cases),
        "p2_cases": sum(row["constructor"] == "P2-base-renoise" for row in private_cases),
        "independent_evaluator_preflight": evaluator_preflight,
        "review_protocol": "pa2-two-stage-blind-v1",
    }


def _clean_terminal_markers(run_dir: Path) -> None:
    for name in ("COMPLETE", "FAILED", "awaiting_reviews", "MACHINE_COMPLETE"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def aggregate_physics_dpo_pair_reviews(cfg: Any) -> dict[str, Any]:
    """只聚合人工 JSONL；不加载 SVD、RAFT 或 CoTracker，不改写人工 verdict。"""
    run_dir = Path(str(cfg.work_dir))
    machine = _read_json(run_dir / "machine_summary.json", label="PA2 machine summary")
    machine_checks = _mapping(machine.get("machine"), label="PA2 machine checks")
    if str(machine.get("status")) != "awaiting_reviews" or not bool(machine_checks.get("machine_pass")):
        raise PairPilotError("仅 machine-pass 的 PA2 awaiting_reviews run 可聚合")
    try:
        human = review_summary(run_dir, dict(cfg.pair.review))
    except PreferenceReviewError as exc:
        raise PairPilotError(str(exc)) from exc
    status = "done" if bool(human["pass"]) else str(human["status"])
    summary = {
        **machine,
        "status": status,
        "human_review": human,
        "next_gate": "PA3-KERNEL-04" if status == "done" else "PA2 review expansion" if status == "needs_more_reviews" else "PA2-PAIR-03",
    }
    atomic_write_json(str(run_dir / "summary.json"), summary)
    _clean_terminal_markers(run_dir)
    if status in {"done", "rejected"}:
        atomic_write_text(str(run_dir / "COMPLETE"), sha256_json(summary) + "\n")
    else:
        atomic_write_text(str(run_dir / "MACHINE_COMPLETE"), sha256_json(machine) + "\n")
        atomic_write_text(str(run_dir / "awaiting_reviews"), sha256_json(summary) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path, label="PA2 manifest")
    manifest.update({"status": status, "ended_at": utc_now(), "exit_reason": "human_review"})
    atomic_write_json(str(manifest_path), manifest)
    return summary


def run_physics_dpo_pair(cfg: Any) -> dict[str, Any]:
    """运行 PA2 机器 candidate/pair legality；human aggregate 由后续命令单独完成。"""
    _validate_pair_config(cfg)
    git = git_state(".")
    if git.get("dirty"):
        raise PairPilotError("正式 PA2 拒绝在 dirty worktree 上运行")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"PA2 run directory 已存在: {work_dir}")
    pa1 = _validate_pa1(cfg)
    horizon = _load_horizon_provenance(cfg.pair)
    split_manifest, split_provenance = _load_scene_split(cfg.pair)
    pa0 = _validate_pa0(cfg.pair)
    selected = _select_pair_conditions(split_manifest, cfg.pair)
    cfg_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=cfg_fp,
        cache_fingerprint="not-applicable:pa2-structure-aligned-pairs", seed=int(cfg.seed), git=git,
        environment=environment_fingerprint(), data_split=str(cfg.pair.condition_partition),
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.pair.task_id), "status": "running", "uses_future_gt": False, "training": False,
        "pa1_branch": pa1, "horizon": horizon, "scene_split": split_provenance, "pa0_review": pa0,
        "condition_selection_rule": {"partition": str(cfg.pair.condition_partition), "condition_count": int(cfg.pair.condition_count),
                                     "condition_offset": int(cfg.pair.condition_offset),
                                     "required_start_index": int(cfg.pair.required_start_index),
                                     "ordering": "one start-index-matched clip per scene; ascending (scene_token, clip_id)"},
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    try:
        dataset = _dataset_for_horizon(cfg.data, num_frames=14)
        by_clip = {str(row["sample_id"]): dict(row) for row in dataset.clip_records}
        selected_with_frame = []
        for row in selected:
            clip_id, first_token = str(row["clip_id"]), str(row["sample_tokens"][0])
            if clip_id not in by_clip or str(by_clip[clip_id]["sample_tokens"][0]) != first_token:
                raise PairPilotError(f"PA2 selected condition 与 dataset 不一致: {clip_id}")
            frame = _load_condition_frame(dataset, first_token)
            selected_with_frame.append({**row, "condition_frame": frame, "condition_frame_sha256": _tensor_fingerprint(frame)})

        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        backbone = build_backbone(cfg.model, load=True, device=str(cfg.device))
        if not isinstance(backbone, SVDBackbone):
            raise PairPilotError("PA2 当前只支持 SVDBackbone")
        backbone.unet.eval()
        backbone.vae.eval()
        backbone.image_encoder.eval()
        metadata = backbone.generation_protocol_metadata()
        base_fp = _base_model_fingerprint(str(cfg.model.pretrained))
        manifest_data.update({
            "base_model_fingerprint": base_fp, "generation_protocol": metadata,
            "selected_source_conditions": [{key: value for key, value in row.items() if key != "condition_frame"} for row in selected_with_frame],
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)

        conditions: list[dict[str, Any]] = []
        p1_candidates: list[dict[str, Any]] = []
        audit_candidates: list[dict[str, Any]] = []
        frames_by_candidate: dict[str, torch.Tensor] = {}
        for index, row in enumerate(selected_with_frame):
            condition = _make_condition_record(
                selected=row, split=str(cfg.pair.condition_partition), camera=str(cfg.data.cameras[0]),
                num_frames=14, fps=int(metadata["fps_input"]), condition_frame_hash=str(row["condition_frame_sha256"]),
                scheduler_fingerprint=str(metadata["scheduler_config_fingerprint"]), base_model_fingerprint=base_fp,
                git_commit=str(git["commit"]), config_fingerprint_value=cfg_fp,
            )
            rows, candidate_frames, _, detail = _generate_condition_group(
                backbone=backbone, condition=condition, condition_frame=row["condition_frame"], branch=cfg.pair,
                work_dir=work_dir, generation_seed=int(cfg.pair.generation_seed_start) + index,
                direction_seed=int(cfg.pair.direction_seed_start) + index, fps=int(metadata["fps_input"]),
                height=int(cfg.data.height), width=int(cfg.data.width), metrics=metrics,
            )
            base = next(item for item in rows if item["candidate_role"] == "base_guard")
            p0_record = _p0_independent_record(condition=condition, work_dir=work_dir, generation_detail=detail)
            frames_by_candidate[p0_record["audit_candidate_id"]] = detail.pop("independent_frames").detach().cpu()
            detail.pop("independent_vae")
            renoise_rows, renoise_frames, _ = _generate_renoise_candidates(
                backbone=backbone, condition=condition, condition_frame=row["condition_frame"], base_record=base,
                generation_seed=int(cfg.pair.generation_seed_start) + index, work_dir=work_dir, cfg=cfg,
                fps=int(metadata["fps_input"]), height=int(cfg.data.height), width=int(cfg.data.width),
            )
            conditions.append(condition)
            p1_candidates.extend(rows)
            audit_candidates.extend([p0_record, *renoise_rows])
            frames_by_candidate.update(candidate_frames)
            frames_by_candidate.update(renoise_frames)
            _json_line(work_dir / "conditions.jsonl", condition)
            for candidate in rows:
                _json_line(work_dir / "candidate_manifest.jsonl", candidate)
                _json_line(work_dir / "candidates.jsonl", candidate)
            for candidate in [p0_record, *renoise_rows]:
                _json_line(work_dir / "constructor_candidates.jsonl", candidate)
            metrics.append(index, {
                "event": "condition_generated", "condition_id": condition["condition_id"],
                "base_guard_exact": True, "common_prefix_callback_verified": True,
                "p2_renoise_count": len(renoise_rows),
            })
        indexed_conditions = validate_conditions(conditions, split_manifest)
        indexed_candidates = validate_candidates(p1_candidates, indexed_conditions, exact_sibling_count=4)

        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _sync_cuda(str(cfg.device))
        provider = RAFTChainGeneratedTrackProvider(device=str(cfg.device), **dict(cfg.pair.generated_tracks))
        scores: dict[str, dict[str, Any]] = {}
        qualities: dict[str, dict[str, Any]] = {}
        all_records: dict[str, Mapping[str, Any]] = {str(row["candidate_id"]): row for row in p1_candidates}
        all_records.update({str(row["audit_candidate_id"]): row for row in audit_candidates})
        for index, (candidate_id, record) in enumerate(sorted(all_records.items())):
            score, quality = _punc_score(frames_by_candidate[candidate_id], provider, dict(cfg.pair.punc))
            scores[candidate_id], qualities[candidate_id] = score, quality
            score_path = work_dir / str(record["score_path"])
            atomic_write_json(str(score_path), {"punc_score": score, "quality": quality})
            destination = work_dir / ("candidate_diagnostics.jsonl" if candidate_id in indexed_candidates else "constructor_diagnostics.jsonl")
            _json_line(destination, {"candidate_id": candidate_id, "condition_id": record["condition_id"], "punc_score": score, "quality": quality})
            metrics.append(index, {"event": "candidate_scored", "candidate_id": candidate_id, "condition_id": record["condition_id"],
                                   "projection_energy": score["projection_energy"], "projection_points": score["projection_points"],
                                   "track_coverage": score["track_coverage"]})

        scorer_fingerprint = sha256_json({"protocol": "punc-pareto-pair-v1", "punc": dict(cfg.pair.punc), "thresholds": dict(cfg.pair.thresholds)})
        core_preferences: list[dict[str, Any]] = []
        core_segments: list[dict[str, Any]] = []
        constructor_pairs: list[dict[str, Any]] = []
        constructor_segments: list[dict[str, Any]] = []
        selections: list[dict[str, Any]] = []
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for candidate in p1_candidates:
            by_condition.setdefault(str(candidate["condition_id"]), []).append(candidate)
        audit_by_condition: dict[str, list[dict[str, Any]]] = {}
        for candidate in audit_candidates:
            audit_by_condition.setdefault(str(candidate["condition_id"]), []).append(candidate)
        for condition in conditions:
            condition_id = str(condition["condition_id"])
            p1_group = by_condition[condition_id]
            base = next(row for row in p1_group if row["candidate_role"] == "base_guard")
            base_id = str(base["candidate_id"])
            sibling_rows = [row for row in p1_group if row["candidate_role"] == "sibling"]
            feasibility = {
                str(row["candidate_id"]): candidate_feasibility(scores[str(row["candidate_id"])], scores[base_id], qualities[str(row["candidate_id"])], dict(cfg.pair.thresholds))
                for row in sibling_rows
            }
            grouped: dict[str, list[dict[str, Any]]] = {}
            for sibling in sibling_rows:
                grouped.setdefault(str(sibling["antithetic_group_id"]), []).append(sibling)
            p1_rows: list[dict[str, Any]] = []
            p1_segments: dict[str, list[dict[str, Any]]] = {}
            for group_id, siblings in sorted(grouped.items()):
                if len(siblings) != 2:
                    raise PairPilotError("PA2 antithetic group 必须恰好包含两条 sibling")
                a, b = sorted(siblings, key=lambda item: str(item["candidate_id"]))
                a_id, b_id = str(a["candidate_id"]), str(b["candidate_id"])
                frame_alignment = bool(
                    a["initial_latent_hash"] == b["initial_latent_hash"]
                    and a["prefix_trace_hash"] == b["prefix_trace_hash"]
                    and _first_frame_metrics(frames_by_candidate[a_id], frames_by_candidate[b_id])["rgb_rms"]
                    <= float(cfg.pair.thresholds.maximum_first_frame_rgb_rms)
                )
                pair_id = f"p1-{condition_id}-{group_id.rsplit('-', 1)[-1]}"
                pair, segments = _pair_record(
                    pair_id=pair_id, constructor="P1-common-prefix", condition=condition, candidate_a=a_id, candidate_b=b_id,
                    score_a=scores[a_id], score_b=scores[b_id], feasibility_a=feasibility[a_id], feasibility_b=feasibility[b_id],
                    quality_a=qualities[a_id], quality_b=qualities[b_id], thresholds=dict(cfg.pair.thresholds),
                    frame_alignment_pass=frame_alignment,
                )
                p1_rows.append(pair)
                p1_segments[pair_id] = segments
                constructor_pairs.append(pair)
                _write_constructor_pair(work_dir / "constructor_pairs.jsonl", pair)
                for segment in segments:
                    constructor_segment = {**segment, "constructor": "P1-common-prefix"}
                    constructor_segments.append(constructor_segment)
                    _json_line(work_dir / "constructor_segments.jsonl", constructor_segment)
            choice = select_condition_pair(p1_rows)
            selected_pair = next((row for row in p1_rows if row["pair_id"] == choice["selected_pair_id"]), None)
            if selected_pair is None:
                selected_pair = sorted(p1_rows, key=lambda row: str(row["pair_id"]))[0]
                choice = {**choice, "selected_pair_id": selected_pair["pair_id"], "selection_status": "abstain_canonical_pair"}
            selections.append({"condition_id": condition_id, **choice})
            _json_line(work_dir / "pair_selection.jsonl", selections[-1])
            core_preference = _core_preference(selected_pair, scorer_fingerprint=scorer_fingerprint)
            core_preferences.append(core_preference)
            core_segments.extend(p1_segments[str(selected_pair["pair_id"])])
            _json_line(work_dir / "preferences.jsonl", core_preference)
            for segment in p1_segments[str(selected_pair["pair_id"])]:
                _json_line(work_dir / "segments.jsonl", segment)

            p0 = next(row for row in audit_by_condition[condition_id] if row["constructor"] == "P0-independent")
            p0_id = str(p0["audit_candidate_id"])
            p0_pair, p0_segments = _pair_record(
                pair_id=f"p0-{condition_id}", constructor="P0-independent", condition=condition, candidate_a=base_id, candidate_b=p0_id,
                score_a=scores[base_id], score_b=scores[p0_id],
                feasibility_a=candidate_feasibility(scores[base_id], scores[base_id], qualities[base_id], dict(cfg.pair.thresholds)),
                feasibility_b=candidate_feasibility(scores[p0_id], scores[base_id], qualities[p0_id], dict(cfg.pair.thresholds)),
                quality_a=qualities[base_id], quality_b=qualities[p0_id], thresholds=dict(cfg.pair.thresholds),
                frame_alignment_pass=_first_frame_metrics(frames_by_candidate[base_id], frames_by_candidate[p0_id])["rgb_rms"] <= float(cfg.pair.thresholds.maximum_first_frame_rgb_rms),
            )
            constructor_pairs.append(p0_pair)
            _write_constructor_pair(work_dir / "constructor_pairs.jsonl", p0_pair)
            for segment in p0_segments:
                constructor_segment = {**segment, "constructor": "P0-independent"}
                constructor_segments.append(constructor_segment)
                _json_line(work_dir / "constructor_segments.jsonl", constructor_segment)
            renoise = sorted([row for row in audit_by_condition[condition_id] if row["constructor"] == "P2-base-renoise"], key=lambda row: str(row["audit_candidate_id"]))
            if len(renoise) != 2:
                raise PairPilotError("P2 必须有两个 Base re-noise audit candidate")
            r0, r1 = str(renoise[0]["audit_candidate_id"]), str(renoise[1]["audit_candidate_id"])
            p2_pair, p2_segments = _pair_record(
                pair_id=f"p2-{condition_id}", constructor="P2-base-renoise", condition=condition, candidate_a=r0, candidate_b=r1,
                score_a=scores[r0], score_b=scores[r1],
                feasibility_a=candidate_feasibility(scores[r0], scores[base_id], qualities[r0], dict(cfg.pair.thresholds)),
                feasibility_b=candidate_feasibility(scores[r1], scores[base_id], qualities[r1], dict(cfg.pair.thresholds)),
                quality_a=qualities[r0], quality_b=qualities[r1], thresholds=dict(cfg.pair.thresholds),
                frame_alignment_pass=_first_frame_metrics(frames_by_candidate[r0], frames_by_candidate[r1])["rgb_rms"] <= float(cfg.pair.thresholds.maximum_first_frame_rgb_rms),
            )
            constructor_pairs.append(p2_pair)
            _write_constructor_pair(work_dir / "constructor_pairs.jsonl", p2_pair)
            for segment in p2_segments:
                constructor_segment = {**segment, "constructor": "P2-base-renoise"}
                constructor_segments.append(constructor_segment)
                _json_line(work_dir / "constructor_segments.jsonl", constructor_segment)

        validate_preferences(core_preferences, indexed_conditions, indexed_candidates)
        validate_segments(core_segments, validate_preferences(core_preferences, indexed_conditions, indexed_candidates), indexed_candidates)
        decisive = [row for row in core_preferences if str(row["global_label"]) in DECISIVE_LABELS]
        decisive_segment_conditions = {
            str(row["condition_id"]) for row in decisive
            if any(segment["pair_id"] == row["pair_id"] and segment["label"] in DECISIVE_LABELS for segment in core_segments)
        }
        constructor_summary = {
            constructor: {
                "pair_count": sum(row["constructor"] == constructor for row in constructor_pairs),
                "decisive_count": sum(row["constructor"] == constructor and row["global_label"] in DECISIVE_LABELS for row in constructor_pairs),
                "abstain_count": sum(row["constructor"] == constructor and row["global_label"] == "abstain" for row in constructor_pairs),
            }
            for constructor in CONSTRUCTORS
        }
        constructor_coverage = _constructor_coverage(constructor_summary, len(conditions))
        checks = {
            "validated_core_schema": True,
            "minimum_valid_pairs": len(decisive) >= int(cfg.pair.minimum_valid_pairs),
            "minimum_non_tie_segment_conditions": len(decisive_segment_conditions) >= int(cfg.pair.minimum_valid_pairs),
            "three_constructor_comparison": bool(constructor_coverage["pass"]),
        }
        mode = str(cfg.pair.mode)
        status = _machine_status(mode=mode, checks=checks)
        review_materials = None
        make_review_materials = status == "awaiting_reviews" or bool(
            cfg.pair.smoke and cfg.pair.review.material_smoke
        )
        if make_review_materials:
            del provider
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _sync_cuda(str(cfg.device))
            review_materials = _write_pair_review_materials(
                work_dir=work_dir, cfg=cfg, frames_by_candidate=frames_by_candidate,
                core_preferences=core_preferences, core_segments=core_segments,
                constructor_pairs=constructor_pairs, constructor_segments=constructor_segments,
                scores=scores, fps=int(metadata["fps_input"]),
            )
        summary = {
            "status": status, "task_id": str(cfg.pair.task_id), "run_id": str(cfg.run_id), "config_fingerprint": cfg_fp,
            "scene_split_fingerprint": split_provenance["split_fingerprint"], "horizon_profile_fingerprint": horizon["profile_fingerprint"],
            "condition_count": len(conditions), "p1_candidate_count": len(p1_candidates), "audit_candidate_count": len(audit_candidates),
            "valid_global_pairs": len(decisive), "non_tie_segment_conditions": len(decisive_segment_conditions),
            "constructor_summary": constructor_summary, "constructor_coverage": constructor_coverage,
            "machine": {"machine_pass": all(checks.values()), "checks": checks},
            "review_materials": review_materials,
            "scorer_fingerprint": scorer_fingerprint, "next_gate": _next_gate(mode=mode, status=status),
            "uses_future_gt": False,
        }
        atomic_write_json(str(work_dir / "machine_summary.json"), summary)
        atomic_write_json(str(work_dir / "summary.json"), summary)
        if status == "awaiting_reviews":
            atomic_write_text(str(work_dir / "MACHINE_COMPLETE"), sha256_json(summary) + "\n")
            atomic_write_text(str(work_dir / "awaiting_reviews"), sha256_json(summary) + "\n")
        else:
            atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({"status": status, "ended_at": utc_now(), "exit_reason": "smoke_complete" if bool(cfg.pair.smoke) else "human_review_required" if status == "awaiting_reviews" else "machine_pair_gate"})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        failure = {"status": "failed", "task_id": str(cfg.pair.task_id), "run_id": str(cfg.run_id), "config_fingerprint": cfg_fp,
                   "error": repr(exc), "uses_future_gt": False}
        atomic_write_json(str(work_dir / "summary.json"), failure)
        atomic_write_text(str(work_dir / "FAILED"), sha256_json(failure) + "\n")
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PA2 structure-aligned preference pair legality")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.aggregate_only:
        parser.error("--preflight 与 --aggregate-only 不能同时使用")
    cfg = load_config(args.config)
    if args.preflight:
        result = preflight_physics_dpo_pair(cfg)
    elif args.aggregate_only:
        result = aggregate_physics_dpo_pair_reviews(cfg)
    else:
        result = run_physics_dpo_pair(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
