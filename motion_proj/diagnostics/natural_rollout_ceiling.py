"""RP-B0：冻结 SVD 自然独立 rollout 的 best-of-N support ceiling。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from ..auditor.generated_tracks import RAFTChainGeneratedTrackProvider
from ..backbones import build_backbone
from ..backbones.svd_backbone import SVDBackbone
from ..config import config_fingerprint, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..eval.independent_tracks import CoTracker3IndependentEvaluator
from ..eval.natural_rollout_ranking import (
    aggregate_b0_gate,
    candidate_eligibility,
    condition_diversity,
    cotracker_plausibility_score,
    generic_smoothness_energy,
    pair_key,
)
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from ..train.trainer import seed_everything
from ..utils.io import to_uint8_video, write_video
from .physics_dpo_pair import _punc_score
from .svd_conditioning_parity import _base_model_fingerprint
from .temporal_sampling_audit import (
    _condition_frame,
    _copy_data_config,
    _generate_case,
    _preflight_model_path,
    _score_case,
    _tensor_fingerprint,
    basic_video_metrics,
    select_scene_distinct_clip_records,
)


class NaturalRolloutCeilingError(RuntimeError):
    """B0 provenance、candidate budget、scorer 隔离或 gate 不合法。"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _condition_id(record: Mapping[str, Any]) -> str:
    payload = f"{record['scene_token']}:{record['sample_id']}"
    return f"b0-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _scene_split_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([
        (str(row["scene_token"]), str(row["sample_id"]), str(row["sample_tokens"][0]))
        for row in records
    ])


def _load_frames(row: Mapping[str, Any], work_dir: Path) -> torch.Tensor:
    path = work_dir / str(row["frames_path"])
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 3:
        raise NaturalRolloutCeilingError(f"candidate frames 非 [T,3,H,W]: {path}")
    return value


def _validate_parent_r1(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if not (path / "COMPLETE").is_file() or summary.get("status") != "done" or int(summary.get("selected_fps")) != 7:
        raise NaturalRolloutCeilingError("R1 parent 未 COMPLETE/done/fps7")
    return {
        "run": str(path),
        "selected_fps": 7,
        "result_fingerprint": summary.get("result_fingerprint"),
    }


def preflight_natural_rollout_ceiling(cfg: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": str(cfg.b0.task_id),
        "status": "ready",
        "uses_gpu": True,
        "uses_future_gt_for_generation": False,
        "uses_future_gt_for_generated_evaluation": False,
        "blockers": [],
    }
    try:
        parent = _validate_parent_r1(Path(str(cfg.b0.parent_r1_run)))
        result["parent_r1"] = parent
    except Exception as exc:
        result["blockers"].append({"kind": "r1_parent", "error": repr(exc)})
    try:
        dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data, split="val"))
        selected = select_scene_distinct_clip_records(
            dataset.clip_records, count=int(cfg.b0.condition_count),
        )
        result["data"] = {
            "dataset_clip_count": len(dataset),
            "condition_count": len(selected),
            "scene_split_fingerprint": _scene_split_fingerprint(selected),
            "sample_ids": [str(row["sample_id"]) for row in selected],
        }
    except Exception as exc:
        result["blockers"].append({"kind": "nuscenes", "error": repr(exc)})
    model = _preflight_model_path(str(cfg.model.pretrained))
    result["model"] = model
    if not bool(model["ready"]):
        result["blockers"].append({"kind": "svd", "missing": model["missing"]})
    evaluator = CoTracker3IndependentEvaluator(dict(cfg.b0.evaluator)).preflight()
    result["evaluator"] = evaluator
    if not bool(evaluator.get("available")):
        result["blockers"].append({"kind": "cotracker3", "reasons": evaluator.get("reasons", [])})
    raft_path = Path(str(cfg.b0.raft_checkpoint_path))
    raft_ready = raft_path.is_file()
    raft_sha = file_fingerprint(str(raft_path)) if raft_ready else None
    result["raft"] = {
        "ready": raft_ready and raft_sha == str(cfg.b0.raft_checkpoint_sha256),
        "path": str(raft_path),
        "sha256": raft_sha,
        "expected_sha256": str(cfg.b0.raft_checkpoint_sha256),
    }
    if not result["raft"]["ready"]:
        result["blockers"].append({"kind": "raft", **result["raft"]})
    usage = shutil.disk_usage(Path(str(cfg.work_dir)).parent)
    free_gb = float(usage.free) / float(1024**3)
    result["disk"] = {"free_gb": free_gb, "minimum_free_gb": float(cfg.b0.minimum_free_disk_gb)}
    if free_gb < float(cfg.b0.minimum_free_disk_gb):
        result["blockers"].append({"kind": "disk", "free_gb": free_gb})
    if result["blockers"]:
        result["status"] = "blocked"
    return result


def _validate_protocol(cfg: Any) -> None:
    b0 = cfg.b0
    if str(b0.task_id) != "RP-B0-05":
        raise NaturalRolloutCeilingError("task_id 必须为 RP-B0-05")
    if int(b0.condition_count) != 16:
        raise NaturalRolloutCeilingError("B0 必须使用 16 preference-dev conditions")
    if int(b0.initial_candidate_count) != 4 or int(b0.maximum_candidate_count) != 8:
        raise NaturalRolloutCeilingError("B0 candidate budget 必须为 4→8")
    if int(b0.base_candidate_index) != 0:
        raise NaturalRolloutCeilingError("B0 固定 Base 必须占 candidate index 0")
    if int(b0.maximum_candidate_count) * int(b0.condition_count) != 128:
        raise NaturalRolloutCeilingError("B0 总视频硬上限必须为 128")
    if str(cfg.model.generation.protocol) != "svd_official_v1" or bool(cfg.model.lora.enable):
        raise NaturalRolloutCeilingError("B0 只允许冻结 svd_official_v1 Base")
    if int(cfg.model.generation.fps) != 7 or int(b0.playback_fps) != 7:
        raise NaturalRolloutCeilingError("B0 generation/playback fps 必须冻结为 7")
    if int(b0.num_frames) != 8 or int(b0.num_inference_steps) != 25:
        raise NaturalRolloutCeilingError("B0 必须使用 8 frames / 25 steps")
    if int(cfg.data.num_frames) != 8 or int(cfg.model.num_frames) != 8:
        raise NaturalRolloutCeilingError("B0 data/model num_frames 必须为 8")
    if int(b0.review.punc_vs_random_cases) != 12 or int(b0.review.punc_vs_base_cases) != 12:
        raise NaturalRolloutCeilingError("B0 review 必须为 12 + 12 cases")


def _build_conditions(cfg: Any) -> tuple[NuScenesFutureVideoDataset, list[dict[str, Any]], dict[str, torch.Tensor]]:
    dataset = NuScenesFutureVideoDataset(_copy_data_config(cfg.data, split="val"))
    selected = select_scene_distinct_clip_records(dataset.clip_records, count=int(cfg.b0.condition_count))
    rows = []
    tensors = {}
    for record in selected:
        condition_id = _condition_id(record)
        frame = _condition_frame(dataset, str(record["sample_tokens"][0]))
        tensors[condition_id] = frame
        rows.append(
            {
                "condition_id": condition_id,
                "scene_name": str(record["scene_name"]),
                "scene_token": str(record["scene_token"]),
                "sample_id": str(record["sample_id"]),
                "first_sample_token": str(record["sample_tokens"][0]),
                "condition_frame_sha256": _tensor_fingerprint(frame),
                "uses_future_gt": False,
            }
        )
    return dataset, rows, tensors


def _generate_candidate_indices(
    cfg: Any,
    work_dir: Path,
    conditions: Sequence[Mapping[str, Any]],
    condition_tensors: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    metrics_log: JsonlMetrics,
    step_start: int,
) -> list[dict[str, Any]]:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.lora.enable = False
    backbone = build_backbone(model_cfg, load=True, device=str(cfg.device))
    if not isinstance(backbone, SVDBackbone):
        raise NaturalRolloutCeilingError("B0 当前只支持 SVDBackbone")
    backbone.unet.eval().requires_grad_(False)
    backbone.vae.eval().requires_grad_(False)
    backbone.image_encoder.eval().requires_grad_(False)
    rows = []
    try:
        for condition in conditions:
            condition_id = str(condition["condition_id"])
            condition_frame = condition_tensors[condition_id]
            for candidate_index in indices:
                generation_seed = int(cfg.b0.candidate_seed_start) + int(candidate_index)
                candidate_id = f"{condition_id}-c{int(candidate_index):02d}"
                artifact_dir = work_dir / "candidates" / condition_id / f"c{int(candidate_index):02d}"
                artifact_dir.mkdir(parents=True, exist_ok=False)
                frames, trace = _generate_case(
                    backbone,
                    condition_frame,
                    fps=int(cfg.model.generation.fps),
                    seed=generation_seed,
                    num_frames=int(cfg.b0.num_frames),
                    num_inference_steps=int(cfg.b0.num_inference_steps),
                    height=int(cfg.data.height),
                    width=int(cfg.data.width),
                )
                if not bool(torch.isfinite(frames).all()):
                    raise NaturalRolloutCeilingError(f"B0 generated RGB NaN/Inf: {candidate_id}")
                frames_path = artifact_dir / "frames.pt"
                torch.save(frames.detach().cpu(), frames_path)
                video_path = artifact_dir / "video.mp4"
                write_video(to_uint8_video(frames), str(video_path), fps=int(cfg.b0.playback_fps))
                if not video_path.is_file() and video_path.with_suffix(".npy").is_file():
                    video_path = video_path.with_suffix(".npy")
                if not video_path.is_file():
                    raise NaturalRolloutCeilingError(f"B0 video writer 未产生 artifact: {candidate_id}")
                row = {
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "scene_name": str(condition["scene_name"]),
                    "sample_id": str(condition["sample_id"]),
                    "candidate_index": int(candidate_index),
                    "candidate_role": "base_fixed" if int(candidate_index) == int(cfg.b0.base_candidate_index) else "natural_candidate",
                    "generation_seed": generation_seed,
                    "fps_input": int(cfg.model.generation.fps),
                    "fps_time_id": int(cfg.model.generation.fps) - 1,
                    "num_inference_steps": int(cfg.b0.num_inference_steps),
                    "frames_path": str(frames_path.relative_to(work_dir)),
                    "video_path": str(video_path.relative_to(work_dir)),
                    "frames_sha256": _tensor_fingerprint(frames),
                    "video_sha256": file_fingerprint(str(video_path)),
                    "uses_future_gt": False,
                    "pixel_metrics": basic_video_metrics(frames, condition_frame),
                    **trace,
                }
                rows.append(row)
                metrics_log.append(
                    step_start + len(rows),
                    {
                        "event": "generated",
                        "candidate_id": candidate_id,
                        "candidate_index": int(candidate_index),
                        "generation_seconds": trace["generation_seconds"],
                    },
                )
    finally:
        del backbone
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _validate_candidate_noise(rows: Sequence[Mapping[str, Any]]) -> None:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    for condition_id, group in by_condition.items():
        indices = [int(row["candidate_index"]) for row in group]
        if len(indices) != len(set(indices)):
            raise NaturalRolloutCeilingError(f"B0 candidate index 重复: {condition_id}")
        initial = [str(row["initial_video_latents_sha256"]) for row in group]
        if len(initial) != len(set(initial)):
            raise NaturalRolloutCeilingError(f"B0 independent seeds 未产生唯一 initial latents: {condition_id}")
        if any(bool(row.get("uses_future_gt")) for row in group):
            raise NaturalRolloutCeilingError("B0 candidate 泄漏 future GT")


def _score_training_side(
    cfg: Any,
    work_dir: Path,
    rows: Sequence[dict[str, Any]],
    metrics_log: JsonlMetrics,
    step_start: int,
) -> None:
    provider_settings = dict(cfg.b0.generated_tracks)
    provider = RAFTChainGeneratedTrackProvider(device=str(cfg.device), **provider_settings)
    try:
        for offset, row in enumerate(rows, start=1):
            frames = _load_frames(row, work_dir)
            score, quality = _punc_score(frames, provider, dict(cfg.b0.punc))
            generic = generic_smoothness_energy(score)
            row["training_score"] = score
            row["generic_smoothness_energy"] = generic
            row["saturation_fraction"] = quality.get("saturation_fraction")
            row["training_quality"] = quality
            metrics_log.append(
                step_start + offset,
                {
                    "event": "training_side_scored",
                    "candidate_id": row["candidate_id"],
                    "punc_valid": bool(score.get("valid")),
                    "projection_energy": score.get("projection_energy"),
                    "generic_smoothness_energy": generic,
                    "track_coverage": score.get("track_coverage"),
                    "survival_rate": score.get("survival_rate"),
                },
            )
    finally:
        del provider
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _apply_eligibility(rows: Sequence[dict[str, Any]], thresholds: Mapping[str, Any]) -> None:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    for condition_id, group in by_condition.items():
        bases = [row for row in group if row["candidate_role"] == "base_fixed"]
        if len(bases) != 1:
            raise NaturalRolloutCeilingError(f"{condition_id} 固定 Base 数量非 1")
        base = bases[0]
        for row in group:
            eligibility = candidate_eligibility(row, base, thresholds)
            row["eligibility"] = eligibility
            row["eligible"] = bool(eligibility["eligible"])


def _pairwise_rgb_rms(
    rows: Sequence[Mapping[str, Any]], work_dir: Path,
) -> dict[str, dict[str, float]]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    output: dict[str, dict[str, float]] = {}
    for condition_id, group in by_condition.items():
        natural = sorted(
            (row for row in group if row["candidate_role"] != "base_fixed"),
            key=lambda row: int(row["candidate_index"]),
        )
        tensors = {str(row["candidate_id"]): _load_frames(row, work_dir).float() for row in natural}
        values = {}
        for left_index, left in enumerate(natural):
            for right in natural[left_index + 1:]:
                left_id, right_id = str(left["candidate_id"]), str(right["candidate_id"])
                values[pair_key(left_id, right_id)] = float(
                    (tensors[left_id] - tensors[right_id]).square().mean().sqrt()
                )
        output[condition_id] = values
    return output


def _diversity_rows(
    rows: Sequence[Mapping[str, Any]],
    pairwise: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    return [
        {
            "condition_id": condition_id,
            **condition_diversity(group, pairwise.get(condition_id, {}), thresholds),
        }
        for condition_id, group in sorted(by_condition.items())
    ]


def _score_independent(
    cfg: Any,
    work_dir: Path,
    rows: Sequence[dict[str, Any]],
    metrics_log: JsonlMetrics,
    step_start: int,
) -> None:
    evaluator = CoTracker3IndependentEvaluator(dict(cfg.b0.evaluator))
    evaluator._load()
    try:
        for offset, row in enumerate(rows, start=1):
            frames = _load_frames(row, work_dir)
            detail, metrics = _score_case(
                evaluator,
                frames,
                outlier_threshold_px=float(cfg.b0.thresholds.acceleration_outlier_threshold_px),
            )
            plausibility = cotracker_plausibility_score(metrics) if bool(detail.get("valid")) else None
            row["cotracker"] = {
                "valid": bool(detail.get("valid")) and plausibility is not None,
                "plausibility_score": plausibility,
                "survival_rate": metrics.get("survival_rate"),
                "track_coverage": detail.get("track_coverage"),
                "metrics": metrics,
                "detail": detail,
                "uses_future_gt": False,
            }
            metrics_log.append(
                step_start + offset,
                {
                    "event": "independent_scored",
                    "candidate_id": row["candidate_id"],
                    "valid": row["cotracker"]["valid"],
                    "plausibility_score": plausibility,
                    "survival_rate": metrics.get("survival_rate"),
                    "velocity": metrics.get("image_plane_velocity_rms_px"),
                },
            )
    finally:
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _make_review_material(
    cfg: Any,
    work_dir: Path,
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["candidate_id"]): row for row in rows}
    review_dir = work_dir / "review"
    video_dir = review_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=False)
    public_rows = []
    key_rows = []
    templates = []
    required = {
        "punc_vs_random": int(cfg.b0.review.punc_vs_random_cases),
        "punc_vs_base": int(cfg.b0.review.punc_vs_base_cases),
    }
    for comparison_type, count in required.items():
        eligible_conditions = [
            row for row in gate["conditions"]
            if bool(row[comparison_type].get("comparable"))
            and row["selection"].get("punc_best_id")
            and row["selection"].get("random_id" if comparison_type == "punc_vs_random" else "base_id")
        ]
        if len(eligible_conditions) < count:
            raise NaturalRolloutCeilingError(f"review {comparison_type} 可用 condition 不足")
        for condition_row in eligible_conditions[:count]:
            sequence = len(public_rows) + 1
            case_id = f"b0-review-{sequence:03d}"
            punc_id = str(condition_row["selection"]["punc_best_id"])
            reference_key = "random_id" if comparison_type == "punc_vs_random" else "base_id"
            reference_id = str(condition_row["selection"][reference_key])
            ordered = [punc_id, reference_id]
            flip = int(hashlib.sha256(f"{cfg.b0.review.seed}:{case_id}".encode("utf-8")).hexdigest(), 16) % 2
            if flip:
                ordered.reverse()
            case_dir = video_dir / case_id
            case_dir.mkdir()
            paths = []
            for side, candidate_id in zip(("a", "b"), ordered):
                source = work_dir / str(by_id[candidate_id]["video_path"])
                suffix = source.suffix
                destination = case_dir / f"{side}{suffix}"
                shutil.copy2(source, destination)
                paths.append(str(destination.relative_to(work_dir)))
            public_rows.append(
                {
                    "case_id": case_id,
                    "video_a": paths[0],
                    "video_b": paths[1],
                    "question": "哪一侧在不减少合理驾驶运动、不损害画质与身份一致性的前提下更物理可信？",
                }
            )
            key_rows.append(
                {
                    "case_id": case_id,
                    "comparison_type": comparison_type,
                    "condition_id": condition_row["condition_id"],
                    "side_a_candidate_id": ordered[0],
                    "side_b_candidate_id": ordered[1],
                    "punc_best_side": "a" if ordered[0] == punc_id else "b",
                }
            )
            templates.append(
                {
                    "case_id": case_id,
                    "overall_preference": None,
                    "motion_plausibility": None,
                    "motion_amount": None,
                    "visual_quality": None,
                    "identity_consistency": None,
                    "low_motion_winner": None,
                    "catastrophic_failure": None,
                    "reason": "",
                }
            )
    _write_jsonl(review_dir / "cases.jsonl", public_rows)
    _write_jsonl(review_dir / "reviews.template.jsonl", templates)
    _write_jsonl(review_dir / "review_key.private.jsonl", key_rows)
    prompt = """# B0 natural rollout 盲审

每个 case 的 A/B 来自同一首帧、同一冻结 SVD-XT 与 official 25-step 协议，只改变独立 initial seed。
Stage A 请勿打开 `review_key.private.jsonl`，也不要根据“运动更多”直接判优。

逐项检查对象/道路/相机运动是否连贯，是否出现 freeze、slow-down、对象消失、track dropout、形变、模糊、
闪烁、首帧损坏或身份漂移。`overall_preference` 只允许 `a / b / tie / both_invalid`；
`motion_plausibility`、`visual_quality`、`identity_consistency` 只允许 `a / b / tie / both_invalid`；
`motion_amount` 只允许 `a_more / b_more / same / invalid`；两个布尔字段填写 true/false。

只有在一侧运动更可信且没有通过少动或灾难性画质获胜时，才给 decisive preference。
"""
    atomic_write_text(str(review_dir / "REVIEW_PROMPT.md"), prompt)
    return {
        "status": "awaiting_reviews",
        "case_count": len(public_rows),
        "cases_path": "review/cases.jsonl",
        "template_path": "review/reviews.template.jsonl",
        "prompt_path": "review/REVIEW_PROMPT.md",
        "private_key_path": "review/review_key.private.jsonl",
    }


def run_natural_rollout_ceiling(cfg: Any) -> dict[str, Any]:
    _validate_protocol(cfg)
    git = git_state(".")
    if git.get("dirty"):
        raise NaturalRolloutCeilingError("正式 B0 拒绝 dirty worktree")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"B0 run directory 已存在: {work_dir}")
    preflight = preflight_natural_rollout_ceiling(cfg)
    if preflight["status"] != "ready":
        raise NaturalRolloutCeilingError(f"B0 preflight blocked: {preflight['blockers']}")
    config_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "candidates").mkdir()
    thresholds = dict(cfg.b0.thresholds)
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=config_fp,
        cache_fingerprint=str(preflight["data"]["scene_split_fingerprint"]),
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split="nuScenes val 16 scene-distinct first-frame conditions",
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.b0.task_id),
        "preflight": preflight,
        "parent_r1_run": str(cfg.b0.parent_r1_run),
        "generation_protocol": "svd_official_v1/natural_independent_seed",
        "base_candidate_index": int(cfg.b0.base_candidate_index),
        "candidate_seed_start": int(cfg.b0.candidate_seed_start),
        "initial_candidate_count": int(cfg.b0.initial_candidate_count),
        "maximum_candidate_count": int(cfg.b0.maximum_candidate_count),
        "maximum_total_videos": int(cfg.b0.condition_count) * int(cfg.b0.maximum_candidate_count),
        "base_model_fingerprint": _base_model_fingerprint(str(cfg.model.pretrained)),
        "training_scorer": "RAFT+P-UNC plus generic RAFT smoothness",
        "independent_evaluator": "CoTracker3 offline",
        "uses_future_gt_for_generation": False,
        "uses_future_gt_for_generated_evaluation": False,
        "preregistered_thresholds": _json_safe(thresholds),
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics_log = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    try:
        seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
        _dataset, conditions, condition_tensors = _build_conditions(cfg)
        _write_jsonl(work_dir / "conditions.jsonl", conditions)
        manifest_data["conditions"] = conditions
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)

        initial_indices = list(range(int(cfg.b0.initial_candidate_count)))
        candidate_rows = _generate_candidate_indices(
            cfg, work_dir, conditions, condition_tensors, initial_indices, metrics_log, 0,
        )
        _validate_candidate_noise(candidate_rows)
        _score_training_side(cfg, work_dir, candidate_rows, metrics_log, len(candidate_rows))
        _apply_eligibility(candidate_rows, thresholds)
        pairwise = _pairwise_rgb_rms(candidate_rows, work_dir)
        diversity = _diversity_rows(candidate_rows, pairwise, thresholds)
        _write_jsonl(work_dir / "diversity_n4.jsonl", diversity)
        diverse_n4 = sum(bool(row["diverse"]) for row in diversity)
        expanded = diverse_n4 < int(cfg.b0.minimum_diverse_conditions_before_expansion)

        if expanded:
            extra_indices = list(range(int(cfg.b0.initial_candidate_count), int(cfg.b0.maximum_candidate_count)))
            new_rows = _generate_candidate_indices(
                cfg, work_dir, conditions, condition_tensors, extra_indices,
                metrics_log, len(candidate_rows) * 2,
            )
            _score_training_side(
                cfg, work_dir, new_rows, metrics_log, len(candidate_rows) * 2 + len(new_rows),
            )
            candidate_rows.extend(new_rows)
            _validate_candidate_noise(candidate_rows)
            _apply_eligibility(candidate_rows, thresholds)
            pairwise = _pairwise_rgb_rms(candidate_rows, work_dir)
            diversity = _diversity_rows(candidate_rows, pairwise, thresholds)
            _write_jsonl(work_dir / "diversity_n8.jsonl", diversity)

        _write_jsonl(work_dir / "generation_cases.jsonl", candidate_rows)
        _write_jsonl(
            work_dir / "pairwise_rgb_rms.jsonl",
            [
                {"condition_id": condition_id, "pair_key": key, "rgb_rms": value}
                for condition_id, values in sorted(pairwise.items())
                for key, value in sorted(values.items())
            ],
        )
        _score_independent(
            cfg, work_dir, candidate_rows, metrics_log,
            len(candidate_rows) * 3,
        )
        _write_jsonl(work_dir / "scored_candidates.jsonl", candidate_rows)
        gate = aggregate_b0_gate(
            candidate_rows,
            pairwise,
            thresholds,
            selection_seed=int(cfg.b0.selection_seed),
        )
        atomic_write_json(str(work_dir / "machine_gate.json"), _json_safe(gate))
        _write_jsonl(work_dir / "condition_rankings.jsonl", gate["conditions"])
        machine_pass = bool(gate["machine_pass"])
        review = (
            _make_review_material(cfg, work_dir, gate, candidate_rows)
            if machine_pass else {
                "status": "not_created_machine_gate_failed",
                "case_count": 0,
                "reason": "human verdict cannot promote a machine-rejected route",
            }
        )
        result = {
            "task_id": str(cfg.b0.task_id),
            "candidate_count_per_condition": int(cfg.b0.maximum_candidate_count if expanded else cfg.b0.initial_candidate_count),
            "total_video_count": len(candidate_rows),
            "expanded_to_n8": expanded,
            "diverse_conditions_at_n4": diverse_n4,
            "final_diverse_conditions": sum(bool(row["diverse"]) for row in diversity),
            "machine_gate": gate,
            "review": review,
            "uses_future_gt_for_generation": False,
            "uses_future_gt_for_generated_evaluation": False,
        }
        atomic_write_json(str(work_dir / "result.json"), _json_safe(result))
        status = "awaiting_reviews" if machine_pass else "rejected"
        next_gate = "RP-D0-08" if machine_pass else "RP-C0-07"
        summary = {
            "status": status,
            "task_id": str(cfg.b0.task_id),
            "run_id": str(cfg.run_id),
            "config_fingerprint": config_fp,
            "scene_split_fingerprint": str(preflight["data"]["scene_split_fingerprint"]),
            "candidate_count_per_condition": result["candidate_count_per_condition"],
            "total_video_count": len(candidate_rows),
            "expanded_to_n8": expanded,
            "diverse_conditions_at_n4": diverse_n4,
            "final_diverse_conditions": result["final_diverse_conditions"],
            "machine_pass": machine_pass,
            "punc_vs_random_win_credit_rate": gate["punc_vs_random_win_credit_rate"],
            "punc_vs_base_win_credit_rate": gate["punc_vs_base_win_credit_rate"],
            "review_status": review["status"],
            "review_case_count": review["case_count"],
            "result_fingerprint": sha256_json(_json_safe(result)),
            "next_gate": next_gate,
        }
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        if not machine_pass:
            atomic_write_text(str(work_dir / "REJECTED"), sha256_json(summary) + "\n")
        manifest_data.update(
            {
                "status": "completed",
                "ended_at": utc_now(),
                "exit_reason": "machine_pass_awaiting_reviews" if machine_pass else "machine_gate_rejected",
                "expanded_to_n8": expanded,
                "machine_pass": machine_pass,
            }
        )
        atomic_write_json(str(work_dir / "manifest.json"), _json_safe(manifest_data))
        return summary
    except Exception as exc:
        failure = {
            "status": "failed", "task_id": str(cfg.b0.task_id), "run_id": str(cfg.run_id), "error": repr(exc),
        }
        atomic_write_json(str(work_dir / "summary.json"), failure)
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), _json_safe(manifest_data))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="natural independent-rollout best-of-N ceiling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, list(args.overrides))
    result = preflight_natural_rollout_ceiling(cfg) if args.preflight else run_natural_rollout_ceiling(cfg)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
