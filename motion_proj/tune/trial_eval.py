"""Trial 轻量生成评估：写出 Optuna 需要的动力学与视觉指标。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from ..auditor import MotionAuditor
from ..auditor.state import Track
from ..config import load_config
from ..data import NuScenesFutureVideoDataset
from ..eval import metrics as M
from ..eval.generate_eval import select_clip_indices
from ..runtime.atomic import atomic_write_json
from ..utils.logging import get_logger
from .trial_summary import build_trial_metrics, merge_trial_summary

log = get_logger(__name__)


def _clone_track(track: Track) -> Track:
    return Track(
        instance_token=track.instance_token,
        category=track.category,
        xyxy=track.xyxy.clone(),
        depth=track.depth.clone(),
        present=track.present.clone(),
    )


def warp_tracks_by_flow(tracks: list[Track], flow: torch.Tensor) -> list[Track]:
    """用生成帧光流推进 GT 框，得到依赖生成质量的伪轨迹后再算加速度。"""
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"flow 形状必须为 [F,H,W,2]，得到 {tuple(flow.shape)}")
    frames = int(flow.shape[0]) + 1
    height, width = int(flow.shape[1]), int(flow.shape[2])
    warped: list[Track] = []
    for track in tracks:
        item = _clone_track(track)
        xyxy = item.xyxy.clone()
        for t in range(frames - 1):
            if not bool(item.present[t] and item.present[t + 1]):
                continue
            u0, v0, u1, v1 = [float(v) for v in xyxy[t]]
            cu = 0.5 * (u0 + u1)
            cv = 0.5 * (v0 + v1)
            ui = int(round(min(max(cu, 0.0), width - 1)))
            vi = int(round(min(max(cv, 0.0), height - 1)))
            du, dv = [float(v) for v in flow[t, vi, ui]]
            xyxy[t + 1, 0] = u0 + du
            xyxy[t + 1, 1] = v0 + dv
            xyxy[t + 1, 2] = u1 + du
            xyxy[t + 1, 3] = v1 + dv
        item.xyxy = xyxy
        warped.append(item)
    return warped


def _mean(values: list[float | None]) -> float:
    finite = [float(v) for v in values if v is not None and v == v]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def evaluate_adapter_on_clips(
    backbone,
    dataset,
    indices: list[int],
    *,
    seed: int,
    num_frames: int,
    num_inference_steps: int,
    device: str,
) -> dict:
    """对固定 clip 子集生成未来帧，并聚合 trial 原始指标。"""
    auditor = MotionAuditor(device=device, enable_depth=True)
    rows: list[dict] = []
    for idx in indices:
        src = dataset[idx]
        gt = src["frames"].to(device)
        _, _, height, width = gt.shape
        task_seed = int(seed) + int(idx)
        generator = torch.Generator(device=device).manual_seed(task_seed)
        decode_chunk = 4
        while True:
            try:
                frames = backbone.generation(
                    src["cond_frame"].to(device),
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    height=height,
                    width=width,
                    decode_chunk_size=decode_chunk,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if decode_chunk <= 1:
                    raise
                decode_chunk = max(1, decode_chunk // 2)
                torch.cuda.empty_cache()
                generator = torch.Generator(device=device).manual_seed(task_seed)
                log.warning("trial eval decode OOM，chunk=%d", decode_chunk)

        k = min(int(frames.shape[0]), int(gt.shape[0]))
        generated = frames[:k]
        reference = gt[:k]
        sample = {
            "frames": generated,
            "boxes": src["boxes"][:k],
            "intrinsics": src["intrinsics"],
            "cam2ego": src["cam2ego"],
            "ego2global": src["ego2global"],
            "lidar_depth": src.get("lidar_depth"),
            "sample_id": src["sample_id"] + "_trial",
        }
        state = auditor.audit(sample)
        warped = warp_tracks_by_flow(state.tracks, state.u_static)
        rows.append({
            "sample_id": src["sample_id"],
            "clip_index": int(idx),
            "static_drift": float(auditor.static_drift_score(state)),
            "track_acceleration": float(M.track_acceleration(warped)),
            "lpips": float(M.lpips_distance(generated, reference)),
            "projection_eligible_fraction": float(state.static_mask.float().mean()),
        })
    return {
        "static_drift": _mean([row["static_drift"] for row in rows]),
        "track_acceleration": _mean([row["track_acceleration"] for row in rows]),
        "lpips": _mean([row["lpips"] for row in rows]),
        "projection_eligible_fraction": _mean(
            [row["projection_eligible_fraction"] for row in rows]
        ),
        "per_clip": rows,
        "clip_indices": list(indices),
        "num_inference_steps": int(num_inference_steps),
        "seed": int(seed),
    }


def _load_backbone(cfg, adapter_path: str | None):
    from ..backbones import build_backbone

    device = str(cfg.device)
    backbone = build_backbone(cfg.model, load=True, device=device)
    if adapter_path:
        backbone.load_adapter(adapter_path)
        if bool(cfg.model.lora.enable):
            backbone._set_lora_enabled(True)
    elif bool(cfg.model.lora.enable):
        backbone._set_lora_enabled(False)
    return backbone


def collect_raw_metrics(
    cfg,
    *,
    adapter_path: str | None,
    num_clips: int = 4,
    clip_indices: list[int] | None = None,
    num_inference_steps: int = 8,
) -> dict:
    dataset = NuScenesFutureVideoDataset(cfg.data)
    indices = select_clip_indices(len(dataset), num_clips, clip_indices)
    backbone = _load_backbone(cfg, adapter_path)
    return evaluate_adapter_on_clips(
        backbone,
        dataset,
        indices,
        seed=int(cfg.get("seed", 1234)),
        num_frames=int(cfg.data.num_frames),
        num_inference_steps=num_inference_steps,
        device=str(cfg.device),
    )


def run_trial_eval(
    cfg,
    *,
    adapter_path: str | None,
    base_metrics: Mapping[str, float],
    out_summary: str,
    train_summary: Mapping | None = None,
    num_clips: int = 4,
    clip_indices: list[int] | None = None,
    num_inference_steps: int = 8,
) -> dict:
    """评估单个 adapter（或 base），并把 Optuna 字段写入 summary。"""
    raw = collect_raw_metrics(
        cfg,
        adapter_path=adapter_path,
        num_clips=num_clips,
        clip_indices=clip_indices,
        num_inference_steps=num_inference_steps,
    )
    metrics = build_trial_metrics(raw, base_metrics)
    metrics["eval"] = {
        "clip_indices": raw["clip_indices"],
        "per_clip": raw["per_clip"],
        "num_inference_steps": raw["num_inference_steps"],
        "adapter_path": adapter_path,
    }
    summary = merge_trial_summary(train_summary or {"status": "completed"}, metrics)
    atomic_write_json(out_summary, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="写入 Optuna trial 评估摘要")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter；缺省评估 base")
    parser.add_argument("--base-metrics", default=None, help="含 base static/track/lpips 的 JSON")
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--train-summary", default=None)
    parser.add_argument("--write-base-metrics", action="store_true",
                        help="只评估冻结 base，写出可作为 --base-metrics 的原始指标")
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--clip-indices", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    explicit = (
        [int(x) for x in args.clip_indices.split(",")] if args.clip_indices else None
    )
    if args.write_base_metrics:
        raw = collect_raw_metrics(
            cfg,
            adapter_path=None,
            num_clips=int(args.num_clips),
            clip_indices=explicit,
            num_inference_steps=int(args.num_inference_steps),
        )
        payload = {
            "static_drift": raw["static_drift"],
            "track_acceleration": raw["track_acceleration"],
            "lpips": raw["lpips"],
            "projection_eligible_fraction": raw["projection_eligible_fraction"],
            "eval": {
                "clip_indices": raw["clip_indices"],
                "per_clip": raw["per_clip"],
                "num_inference_steps": raw["num_inference_steps"],
                "role": "base",
            },
        }
        atomic_write_json(args.out_summary, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not args.base_metrics:
        raise SystemExit("非 --write-base-metrics 模式必须提供 --base-metrics")
    base = json.loads(Path(args.base_metrics).read_text(encoding="utf-8"))
    train = None
    if args.train_summary:
        train = json.loads(Path(args.train_summary).read_text(encoding="utf-8"))
    summary = run_trial_eval(
        cfg,
        adapter_path=args.adapter,
        base_metrics=base,
        out_summary=args.out_summary,
        train_summary=train,
        num_clips=int(args.num_clips),
        clip_indices=explicit,
        num_inference_steps=int(args.num_inference_steps),
    )
    print(json.dumps({k: summary[k] for k in (
        "static_drift", "track_acceleration", "lpips",
        "projection_eligible_fraction",
        "normalized_static_drift_improvement",
        "normalized_track_acceleration_improvement",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
