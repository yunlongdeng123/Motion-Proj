"""固定 seed 的生成评估：对比 base SVD 与各 LoRA checkpoint（方案第 10 节）。

对固定的 clip 子集、固定随机种子，逐 checkpoint 从条件帧采样未来片段，离线
审计静态漂移，并与 GT 未来帧计算视觉相似度（LPIPS/PSNR/SSIM），产出：

  - 每个 (adapter, clip) 的生成视频，以及可选的 GT|gen 对比视频
  - 每个 (adapter, clip) 的 per-clip 指标 JSON
  - 跨 clip 聚合 + 跨 adapter 对比的 summary.json（含推荐的最稳 checkpoint）

设计要点：
  - ``base`` 表示禁用 LoRA 的冻结 SVD（与训练里 anchor 预测同一条路径）。
  - 每个 clip 用 ``seed + clip_index`` 独立播种，保证同一 clip 在不同 adapter
    之间使用完全相同的初始噪声，从而可比。
  - FVD 需要 I3D 权重，V1 未实现，故不作为阻塞项；这里聚焦静态漂移 + 视觉相似度。

运行提示：直接调用本模块时建议加 override ``model.enable_xformers=false``——xformers
flash-attention 在生成阶段的时空注意力形状下会报 "invalid configuration argument"，
禁用后回退到稳健的 PyTorch SDPA（``scripts/eval_adapters.sh`` 已默认关闭）。
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch

from ..auditor import MotionAuditor
from ..config import get_paths, load_config
from ..data import NuScenesFutureVideoDataset
from ..utils.io import save_json, to_uint8_video, write_video
from ..utils.logging import get_logger
from ..utils.viz import make_comparison_panel
from ..runtime.tasks import TaskStore
from . import metrics as M
from .drivinggen import PROTOCOL as DRIVINGGEN_PROTOCOL

log = get_logger(__name__)

BASE_NAME = "base"
_METRIC_KEYS = (
    "static_drift", "lpips", "psnr", "ssim", "fid_future", "fvd8",
    "drivinggen_scene_consistency", "drivinggen_agent_consistency",
    "drivinggen_agent_disappearance_consistency", "speed_consistency",
    "acceleration_consistency", "trajectory_consistency",
)
# 是否越低越好（用于聚合排序）
_LOWER_IS_BETTER = {"static_drift": True, "lpips": True, "psnr": False, "ssim": False}


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _audit_generated(auditor: MotionAuditor, gen_frames: torch.Tensor, src: dict):
    """审计生成片段：复用源片段几何，生成帧上没有 GT 框（物体层为空）。"""
    sample = {
        "frames": gen_frames,
        "boxes": [[] for _ in range(gen_frames.shape[0])],
        "intrinsics": src["intrinsics"],
        "cam2ego": src["cam2ego"],
        "ego2global": src["ego2global"],
        "sample_id": src["sample_id"] + "_gen",
    }
    return auditor.audit(sample)


def _safe_metric(fn, *args):
    """计算单个指标；失败（如缺权重/依赖）时记录并返回 None，不阻断评估。"""
    try:
        val = float(fn(*args))
        if val != val:  # NaN
            return None
        return val
    except Exception as exc:  # pragma: no cover - 依赖环境
        log.warning("metric %s failed: %s", getattr(fn, "__name__", fn), exc)
        return None


def resolve_adapters(spec: list[str], adapter_dir: str) -> list[tuple[str, str | None]]:
    """把 ['base','adapter_step1000','/abs/x.safetensors'] 解析成 [(name, path|None)]。"""
    out: list[tuple[str, str | None]] = []
    for raw in spec:
        tok = raw.strip()
        if not tok:
            continue
        if tok.lower() == BASE_NAME:
            out.append((BASE_NAME, None))
            continue
        if os.path.isabs(tok) or tok.endswith(".safetensors"):
            path = tok
            name = os.path.splitext(os.path.basename(tok))[0]
        else:
            path = os.path.join(adapter_dir, tok + ".safetensors")
            name = tok
        out.append((name, path))
    return out


def select_clip_indices(n_total: int, num_clips: int, explicit: list[int] | None) -> list[int]:
    """固定、确定性的 clip 子集：显式指定优先，否则在全集上等间隔取样。"""
    if explicit:
        return sorted({i for i in explicit if 0 <= i < n_total})
    num = min(max(1, num_clips), n_total)
    if num >= n_total:
        return list(range(n_total))
    step = n_total / num
    return sorted({int(i * step) for i in range(num)})


def _aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    for key in _METRIC_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = sum(vals) / len(vals)
            out[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out[f"{key}_n"] = len(vals)
        else:
            out[f"{key}_mean"] = None
            out[f"{key}_n"] = 0
    return out


def _rank(adapters: dict) -> dict:
    def sort_key(metric: str):
        lower = _LOWER_IS_BETTER[metric]
        key = f"{metric}_mean"

        def _k(item):
            v = item[1]["aggregate"].get(key)
            if v is None:
                return (1, 0.0)
            return (0, v if lower else -v)

        return _k

    items = list(adapters.items())
    by_drift = [k for k, _ in sorted(items, key=sort_key("static_drift"))]
    by_lpips = [k for k, _ in sorted(items, key=sort_key("lpips"))]
    ckpts = [k for k in by_drift if k != BASE_NAME]
    recommended = ckpts[0] if ckpts else (by_drift[0] if by_drift else None)
    return {
        "by_static_drift": by_drift,
        "by_lpips": by_lpips,
        "recommended_checkpoint": recommended,
        "criterion": "lowest mean static_drift among LoRA checkpoints (base 仅作参考)",
    }


def _log_ranking(summary: dict) -> None:
    log.info("==== generate-eval 聚合结果 ====")
    for name, blob in summary["adapters"].items():
        agg = blob["aggregate"]
        log.info(
            "%-20s drift=%s lpips=%s psnr=%s ssim=%s (n=%s)",
            name,
            _fmt(agg.get("static_drift_mean")),
            _fmt(agg.get("lpips_mean")),
            _fmt(agg.get("psnr_mean")),
            _fmt(agg.get("ssim_mean")),
            agg.get("static_drift_n"),
        )
    rank = summary.get("ranking", {})
    log.info("按静态漂移排序: %s", rank.get("by_static_drift"))
    log.info("推荐 checkpoint: %s", rank.get("recommended_checkpoint"))


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


@torch.no_grad()
def run_generate_eval(
    cfg,
    adapters: list[tuple[str, str | None]],
    seed: int,
    out_dir: str,
    num_frames: int,
    num_inference_steps: int,
    num_clips: int = 4,
    clip_indices: list[int] | None = None,
    save_video: bool = True,
    save_compare: bool = True,
) -> dict:
    device = cfg.device
    _seed_everything(seed)

    ds = NuScenesFutureVideoDataset(cfg.data)
    camera = ds.camera
    indices = select_clip_indices(len(ds), num_clips, clip_indices)
    log.info("Generate-eval on %d clips (indices=%s), seed=%d", len(indices), indices, seed)

    from ..backbones import build_backbone

    backbone = build_backbone(cfg.model, load=True, device=device)
    auditor = MotionAuditor(device=device, enable_depth=True)
    lora_enabled = bool(cfg.model.lora.enable)

    os.makedirs(out_dir, exist_ok=True)
    tasks = TaskStore(os.path.join(out_dir, "_tasks"))
    # 预取 clip 源数据（GT 未来帧 + 几何），避免逐 adapter 重复解码数据集
    clips = {idx: ds[idx] for idx in indices}

    summary: dict = {
        "seed": seed,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "clip_indices": indices,
        "clip_ids": [clips[i]["sample_id"] for i in indices],
        "camera": camera,
        "metric_protocol": DRIVINGGEN_PROTOCOL,
        "static_drift_role": "internal_diagnostic_not_standard_benchmark",
        "adapters": {},
    }

    for name, path in adapters:
        if name == BASE_NAME:
            if lora_enabled:
                backbone._set_lora_enabled(False)
        else:
            if path is None or not os.path.isfile(path):
                log.error("adapter file missing: %s (skip %s)", path, name)
                continue
            backbone.load_adapter(path)
            if lora_enabled:
                backbone._set_lora_enabled(True)
            else:
                log.warning("model.lora.enable=false; adapter %s 无法真正生效", name)

        adapter_out = os.path.join(out_dir, name)
        os.makedirs(adapter_out, exist_ok=True)
        per_clip: list[dict] = []

        for idx in indices:
            src = clips[idx]
            sid = src["sample_id"]
            task_seed = seed + idx
            cached = tasks.completed_result(name, task_seed, sid, camera)
            if cached is not None:
                per_clip.append(cached)
                log.info("[%s] %s 已完成，跳过生成", name, sid)
                continue
            tasks.mark(name, task_seed, sid, "running", camera=camera)
            gt = src["frames"].to(device)          # [K,3,H,W]
            _, _, gh, gw = gt.shape
            gen = torch.Generator(device=device).manual_seed(task_seed)
            # 显式指定 height/width，否则 pipeline 会用 SVD-xt 原生分辨率(576x1024)，
            # 与训练/GT 的 256x448 不一致，导致无法与 GT 逐帧比较。
            decode_chunk = 4
            while True:
                try:
                    frames = backbone.generation(
                        src["cond_frame"].to(device), num_frames=num_frames,
                        num_inference_steps=num_inference_steps, generator=gen,
                        height=gh, width=gw, decode_chunk_size=decode_chunk,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    if decode_chunk <= 1:
                        tasks.mark(name, task_seed, sid, "failed", camera=camera, reason="decode_oom")
                        raise
                    decode_chunk = max(1, decode_chunk // 2)
                    torch.cuda.empty_cache()
                    gen = torch.Generator(device=device).manual_seed(task_seed)
                    log.warning("评估 decode OOM，保持任务语义并将 chunk 减至 %d", decode_chunk)
            k = min(frames.shape[0], gt.shape[0])
            fk, gk = frames[:k], gt[:k]

            state = _audit_generated(auditor, fk, src)
            row = {
                "checkpoint": name,
                "seed": task_seed,
                "camera": camera,
                "clip_index": idx,
                "sample_id": sid,
                "static_drift": _safe_metric(auditor.static_drift_score, state),
                "lpips": _safe_metric(M.lpips_distance, fk, gk),
                "psnr": _safe_metric(M.psnr, fk, gk),
                "ssim": _safe_metric(M.ssim, fk, gk),
                "metric_protocol": DRIVINGGEN_PROTOCOL["name"],
                "static_drift_role": "internal_diagnostic",
            }
            per_clip.append(row)
            tasks.mark(name, task_seed, sid, "completed", camera=camera,
                       result=row, decode_chunk_size=decode_chunk)
            save_json(row, os.path.join(adapter_out, f"{sid}.json"))
            if save_video:
                write_video(to_uint8_video(fk), os.path.join(adapter_out, f"{sid}.mp4"), fps=4)
            if save_compare:
                write_video(
                    make_comparison_panel(gk, fk),
                    os.path.join(adapter_out, f"{sid}_cmp.mp4"),
                    fps=4,
                )
            log.info(
                "[%s] %s | drift=%s lpips=%s psnr=%s ssim=%s",
                name, sid, _fmt(row["static_drift"]), _fmt(row["lpips"]),
                _fmt(row["psnr"]), _fmt(row["ssim"]),
            )

        blob = {"per_clip": per_clip, "aggregate": _aggregate(per_clip)}
        summary["adapters"][name] = blob
        save_json(blob, os.path.join(adapter_out, "adapter_summary.json"))

    summary["ranking"] = _rank(summary["adapters"])
    save_json(summary, os.path.join(out_dir, "summary.json"))
    _log_ranking(summary)
    log.info("Generate-eval done -> %s", os.path.join(out_dir, "summary.json"))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="固定 seed 的 base SVD vs LoRA 生成评估")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--adapters",
        default="base,adapter_final",
        help="逗号分隔；'base'=不加 LoRA；其余为 ckpt_dir 下的名字或 .safetensors 绝对路径",
    )
    ap.add_argument("--adapter-dir", default=None, help="默认 = paths.ckpt_dir")
    ap.add_argument("--num-clips", type=int, default=4)
    ap.add_argument("--clip-indices", default=None, help="逗号分隔的显式 clip 下标，覆盖 --num-clips")
    ap.add_argument("--seed", type=int, default=None, help="默认 = cfg.seed")
    ap.add_argument("--num-frames", type=int, default=None, help="默认 = cfg.data.num_frames")
    ap.add_argument("--num-inference-steps", type=int, default=25)
    ap.add_argument("--out-dir", default=None, help="默认 = <work_dir>/eval/generate")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    paths = get_paths(cfg)
    adapter_dir = args.adapter_dir or paths.ckpt_dir
    adapters = resolve_adapters(args.adapters.split(","), adapter_dir)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 1234))
    num_frames = args.num_frames or int(cfg.data.num_frames)
    out_dir = args.out_dir or os.path.join(str(cfg.work_dir), "eval", "generate")
    explicit = [int(x) for x in args.clip_indices.split(",")] if args.clip_indices else None

    run_generate_eval(
        cfg,
        adapters=adapters,
        seed=seed,
        out_dir=out_dir,
        num_frames=num_frames,
        num_inference_steps=int(args.num_inference_steps),
        num_clips=int(args.num_clips),
        clip_indices=explicit,
        save_video=not args.no_video,
        save_compare=not args.no_compare,
    )


if __name__ == "__main__":
    main()
