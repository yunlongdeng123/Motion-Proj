"""使用冻结 synthetic checkpoint 挖掘高漂移 replay cache。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

from ..auditor import MotionAuditor
from ..auditor.state import MotionState
from ..backbones import build_backbone
from ..cache.build_cache import _flow_to_resolution
from ..cache.writer import ProjectionCacheWriter
from ..config import cache_config_fingerprint, get_paths, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..projector import DynamicsProjector
from ..projector.mask import downsample_mask_to_latent
from ..runtime.atomic import atomic_write_text
from ..runtime.fingerprint import file_fingerprint, git_state, sha256_json
from ..runtime.stage import StageManifest
from ..runtime.tasks import TaskStore
from ..utils.io import load_json
from ..utils.logging import get_logger

log = get_logger(__name__)


def _adapter_file(path: str) -> str:
    candidate = os.path.join(path, "adapter.safetensors") if os.path.isdir(path) else path
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"synthetic checkpoint adapter 不存在: {candidate}")
    return os.path.abspath(candidate)


def _audit_generated(auditor: MotionAuditor, gen_frames: torch.Tensor, src_sample: dict) -> MotionState:
    """审计生成片段时仅复用相机/自车几何，不把 GT 框泄漏给生成结果。"""
    sample = {
        "frames": gen_frames,
        "boxes": [[] for _ in range(gen_frames.shape[0])],
        "intrinsics": src_sample["intrinsics"],
        "cam2ego": src_sample["cam2ego"],
        "ego2global": src_sample["ego2global"],
        "sample_id": src_sample["sample_id"] + "_gen",
    }
    return auditor.audit(sample)


def replay_energy_decreased(result, tolerance: float = 1e-6) -> bool:
    """要求投影后总能量严格下降；禁止空 track 时 obj/prior 的 0<=0 虚报。"""
    before = result.energy_before if hasattr(result, "energy_before") else {}
    after = result.energy_after if hasattr(result, "energy_after") else {}
    if "total" in before and "total" in after:
        return float(after["total"]) < float(before["total"]) - tolerance
    return bool(result.diagnostics.get("energy_decreased", False))


def replay_is_eligible(drift: float, result, drift_thresh: float,
                       min_eligible_fraction: float = 0.70) -> bool:
    diagnostics = result.diagnostics
    return (
        drift >= drift_thresh
        and replay_energy_decreased(result)
        and float(diagnostics.get("eligible_fraction", 0.0)) >= min_eligible_fraction
    )


def _generation_settings(cfg) -> dict:
    return {
        "num_inference_steps": int(cfg.cache.get("num_inference_steps", 25)),
        "decode_chunk_size": int(cfg.cache.get("decode_chunk_size", 4)),
    }


def _summarize_rows(rows: list[dict]) -> dict:
    eligible = [float(row["eligible_fraction"]) for row in rows
                if row.get("eligible_fraction") is not None]
    return {
        "considered": len(rows),
        "kept": sum(bool(row.get("kept")) for row in rows),
        "rejected": {
            reason: sum(row.get("reject_reason") == reason for row in rows)
            for reason in ("low_drift", "energy", "eligible")
        },
        "eligible_fraction": {
            "mean": sum(eligible) / len(eligible) if eligible else None,
            "min": min(eligible) if eligible else None,
            "max": max(eligible) if eligible else None,
            "n": len(eligible),
        },
    }


@torch.no_grad()
def mine(cfg, adapter: str, drift_thresh: float, max_samples: int,
         min_eligible_fraction: float = 0.70) -> dict:
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 replay mining 拒绝在 dirty worktree 上运行")
    adapter = _adapter_file(adapter)
    parent_fingerprint = file_fingerprint(adapter)
    paths = get_paths(cfg)
    device = cfg.device
    ds = NuScenesFutureVideoDataset(cfg.data)
    backbone = build_backbone(cfg.model, load=True, device=device)
    backbone.load_adapter(adapter)
    backbone.set_train_mode(False)
    auditor = MotionAuditor(device=device)
    projector = DynamicsProjector()
    n = min(max_samples, len(ds)) if max_samples else len(ds)
    generation = _generation_settings(cfg)
    replay_fingerprint = sha256_json({
        "cache": cache_config_fingerprint(cfg), "parent": parent_fingerprint,
        "seed": int(cfg.seed), "drift_thresh": drift_thresh,
        "min_eligible_fraction": min_eligible_fraction, "samples": n,
        "generation": generation,
    })
    writer = ProjectionCacheWriter(
        paths.cache_dir, store=cfg.cache.store, overwrite=False,
        fingerprint=replay_fingerprint,
    )
    stage = StageManifest(os.path.join(paths.cache_dir, "_stage"), "replay", replay_fingerprint)
    if stage.is_complete():
        return {"status": "completed", "fingerprint": replay_fingerprint, "skipped": True}
    stage.begin({
        "source": "replay", "parent_checkpoint": adapter,
        "parent_fingerprint": parent_fingerprint, "command": list(sys.argv),
        "git": git, "samples_considered": n, "generation": generation,
    })
    save_resolved_config(cfg, os.path.join(paths.cache_dir, "_stage", "resolved.yaml"))
    scale = int(cfg.model.vae_scale_factor)
    tasks = TaskStore(os.path.join(paths.cache_dir, "_stage", "tasks"))
    rows: list[dict] = []
    camera = str(getattr(ds, "camera", "CAM_FRONT"))
    try:
        for i in tqdm(range(n)):
            src = ds[i]
            sid = src["sample_id"] + "_replay"
            task_seed = int(cfg.seed) + i
            cached = tasks.completed_result(parent_fingerprint, task_seed, sid, camera)
            if cached is not None:
                rows.append(cached)
                continue
            if writer.exists(sid):
                metadata = load_json(os.path.join(paths.cache_dir, sid, "metadata.json"))
                row = {
                    "clip_index": i, "sample_id": sid, "generation_seed": task_seed,
                    "drift": metadata.get("drift"),
                    "energy_decreased": metadata.get("energy_decreased"),
                    "eligible_fraction": metadata.get("eligible_fraction"),
                    "kept": True, "reject_reason": None,
                }
                tasks.mark(parent_fingerprint, task_seed, sid, "completed",
                           camera=camera, result=row)
                rows.append(row)
                continue
            tasks.mark(parent_fingerprint, task_seed, sid, "running", camera=camera)
            generator = torch.Generator(device=device).manual_seed(task_seed)
            gen = backbone.generation(
                src["cond_frame"].to(device), num_frames=int(cfg.data.num_frames),
                num_inference_steps=generation["num_inference_steps"],
                generator=generator, height=int(cfg.data.height), width=int(cfg.data.width),
                decode_chunk_size=generation["decode_chunk_size"],
            )
            state = _audit_generated(auditor, gen, src)
            drift = auditor.static_drift_score(state)
            row = {
                "clip_index": i, "sample_id": sid, "generation_seed": task_seed,
                "drift": drift, "energy_decreased": None, "eligible_fraction": None,
                "kept": False, "reject_reason": None,
            }
            if drift < drift_thresh:
                row["reject_reason"] = "low_drift"
                tasks.mark(parent_fingerprint, task_seed, sid, "completed",
                           camera=camera, result=row)
                rows.append(row)
                continue
            result = projector.project(gen, state)
            row["energy_decreased"] = replay_energy_decreased(result)
            row["eligible_fraction"] = float(result.diagnostics.get("eligible_fraction", 0.0))
            row["energy_before"] = float(result.energy_before.get("total", 0.0))
            row["energy_after"] = float(result.energy_after.get("total", 0.0))
            row["num_tracks"] = int(result.diagnostics.get("num_tracks", 0))
            if not row["energy_decreased"]:
                row["reject_reason"] = "energy"
                tasks.mark(parent_fingerprint, task_seed, sid, "completed",
                           camera=camera, result=row)
                rows.append(row)
                continue
            if row["eligible_fraction"] < min_eligible_fraction:
                row["reject_reason"] = "eligible"
                tasks.mark(parent_fingerprint, task_seed, sid, "completed",
                           camera=camera, result=row)
                rows.append(row)
                continue

            metadata = dict(result.metadata)
            metadata.update({
                "sample_id": sid, "source": "replay", "drift": drift,
                "energy_decreased": row["energy_decreased"],
                "eligible_fraction": row["eligible_fraction"],
                "generation_seed": task_seed, "parent_checkpoint": adapter,
                "parent_fingerprint": parent_fingerprint,
            })
            if cfg.cache.store == "rgb":
                writer.write(
                    sid, gen.cpu(), result.x_dagger.cpu(), result.mask.cpu(), metadata,
                    clean=src["frames"].cpu(), latent_flow=state.u_static.cpu(),
                    flow_confidence=state.flow_conf.cpu().unsqueeze(1), source="replay",
                    generation_seed=task_seed, parent_checkpoint=adapter,
                    source_fingerprint=parent_fingerprint,
                )
            else:
                cond = backbone.build_conditioning({"cond_frame": src["cond_frame"].unsqueeze(0)})
                clean_lat = backbone.encode(src["frames"].to(device).unsqueeze(0))[0]
                y_lat = backbone.encode(gen.unsqueeze(0))[0]
                target_lat = backbone.encode(result.x_dagger.unsqueeze(0))[0]
                mask_lat = downsample_mask_to_latent(result.mask.to(device), scale)
                flow_lat, confidence_lat = _flow_to_resolution(
                    state.u_static, state.flow_conf.unsqueeze(1), y_lat.shape[-2], y_lat.shape[-1]
                )
                context = {key: cond.data[key][0] for key in
                           ("image_embeds", "image_latents", "added_time_ids")}
                writer.write(
                    sid, y_lat.cpu(), target_lat.cpu(), mask_lat.cpu(), metadata, context,
                    clean=clean_lat.cpu(), latent_flow=flow_lat.cpu(),
                    flow_confidence=confidence_lat.cpu(), source="replay",
                    generation_seed=task_seed, parent_checkpoint=adapter,
                    source_fingerprint=parent_fingerprint,
                )
            row["kept"] = True
            tasks.mark(parent_fingerprint, task_seed, sid, "completed",
                       camera=camera, result=row)
            rows.append(row)
        rows.sort(key=lambda row: int(row["clip_index"]))
        atomic_write_text(
            os.path.join(paths.cache_dir, "_stage", "metrics.jsonl"),
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        )
        summary = {
            "status": "completed", "fingerprint": replay_fingerprint,
            **_summarize_rows(rows), "min_eligible_fraction": min_eligible_fraction,
            "generation": generation,
        }
        stage.complete(summary)
        return summary
    except Exception as exc:
        stage.fail(repr(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True, help="冻结 synthetic checkpoint 或 adapter.safetensors")
    parser.add_argument("--drift-thresh", type=float, default=1.0)
    parser.add_argument("--min-eligible-fraction", type=float, default=0.70)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    mine(cfg, args.adapter, args.drift_thresh, args.max_samples, args.min_eligible_fraction)


if __name__ == "__main__":
    main()
