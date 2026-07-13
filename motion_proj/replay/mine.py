"""从冻结 Base rollout 构建无 future-GT 的 V5 replay candidate。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

from ..auditor import MotionAuditor, RAFTChainGeneratedTrackProvider
from ..auditor.state import MotionState
from ..backbones import build_backbone
from ..cache.build_cache import _flow_to_resolution
from ..cache.writer import ProjectionCacheWriter
from ..config import cache_config_fingerprint, get_paths, load_config, save_resolved_config
from ..data import NuScenesFutureVideoDataset
from ..projector import DynamicsProjector
from ..projector.mask import downsample_mask_to_latent
from ..runtime.atomic import atomic_write_json, atomic_write_text
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


def _audit_generated(
    auditor: MotionAuditor,
    gen_frames: torch.Tensor,
    src_sample: dict,
    geometry_mode: str,
) -> MotionState:
    """按显式几何模式审计生成片段，并在正式模式移除未来 GT。"""
    sample = {
        "frames": gen_frames,
        "boxes": [[] for _ in range(gen_frames.shape[0])],
        "intrinsics": src_sample["intrinsics"],
        "cam2ego": src_sample["cam2ego"],
        "sample_id": src_sample["sample_id"] + "_gen",
    }
    if geometry_mode == "gt_ego_debug":
        sample["ego2global"] = src_sample["ego2global"]
    elif geometry_mode == "controlled_ego":
        if "control_ego2global" not in src_sample:
            raise ValueError("controlled_ego 要求 backbone 实际接收的 control_ego2global")
        sample["control_ego2global"] = src_sample["control_ego2global"]
    return auditor.audit(sample, generated_geometry_mode=geometry_mode)


def replay_energy_decreased(result, tolerance: float = 1e-6,
                            *, drift_before: float | None = None,
                            drift_after: float | None = None) -> bool:
    """优先用重审计后的 static drift；否则回退到投影总能量严格下降。"""
    if drift_before is not None and drift_after is not None:
        return float(drift_after) < float(drift_before) - tolerance
    before = result.energy_before if hasattr(result, "energy_before") else {}
    after = result.energy_after if hasattr(result, "energy_after") else {}
    if "total" in before and "total" in after:
        return float(after["total"]) < float(before["total"]) - tolerance
    return bool(result.diagnostics.get("energy_decreased", False))


def replay_is_eligible(drift: float, result, drift_thresh: float,
                       min_eligible_fraction: float = 0.70,
                       *, drift_after: float | None = None) -> bool:
    diagnostics = result.diagnostics
    return (
        drift >= drift_thresh
        and replay_energy_decreased(result, drift_before=drift, drift_after=drift_after)
        and float(diagnostics.get("eligible_fraction", 0.0)) >= min_eligible_fraction
    )


def _generation_settings(cfg) -> dict:
    return {
        "num_inference_steps": int(cfg.cache.get("num_inference_steps", 25)),
        "decode_chunk_size": int(cfg.cache.get("decode_chunk_size", 4)),
    }


def _generated_track_provider(cfg) -> RAFTChainGeneratedTrackProvider:
    options = dict(cfg.auditor.get("generated_tracks", {}))
    provider = str(options.pop("provider", "raft_chain"))
    if provider != "raft_chain":
        raise ValueError("replay mining 的训练 auditor 只允许 raft_chain；cotracker3 只能作为独立 evaluator")
    return RAFTChainGeneratedTrackProvider(device=cfg.device, **options)


def _base_model_fingerprints(pretrained: str) -> tuple[str, str]:
    """用模型配置文件绑定 Base/ VAE provenance，不把目录路径当作内容指纹。"""
    root = os.path.abspath(pretrained)
    required = {
        "unet": os.path.join(root, "unet", "config.json"),
        "vae": os.path.join(root, "vae", "config.json"),
        "scheduler": os.path.join(root, "scheduler", "scheduler_config.json"),
    }
    missing = [name for name, path in required.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Base 模型缺少 fingerprint 配置: {', '.join(missing)}")
    digests = {name: file_fingerprint(path) for name, path in required.items()}
    return sha256_json(digests), digests["vae"]


def _evenly_spaced_indices(total: int, requested: int) -> list[int]:
    """固定覆盖整个 split，避免只从低 index 或高 drift 区间挖样本。"""
    if requested <= 0 or requested > total:
        raise ValueError(f"max_conditions 必须在 [1,{total}] 内")
    if requested == 1:
        return [0]
    return [round(index * (total - 1) / (requested - 1)) for index in range(requested)]


def _component_energy(result, drift_before: float, drift_after: float) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {"static": float(drift_before), "object": float(result.energy_before.get("obj", 0.0))},
        {"static": float(drift_after), "object": float(result.energy_after.get("obj", 0.0))},
    )


def _formal_metadata(
    sample_id: str,
    source_sample_id: str,
    condition_index: int,
    generation_seed: int,
    generation: dict,
    base_model_fingerprint: str,
    vae_fingerprint: str,
    geometry_mode: str,
    result,
    energy_before: dict[str, float],
    energy_after: dict[str, float],
    static_mask: torch.Tensor,
    object_mask: torch.Tensor,
    static_confidence: torch.Tensor,
    object_confidence: torch.Tensor,
) -> dict:
    return {
        "sample_id": sample_id,
        "source": "replay_v2",
        "parent_kind": "base",
        "base_model_fingerprint": base_model_fingerprint,
        "adapter_loaded": False,
        "condition_id": source_sample_id,
        "condition_frame": int(condition_index),
        "generation_seed": int(generation_seed),
        "generation_sampler": "torch.Generator",
        "generation_steps": int(generation["num_inference_steps"]),
        "generation_settings": generation,
        "first_frame_frozen": True,
        "auditor_version": "generated-point-track-v1",
        "projector_version": "dynamics-projector-v5",
        "geometry_mode": geometry_mode,
        "uses_future_gt_ego": False,
        "uses_future_gt_track": False,
        "energy_before_by_component": energy_before,
        "energy_after_by_component": energy_after,
        "projector_diagnostics": result.diagnostics,
        "base_vae_fingerprint": vae_fingerprint,
        "projected_vae_fingerprint": vae_fingerprint,
        "vae_fingerprint": vae_fingerprint,
        "static_valid_fraction": float(static_mask.gt(0).float().mean()),
        "object_valid_fraction": float(object_mask.gt(0).float().mean()),
        "static_confidence_mean": float(static_confidence.mean()),
        "object_confidence_mean": float(object_confidence.mean()),
    }


@torch.no_grad()
def mine_base(cfg, max_conditions: int, generation_seeds: list[int]) -> dict:
    """生成 Base-parent V5 candidate；质量拒绝由 writer 原子执行。"""
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 Base replay mining 拒绝在 dirty worktree 上运行")
    if bool(cfg.model.lora.enable):
        raise ValueError("正式 Base replay 必须 model.lora.enable=false")
    if str(cfg.auditor.generated_geometry_mode) != "estimated_background_motion":
        raise ValueError("正式 Base replay 必须使用 estimated_background_motion")
    if not generation_seeds or len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError("generation_seeds 必须是非空且互异的整数列表")
    paths = get_paths(cfg)
    dataset = NuScenesFutureVideoDataset(cfg.data)
    indices = _evenly_spaced_indices(len(dataset), int(max_conditions))
    generation = _generation_settings(cfg)
    base_fingerprint, vae_fingerprint = _base_model_fingerprints(str(cfg.model.pretrained))
    replay_fingerprint = sha256_json({
        "schema": "replay-v5-base", "cache": cache_config_fingerprint(cfg),
        "base_model_fingerprint": base_fingerprint, "conditions": indices,
        "generation_seeds": generation_seeds, "generation": generation,
        "geometry_mode": str(cfg.auditor.generated_geometry_mode),
    })
    writer = ProjectionCacheWriter(
        paths.cache_dir, store=cfg.cache.store, overwrite=False,
        fingerprint=replay_fingerprint, formal_v2=True,
    )
    stage = StageManifest(os.path.join(paths.cache_dir, "_stage"), "replay_v2", replay_fingerprint)
    if stage.is_complete():
        return {"status": "completed", "fingerprint": replay_fingerprint, "skipped": True}
    stage.begin({
        "source": "replay_v2", "parent_kind": "base", "adapter_loaded": False,
        "base_model_fingerprint": base_fingerprint, "generation_seeds": generation_seeds,
        "condition_indices": indices, "generation": generation, "git": git,
        "command": list(sys.argv),
    })
    save_resolved_config(cfg, os.path.join(paths.cache_dir, "_stage", "resolved.yaml"))
    backbone = build_backbone(cfg.model, load=True, device=cfg.device)
    backbone.set_train_mode(False)
    auditor = MotionAuditor(
        device=cfg.device, generated_geometry_mode=str(cfg.auditor.generated_geometry_mode),
        background_fit_options=dict(cfg.auditor.get("background_fit", {})),
        generated_track_provider=_generated_track_provider(cfg),
    )
    projector = DynamicsProjector()
    rows: list[dict] = []
    try:
        for condition_index in tqdm(indices):
            source = dataset[condition_index]
            for generation_seed in generation_seeds:
                sample_id = f"{source['sample_id']}_base_s{generation_seed}"
                row = {
                    "condition_index": condition_index, "sample_id": sample_id,
                    "generation_seed": generation_seed, "kept": False, "reject_reason": None,
                }
                if writer.exists(sample_id):
                    row["kept"] = True
                    rows.append(row)
                    continue
                generator = torch.Generator(device=cfg.device).manual_seed(int(generation_seed))
                base = backbone.generation(
                    source["cond_frame"].to(cfg.device), num_frames=int(cfg.data.num_frames),
                    num_inference_steps=generation["num_inference_steps"], generator=generator,
                    height=int(cfg.data.height), width=int(cfg.data.width),
                    decode_chunk_size=generation["decode_chunk_size"],
                )
                state = _audit_generated(auditor, base, source, str(cfg.auditor.generated_geometry_mode))
                result = projector.project(base, state)
                # static 分支尚未通过人工门禁；V5 仅保留 object 局部修正，绝不把
                # 自估背景 warp 写入可训练 target。
                object_mask = result.object_mask
                if object_mask is None:
                    raise RuntimeError("projector 未返回 V5 component mask")
                static_mask = torch.zeros_like(object_mask)
                object_mask = object_mask.clone()
                object_mask[0] = 0
                projected = base + object_mask * (result.target - base)
                projected[0] = base[0]
                combined_mask = object_mask
                projected_state = _audit_generated(auditor, projected, source, str(cfg.auditor.generated_geometry_mode))
                before, after = _component_energy(
                    result, auditor.static_drift_score(state), auditor.static_drift_score(projected_state),
                )
                static_confidence = static_mask
                object_confidence = object_mask
                if cfg.cache.store == "latent":
                    scale = int(cfg.model.vae_scale_factor)
                    stored_static_mask = downsample_mask_to_latent(static_mask, scale)
                    stored_object_mask = downsample_mask_to_latent(object_mask, scale)
                    stored_static_confidence = downsample_mask_to_latent(static_confidence, scale)
                    stored_object_confidence = downsample_mask_to_latent(object_confidence, scale)
                else:
                    stored_static_mask = static_mask
                    stored_object_mask = object_mask
                    stored_static_confidence = static_confidence
                    stored_object_confidence = object_confidence
                metadata = _formal_metadata(
                    sample_id, source["sample_id"], condition_index, generation_seed, generation,
                    base_fingerprint, vae_fingerprint, str(cfg.auditor.generated_geometry_mode),
                    result, before, after, stored_static_mask, stored_object_mask,
                    stored_static_confidence, stored_object_confidence,
                )
                condition = backbone.build_conditioning({"cond_frame": source["cond_frame"].unsqueeze(0)})
                base_latent = backbone.encode(base.unsqueeze(0))[0]
                projected_latent = backbone.encode(projected.unsqueeze(0))[0]
                latent_residual = projected_latent - base_latent
                try:
                    if cfg.cache.store == "rgb":
                        writer.write(
                            sample_id, base.cpu(), projected.cpu(), combined_mask.cpu(), metadata,
                            static_mask=stored_static_mask.cpu(), object_mask=stored_object_mask.cpu(),
                            static_confidence=stored_static_confidence.cpu(), object_confidence=stored_object_confidence.cpu(),
                            base_rgb=base.cpu(), projected_rgb=projected.cpu(), source="replay_v2",
                            generation_seed=generation_seed, source_fingerprint=base_fingerprint,
                            base_latent=base_latent.cpu(), projected_latent=projected_latent.cpu(),
                            latent_residual=latent_residual.cpu(),
                        )
                    else:
                        context = {key: condition.data[key][0] for key in
                                   ("image_embeds", "image_latents", "added_time_ids")}
                        writer.write(
                            sample_id, base_latent.cpu(), projected_latent.cpu(),
                            stored_object_mask.cpu(), metadata, context,
                            clean=base_latent.cpu(), static_mask=stored_static_mask.cpu(),
                            object_mask=stored_object_mask.cpu(),
                            static_confidence=stored_static_confidence.cpu(),
                            object_confidence=stored_object_confidence.cpu(),
                            base_rgb=base.cpu(), projected_rgb=projected.cpu(), source="replay_v2",
                            generation_seed=generation_seed, source_fingerprint=base_fingerprint,
                            base_latent=base_latent.cpu(), projected_latent=projected_latent.cpu(),
                            latent_residual=latent_residual.cpu(),
                        )
                except ValueError as exc:
                    row["reject_reason"] = str(exc)
                    rows.append(row)
                    continue
                row.update({"kept": True, "energy_before_by_component": before,
                            "energy_after_by_component": after,
                            "static_coverage": float(static_mask.mean()),
                            "object_coverage": float(object_mask.mean())})
                rows.append(row)
        atomic_write_text(
            os.path.join(paths.cache_dir, "_stage", "metrics.jsonl"),
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        )
        summary = {
            "status": "completed", "task_id": "P2-V2-REPLAY-05", "fingerprint": replay_fingerprint,
            "candidates": len(rows), "kept": sum(bool(row["kept"]) for row in rows),
            "rejected": sum(not bool(row["kept"]) for row in rows),
            "condition_indices": indices, "generation_seeds": generation_seeds,
        }
        atomic_write_json(os.path.join(paths.cache_dir, "_stage", "summary.json"), summary)
        stage.complete(summary)
        return summary
    except Exception as exc:
        stage.fail(repr(exc))
        raise


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
    geometry_mode = str(cfg.auditor.generated_geometry_mode)
    track_options = dict(cfg.auditor.get("generated_tracks", {}))
    auditor = MotionAuditor(
        device=device,
        generated_geometry_mode=geometry_mode,
        background_fit_options=dict(cfg.auditor.get("background_fit", {})),
        generated_track_provider=_generated_track_provider(cfg),
    )
    projector = DynamicsProjector()
    n = min(max_samples, len(ds)) if max_samples else len(ds)
    generation = _generation_settings(cfg)
    replay_fingerprint = sha256_json({
        "cache": cache_config_fingerprint(cfg), "parent": parent_fingerprint,
        "seed": int(cfg.seed), "drift_thresh": drift_thresh,
        "min_eligible_fraction": min_eligible_fraction, "samples": n,
        "generation": generation,
        "generated_tracks": track_options,
        "energy_gate": "reaudit-static-drift-v1",
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
            state = _audit_generated(auditor, gen, src, geometry_mode)
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
            # 生成帧无 GT track 时 projector 总能量不反映 static 修复；对 x_dagger 重审计。
            state_after = _audit_generated(auditor, result.target, src, geometry_mode)
            drift_after = auditor.static_drift_score(state_after)
            row["energy_decreased"] = replay_energy_decreased(
                result, drift_before=drift, drift_after=drift_after
            )
            row["eligible_fraction"] = float(result.diagnostics.get("eligible_fraction", 0.0))
            row["energy_before"] = drift
            row["energy_after"] = drift_after
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
    parser.add_argument("--max-conditions", type=int, required=True)
    parser.add_argument("--generation-seeds", required=True, help="逗号分隔的 Base rollout seed")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    seeds = [int(value) for value in args.generation_seeds.split(",") if value]
    mine_base(cfg, args.max_conditions, seeds)


if __name__ == "__main__":
    main()
