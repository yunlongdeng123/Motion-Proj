"""P2-V2 固定 noise bank 的 8-pair 单步容量测试。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from ..backbones import Conditioning, build_backbone
from ..cache.dataset import ProjectionCacheDataset
from ..config import config_fingerprint, get_paths, load_config, save_resolved_config
from ..losses import correction_v_loss, outside_mask_preserve_v_loss, teacher_relative_v_target
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import JsonlMetrics, RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json
from .trainer import seed_everything


VARIANTS = {"A", "B", "C", "D", "E"}


def select_pair_indices(total: int, *, pair_count: int, train_pair_count: int, seed: int) -> dict[str, list[int]]:
    """固定抽样并分离 capacity 与 held-out pair，禁止按训练结果挑样本。"""
    if not 0 < train_pair_count < pair_count <= total:
        raise ValueError("pair_count/train_pair_count 与有效 V5 sample 数不兼容")
    selected = sorted(random.Random(seed).sample(range(total), pair_count))
    return {"all": selected, "train": selected[:train_pair_count], "held_out": selected[train_pair_count:]}


def capacity_decision(
    metrics: dict,
    *,
    required_error_reduction: float,
    max_outside_teacher_drift_ratio: float,
    max_target_roundtrip_relative_error: float = 2.0e-3,
) -> dict:
    initial = float(metrics["initial_target_error"])
    final = float(metrics["final_target_error"])
    reduction = 1.0 - final / max(initial, 1.0e-12)
    checks = {
        "target_error_reduction": reduction >= required_error_reduction,
        "outside_teacher_drift": float(metrics["outside_teacher_drift_ratio"]) <= max_outside_teacher_drift_ratio,
        "frame0_teacher_drift": abs(float(metrics["frame0_teacher_drift"])) <= 1.0e-6,
        "gradient_finite": bool(metrics["gradient_finite"]),
        "gradient_nonzero": bool(metrics["gradient_nonzero"]),
        "target_roundtrip": (
            float(metrics["target_roundtrip_max_relative_error"])
            <= max_target_roundtrip_relative_error
        ),
        "correction_direction": float(metrics["correction_direction_cosine"]) > 0.0,
    }
    return {"passed": all(checks.values()), "target_error_reduction": reduction, "checks": checks}


def _conditioning(item: dict, device: torch.device) -> Conditioning:
    context = item.get("context")
    if context is None:
        raise RuntimeError("V5 capacity pilot 要求保存的 conditioning context")
    return Conditioning({key: value.unsqueeze(0).to(device) for key, value in context.items()})


def _to_batch(item: dict, bank: dict, device: torch.device) -> dict:
    required = ("base_latent", "projected_latent", "static_mask", "object_mask", "latent_residual")
    missing = [key for key in required if key not in item]
    if missing:
        raise RuntimeError(f"V5 capacity pilot 缺少 tensor: {', '.join(missing)}")
    meta = item["metadata"]
    if (meta.get("source") != "replay_v2" or meta.get("parent_kind") != "base"
            or bool(meta.get("adapter_loaded")) or bool(meta.get("uses_future_gt_ego"))
            or bool(meta.get("uses_future_gt_track"))):
        raise RuntimeError("capacity pilot 只接受无 adapter、无 future-GT 的 Base V5 pair")
    base = item["base_latent"].unsqueeze(0).to(device)
    projected = item["projected_latent"].unsqueeze(0).to(device)
    residual = item["latent_residual"].unsqueeze(0).to(device)
    if not torch.allclose(projected - base, residual, atol=1.0e-6, rtol=1.0e-5):
        raise RuntimeError("V5 latent_residual 与 projected-base 不一致")
    static = item["static_mask"].unsqueeze(0).to(device)
    obj = item["object_mask"].unsqueeze(0).to(device)
    if bool(static.any()) or not bool(obj.any()) or bool(obj[:, 0].any()):
        raise RuntimeError("object-only pilot 要求 static=0、object 非空且 frame 0 冻结")
    return {
        "base": base, "projected": projected, "residual": residual,
        "static": static, "object": obj, "sigma": bank["sigma"].to(device),
        # noise bank 按单 pair 保存 [T,C,H,W]，进入 SVD 前必须恢复 batch 维。
        "noise": bank["noise"].unsqueeze(0).to(device), "z": bank["z_sigma"].unsqueeze(0).to(device),
        "condition": _conditioning(item, device), "sample_id": meta["sample_id"],
    }


def _masked_mse(error: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | float = 1.0) -> torch.Tensor:
    expanded = mask.to(error).expand_as(error)
    weighted = expanded * torch.as_tensor(weight, dtype=error.dtype, device=error.device)
    return (error.square() * weighted).sum() / weighted.sum().clamp_min(1.0)


def _model_output_roundtrip_error(backbone, z: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> dict:
    """以 float32 审计代数回环，并同时报告绝对与相对误差。"""
    z_fp32 = z.float()
    sigma_fp32 = sigma.float()
    target_fp32 = target.float()
    x0 = backbone.x0_from_model_output(z_fp32, sigma_fp32, target_fp32)
    roundtrip = backbone.model_output_from_x0(z_fp32, sigma_fp32, x0)
    absolute = (roundtrip - target_fp32).abs().max()
    relative = absolute / target_fp32.abs().max().clamp_min(1.0e-12)
    return {"absolute": absolute, "relative": relative}


def _variant_loss(backbone, batch: dict, variant: str, pilot_cfg) -> tuple[torch.Tensor, dict]:
    sigma, z, cond = batch["sigma"], batch["z"], batch["condition"]
    student = backbone.predict_model_output(z, sigma, cond)
    with torch.no_grad():
        teacher = backbone.anchor_predict_model_output(z, sigma, cond)
    obj, static = batch["object"], batch["static"]
    if variant in {"A", "B"}:
        student_x0 = backbone.x0_from_model_output(z, sigma, student)
        weight = 1.0 if variant == "A" else sigma.reciprocal().square().view(-1, 1, 1, 1, 1)
        correction = _masked_mse(student_x0 - batch["projected"], obj, weight)
        target = backbone.model_output_from_x0(z, sigma, batch["projected"])
    elif variant == "C":
        target = backbone.model_output_from_x0(z, sigma, batch["projected"])
        correction = correction_v_loss(student, target, static, obj)["loss"]
    else:
        trust = float(pilot_cfg.trust_region_B) if variant == "E" else 1.0e9
        target_info = teacher_relative_v_target(
            backbone, z, sigma, cond, batch["base"], batch["projected"], static, obj,
            eta=float(pilot_cfg.eta), trust_region_B=trust,
        )
        target = target_info["target"]
        correction = correction_v_loss(student, target, static, obj)["loss"]
    preserve = outside_mask_preserve_v_loss(
        student, teacher, (static + obj).clamp_max(1.0),
        weight=float(pilot_cfg.preserve_weight), dilation_radius=int(pilot_cfg.dilation_radius),
    )
    total = correction + preserve["loss"]
    roundtrip_error = _model_output_roundtrip_error(backbone, z, sigma, target)
    return total, {
        "target_error": correction.detach(), "preserve_loss": preserve["loss"].detach(),
        "student": student.detach(), "teacher": teacher.detach(), "target": target.detach(),
        "outside": preserve["outside_mask"].detach(),
        "roundtrip_error": roundtrip_error["absolute"].detach(),
        "roundtrip_relative_error": roundtrip_error["relative"].detach(),
    }


def _aggregate_evaluation_rows(values: list[dict]) -> dict:
    """聚合 pair 指标；严格门槛使用逐 pair 最坏值，避免均值掩盖失败。"""
    if not values:
        raise ValueError("capacity evaluation 至少需要一个 pair")
    mean_keys = (
        "target_error",
        "outside_teacher_drift_ratio",
        "correction_direction_cosine",
    )
    aggregate = {
        key: sum(float(row[key]) for row in values) / len(values)
        for key in mean_keys
    }
    aggregate.update({
        "outside_teacher_drift_ratio_max": max(float(row["outside_teacher_drift_ratio"]) for row in values),
        "frame0_teacher_drift": max(abs(float(row["frame0_teacher_drift"])) for row in values),
        "target_roundtrip_max_error": max(float(row["target_roundtrip_max_error"]) for row in values),
        "target_roundtrip_max_relative_error": max(
            float(row["target_roundtrip_max_relative_error"]) for row in values
        ),
    })
    return aggregate


def _evaluation(backbone, batches: list[dict], variant: str, pilot_cfg) -> dict:
    values = []
    with torch.no_grad():
        for batch in batches:
            _, info = _variant_loss(backbone, batch, variant, pilot_cfg)
            outside = info["outside"].expand_as(info["student"])
            teacher_rms = (info["teacher"].square() * outside).sum().div(outside.sum().clamp_min(1.0)).sqrt()
            drift = ((info["student"] - info["teacher"]).square() * outside).sum().div(outside.sum().clamp_min(1.0)).sqrt()
            frame0 = (info["student"][:, 0] - info["teacher"][:, 0]).abs().max()
            student_x0 = backbone.x0_from_model_output(batch["z"], batch["sigma"], info["student"])
            teacher_x0 = backbone.x0_from_model_output(batch["z"], batch["sigma"], info["teacher"])
            delta = batch["projected"] - batch["base"]
            direction = ((student_x0 - teacher_x0) * delta * batch["object"]).sum()
            denom = (((student_x0 - teacher_x0).square() * batch["object"]).sum().sqrt()
                     * ((delta.square() * batch["object"]).sum().sqrt())).clamp_min(1.0e-12)
            values.append({
                "sample_id": batch["sample_id"],
                "sigma": float(batch["sigma"].flatten()[0]),
                "target_error": float(info["target_error"]),
                "outside_teacher_drift_ratio": float(drift / teacher_rms.clamp_min(1.0e-12)),
                "frame0_teacher_drift": float(frame0),
                "target_roundtrip_max_error": float(info["roundtrip_error"]),
                "target_roundtrip_max_relative_error": float(info["roundtrip_relative_error"]),
                "correction_direction_cosine": float(direction / denom),
            })
    return {**_aggregate_evaluation_rows(values), "per_pair": values}


def _record_backbone_provenance(work_dir: Path, manifest_data: dict, backbone, cfg) -> None:
    """在首个 update 前固化 LoRA 模块清单与实际参数统计。"""
    adapter = backbone.adapter_metadata()
    if int(adapter["trainable_tensor_count"]) != int(adapter["adapter_tensor_count"]):
        raise RuntimeError("可训练 tensor 数与待保存 adapter tensor 数不一致")
    if int(adapter["selected_module_count"]) <= 0:
        raise RuntimeError("capacity pilot 的 selected module list 为空")
    module_names = [str(name) for name in adapter["selected_module_names"]]
    atomic_write_text(str(work_dir / "selected_modules.txt"), "".join(f"{name}\n" for name in module_names))
    manifest_data["model"] = {"name": str(cfg.model.name), "adapter": adapter}
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)


def _noise_bank(dataset: ProjectionCacheDataset, indices: dict[str, list[int]], pilot_cfg, destination: Path) -> dict:
    rows = []
    sigmas = list(pilot_cfg.sigma_by_train_pair)
    if len(sigmas) != len(indices["train"]):
        raise ValueError("sigma_by_train_pair 必须与 train_pair_count 一致")
    for offset, index in enumerate(indices["all"]):
        item = dataset[index]
        base = item["base_latent"].float()
        sigma_value = float(sigmas[offset]) if offset < len(sigmas) else float(sigmas[offset % len(sigmas)])
        generator = torch.Generator(device="cpu").manual_seed(int(pilot_cfg.noise_seed) + index)
        noise = torch.randn(base.shape, generator=generator, dtype=base.dtype)
        sigma = torch.tensor([sigma_value], dtype=base.dtype)
        rows.append({
            "dataset_index": index, "sample_id": item["metadata"]["sample_id"], "split": "train" if index in indices["train"] else "held_out",
            "sigma": sigma, "noise": noise, "z_sigma": base + sigma_value * noise,
        })
    payload = {"selection": indices, "noise_seed": int(pilot_cfg.noise_seed), "rows": rows}
    torch.save(payload, destination)
    return payload


def run_capacity_pilot(cfg, *, variants: list[str] | None = None, max_steps: int | None = None) -> dict:
    pilot_cfg = cfg.pilot
    requested = [str(value) for value in (variants or pilot_cfg.variants)]
    if not requested or any(value not in VARIANTS for value in requested):
        raise ValueError(f"variants 只能为 {sorted(VARIANTS)} 的非空子集")
    limit = int(max_steps if max_steps is not None else cfg.train.max_steps)
    if not 0 < limit <= 200:
        raise ValueError("P2-V2 capacity pilot 的 max_steps 必须在 [1,200]")
    git = git_state(".")
    if git.get("dirty"):
        raise RuntimeError("正式 capacity pilot 拒绝在 dirty worktree 上运行")
    work_dir = Path(cfg.work_dir)
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(f"run 目录不可复用: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=False)
    paths = get_paths(cfg)
    dataset = ProjectionCacheDataset(str(paths.cache_dir), expected_fingerprint=str(pilot_cfg.cache_fingerprint))
    selection = select_pair_indices(len(dataset), pair_count=int(pilot_cfg.pair_count),
                                    train_pair_count=int(pilot_cfg.train_pair_count), seed=int(pilot_cfg.selection_seed))
    bank_path = work_dir / "noise_bank.pt"
    bank = _noise_bank(dataset, selection, pilot_cfg, bank_path)
    config_fp = config_fingerprint(cfg)
    manifest = RunManifest(run_id=str(cfg.run_id), command=list(sys.argv), config_fingerprint=config_fp,
                           cache_fingerprint=str(pilot_cfg.cache_fingerprint), seed=int(cfg.seed), git=git,
                           environment=environment_fingerprint(), data_split=str(cfg.data.split))
    bank_fingerprint = file_fingerprint(str(bank_path))
    expected_bank_fingerprint = pilot_cfg.get("expected_noise_bank_fingerprint")
    manifest_data = manifest.__dict__ | {"task_id": str(pilot_cfg.task_id), "selection": selection,
                                          "noise_bank_fingerprint": bank_fingerprint, "variants": requested}
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    metrics = JsonlMetrics(str(work_dir / "metrics.jsonl"))
    results = {}
    try:
        if expected_bank_fingerprint and str(expected_bank_fingerprint) != bank_fingerprint:
            raise RuntimeError(
                "noise bank fingerprint 与预注册值不一致: "
                f"expected={expected_bank_fingerprint}, actual={bank_fingerprint}"
            )
        device = torch.device(cfg.device)
        train_batches = [_to_batch(dataset[index], bank["rows"][pos], device) for pos, index in enumerate(selection["train"])]
        for variant in requested:
            # 参数化对照必须共享相同初始化，不能让 variant 顺序改变 optimizer 起点。
            seed_everything(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
            backbone = build_backbone(cfg.model, load=True, device=str(device))
            backbone.set_train_mode(True)
            if "model" not in manifest_data:
                _record_backbone_provenance(work_dir, manifest_data, backbone, cfg)
            optimizer = torch.optim.AdamW(backbone.trainable_parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
            initial = _evaluation(backbone, train_batches, variant, pilot_cfg)
            gradient_finite = True
            gradient_nonzero = False
            for step in range(limit):
                batch = train_batches[step % len(train_batches)]
                optimizer.zero_grad(set_to_none=True)
                loss, loss_info = _variant_loss(backbone, batch, variant, pilot_cfg)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(backbone.trainable_parameters(), float(cfg.train.max_grad_norm))
                gradient_finite = gradient_finite and bool(torch.isfinite(norm))
                gradient_nonzero = gradient_nonzero or float(norm) > 0.0
                optimizer.step()
                if (step + 1) % int(cfg.train.log_every) == 0 or step + 1 == limit:
                    metrics.append(step + 1, {
                        "variant": variant,
                        "sample_id": batch["sample_id"],
                        "sigma": float(batch["sigma"].flatten()[0]),
                        "loss": float(loss),
                        "target_error": float(loss_info["target_error"]),
                        "preserve_loss": float(loss_info["preserve_loss"]),
                        "grad_norm": float(norm),
                    })
            final = _evaluation(backbone, train_batches, variant, pilot_cfg)
            variant_dir = work_dir / "variants" / variant
            variant_dir.mkdir(parents=True, exist_ok=False)
            checkpoint = variant_dir / "adapter.safetensors"
            backbone.save_adapter(str(checkpoint))
            decision = capacity_decision({
                "initial_target_error": initial["target_error"], "final_target_error": final["target_error"],
                **final, "gradient_finite": gradient_finite, "gradient_nonzero": gradient_nonzero,
            }, required_error_reduction=float(pilot_cfg.required_error_reduction),
                max_outside_teacher_drift_ratio=float(pilot_cfg.max_outside_teacher_drift_ratio),
                max_target_roundtrip_relative_error=float(pilot_cfg.max_target_roundtrip_relative_error))
            results[variant] = {"initial": initial, "final": final, "gradient_finite": gradient_finite,
                                "gradient_nonzero": gradient_nonzero, "checkpoint": str(checkpoint), "decision": decision}
            del optimizer, backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = {"status": "completed", "task_id": str(pilot_cfg.task_id), "variants": results,
                   "max_steps": limit, "selection": selection, "noise_bank_fingerprint": bank_fingerprint,
                   "experiment_fingerprint": sha256_json({"config": config_fp, "noise_bank": bank_fingerprint, "results": results})}
        atomic_write_json(str(work_dir / "summary.json"), summary)
        atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(summary) + "\n")
        manifest_data.update({
            "status": "completed",
            "ended_at": utc_now(),
            "exit_reason": "capacity_passed" if all(row["decision"]["passed"] for row in results.values()) else "capacity_failed",
        })
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        return summary
    except Exception as exc:
        atomic_write_json(str(work_dir / "summary.json"), {"status": "failed", "error": repr(exc)})
        manifest_data.update({"status": "failed", "ended_at": utc_now(), "exit_reason": repr(exc)})
        atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, list(args.overrides))
    print(json.dumps(run_capacity_pilot(cfg, variants=args.variants, max_steps=args.max_steps), ensure_ascii=False))


if __name__ == "__main__":
    main()
