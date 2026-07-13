"""P2-V2 的固定 batch/noise 梯度审计；不执行参数更新或 rollout 训练。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from statistics import median
from typing import Any

import numpy as np
import torch

from ..backbones import Conditioning, build_backbone
from ..backbones.svd_backbone import classify_attention_module_path
from ..cache.dataset import ProjectionCacheDataset, cache_collate
from ..config import config_fingerprint, load_config, save_resolved_config
from ..losses import correction_v_loss, outside_mask_preserve_v_loss, real_loss, teacher_relative_v_target
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.fingerprint import directory_manifest_fingerprint, environment_fingerprint, git_state


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return mask.to(device=like.device, dtype=like.dtype).unsqueeze(2) if mask.dim() == like.dim() - 1 else mask.to(device=like.device, dtype=like.dtype)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _mask(mask, pred).expand_as(pred)
    return ((pred - target.detach()).square() * m).sum() / m.sum().clamp_min(1.0)


def _gradient_vector(loss: torch.Tensor, named_parameters: list[tuple[str, torch.nn.Parameter]], *, retain_graph: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    grads = torch.autograd.grad(loss, [p for _, p in named_parameters], retain_graph=retain_graph, allow_unused=True)
    pieces = []
    by_scope: dict[str, list[torch.Tensor]] = defaultdict(list)
    for (name, parameter), grad in zip(named_parameters, grads):
        value = torch.zeros_like(parameter, dtype=torch.float32) if grad is None else grad.detach().float()
        pieces.append(value.reshape(-1))
        module = name.split(".lora_", 1)[0]
        kind = classify_attention_module_path(module)
        if kind is not None:
            by_scope[kind].append(value.reshape(-1))
    vector = torch.cat(pieces)
    def stats(values: list[torch.Tensor]) -> dict[str, float]:
        flat = torch.cat(values) if values else torch.zeros(1, device=vector.device)
        return {"l2": float(torch.linalg.vector_norm(flat)), "rms": float(flat.square().mean().sqrt())}
    return vector, {"l2": float(torch.linalg.vector_norm(vector)), "rms": float(vector.square().mean().sqrt()), "temporal": stats(by_scope["temporal"]), "spatial": stats(by_scope["spatial"])}


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(1.0e-12))


def _write_csv(path: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Conditioning]:
    context = {key: value.to(device) for key, value in batch["context"].items()}
    return (batch["clean"].to(device), batch["y"].to(device), batch["x_dagger"].to(device), batch["mask"].to(device), Conditioning(context))


def run(cfg) -> dict[str, Any]:
    dcfg = cfg.diagnostics
    output_dir = str(dcfg.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError(f"gradient audit run 目录不可复用: {output_dir}")
    os.makedirs(output_dir)
    _seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    dataset = ProjectionCacheDataset(str(dcfg.cache_dir), expected_fingerprint=dcfg.get("cache_fingerprint"))
    backbone = build_backbone(cfg.model, load=True, device=str(device))
    student_adapter = dcfg.get("student_adapter")
    if student_adapter:
        backbone.load_adapter(str(student_adapter))
    backbone.set_train_mode(True)
    params = [(name, param) for name, param in backbone.training_module().named_parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("gradient audit 要求存在可训练 LoRA 参数")
    generator = torch.Generator(device=device).manual_seed(int(dcfg.noise_seed))
    sigma_values = [float(value) for value in dcfg.sigmas]
    rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    for batch_index in range(int(dcfg.num_batches)):
        batch = cache_collate([dataset[batch_index % len(dataset)] for _ in range(int(dcfg.batch_size))])
        clean, base, projected, legacy_mask, cond = _to_device(batch, device)
        static_mask = legacy_mask
        object_mask = torch.zeros_like(legacy_mask)
        for sigma_value in sigma_values:
            sigma = torch.full((base.shape[0],), sigma_value, device=device, dtype=base.dtype)
            noise = torch.randn(base.shape, generator=generator, device=device, dtype=base.dtype)
            z = backbone.add_noise(base, sigma, noise)
            student_v = backbone.predict_model_output(z, sigma, cond)
            target_info = teacher_relative_v_target(
                backbone, z, sigma, cond, base, projected, static_mask, object_mask,
                eta=float(dcfg.eta), trust_region_B=float(dcfg.trust_region_B),
            )
            x0_student = backbone.x0_from_model_output(z, sigma, student_v)
            losses = {
                "real": real_loss(backbone, clean, cond, use_edm_weight=False, sigma=sigma, noise=noise)["loss"],
                "x0_proj": _masked_mse(x0_student, projected, legacy_mask),
                "direct_v": correction_v_loss(student_v, target_info["target"], static_mask, object_mask)["loss"],
                "anchor": (x0_student - backbone.x0_from_model_output(z, sigma, target_info["teacher"])).square().mean(),
                "preserve": outside_mask_preserve_v_loss(student_v, target_info["teacher"], target_info["union_mask"], dilation_radius=int(dcfg.dilation_radius))["loss"],
            }
            vectors: dict[str, torch.Tensor] = {}
            stats: dict[str, dict[str, Any]] = {}
            names = list(losses)
            for offset, name in enumerate(names):
                vectors[name], stats[name] = _gradient_vector(losses[name], params, retain_graph=offset < len(names) - 1)
            cosines = {f"{left}__{right}": _cosine(vectors[left], vectors[right]) for index, left in enumerate(names) for right in names[index + 1:]}
            coverage = float(_mask(static_mask, base).mean())
            row = {
                "batch_index": batch_index,
                "sample_ids": [meta.get("sample_id") for meta in batch["metadata"]],
                "sigma": sigma_value,
                "input_source": str(dcfg.input_source),
                "mask_coverage": coverage,
                "static_mask_coverage": coverage,
                "object_mask_coverage": 0.0,
                "correction_rms": [float(value) for value in target_info["correction_rms"]],
                "trust_region_clipping_fraction": float(target_info["trust_region_clipping_fraction"]),
                "eta_eff": [float(value) for value in target_info["eta_eff"]],
                "loss": {key: float(value.detach()) for key, value in losses.items()},
                "gradient": stats,
                "cosine": cosines,
            }
            rows.append(row)
            for name, value in stats.items():
                norm_rows.append({"batch_index": batch_index, "sigma": sigma_value, "loss": name, "l2": value["l2"], "rms": value["rms"], "temporal_l2": value["temporal"]["l2"], "temporal_rms": value["temporal"]["rms"], "spatial_l2": value["spatial"]["l2"], "spatial_rms": value["spatial"]["rms"]})
            for key, value in cosines.items():
                cosine_rows.append({"batch_index": batch_index, "sigma": sigma_value, "pair": key, "cosine": value})
            del student_v, target_info, x0_student, losses, vectors
            torch.cuda.empty_cache()
    weighted_ratio = [row["gradient"]["x0_proj"]["l2"] / max(row["gradient"]["real"]["l2"], 1.0e-12) for row in rows]
    direct_ratio = [row["gradient"]["direct_v"]["l2"] / max(row["gradient"]["real"]["l2"], 1.0e-12) for row in rows]
    anchor_conflicts = [row["cosine"]["direct_v__anchor"] < -0.3 and row["gradient"]["anchor"]["l2"] / max(row["gradient"]["direct_v"]["l2"], 1.0e-12) >= 0.5 for row in rows]
    summary = {
        "status": "completed",
        "task_id": "P2-V2-GRAD-02",
        "git": git_state(),
        "config_fingerprint": config_fingerprint(cfg),
        "cache_fingerprint": directory_manifest_fingerprint(str(dcfg.cache_dir)),
        "input_source": str(dcfg.input_source),
        "student_adapter": None if not student_adapter else str(student_adapter),
        "input_limitations": "legacy synthetic schema-v4 cache；仅用于参数化/梯度工程诊断，不能作为 Base replay 或 rollout 收益证据。",
        "num_rows": len(rows),
        "sigmas": sigma_values,
        "selected_modules": backbone.adapter_metadata(),
        "median_x0_proj_to_real_l2": median(weighted_ratio),
        "median_direct_v_to_real_l2": median(direct_ratio),
        "anchor_conflict_fraction": sum(anchor_conflicts) / len(anchor_conflicts),
        "finite": all(math.isfinite(row["gradient"][name]["l2"]) for row in rows for name in row["gradient"]),
    }
    atomic_write_text(os.path.join(output_dir, "gradient_audit.jsonl"), "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    atomic_write_json(os.path.join(output_dir, "gradient_audit_summary.json"), summary)
    _write_csv(os.path.join(output_dir, "gradient_norm_by_sigma.csv"), norm_rows, ["batch_index", "sigma", "loss", "l2", "rms", "temporal_l2", "temporal_rms", "spatial_l2", "spatial_rms"])
    _write_csv(os.path.join(output_dir, "gradient_cosine_matrix.csv"), cosine_rows, ["batch_index", "sigma", "pair", "cosine"])
    atomic_write_text(os.path.join(output_dir, "selected_modules.txt"), "".join(f"{name}\n" for name in backbone.adapter_metadata()["selected_module_names"]))
    save_resolved_config(cfg, os.path.join(output_dir, "resolved.yaml"))
    atomic_write_json(os.path.join(output_dir, "manifest.json"), {"status": "completed", "task_id": "P2-V2-GRAD-02", "git": git_state(), "environment": environment_fingerprint(), "config_fingerprint": config_fingerprint(cfg), "cache_fingerprint": directory_manifest_fingerprint(str(dcfg.cache_dir)), "input_source": str(dcfg.input_source), "student_adapter": None if not student_adapter else str(student_adapter), "seed": int(cfg.seed)})
    atomic_write_text(os.path.join(output_dir, "COMPLETE"), "completed\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config, args.overrides)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
