"""单片段固定输入容量探针；不得用作正式训练或泛化评测。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import default_collate

from .model_bridge import (DEFAULT_EXTERNAL, DEFAULT_FCNET, DEFAULT_RAFT, DEFAULT_SVD,
                           load_components, load_trainable_state, trainable_state)
from .p1_capacity_probe import (CLIP_START, MAX_UPDATES, VIDEO_ID, measure_teacher,
                                prepare_teacher_cache, run_query, save_teacher_frames)
from .p1_data import YouTubeVOSP1Dataset
from .reference_propagation import propagate_reference_latents, static_fcnet_pair_masks
from .train import gradient_report
from .train_p1 import (DEFAULT_DATA, FORMAT, _official_util, paired_flow_loss,
                       validate_resume_protocol)
from .infer_p1 import _official_pipeline_class


PROBE_FORMAT = "worldsim_v81_fixed_input_capacity_probe_v1"
SOURCE_STEPS = (100, 500)


def validate_source(state: dict, profile: dict, batch: dict, *, data_root: Path,
                    updates: int) -> tuple[int, str]:
    """只允许显式列出的正式断点，探针断点不能反向进入正式训练。"""
    if not 1 <= updates <= MAX_UPDATES:
        raise ValueError(f"固定输入容量探针只允许1..{MAX_UPDATES}次更新")
    if state.get("format") != FORMAT or state.get("step") not in SOURCE_STEPS:
        raise ValueError("仅接受正式paper-bidirectional-m4的step100或step500断点")
    validate_resume_protocol(state, "paper-bidirectional-m4")
    if state.get("optimizer_name") != "Adam":
        raise ValueError("源断点必须使用Adam")
    groups = state.get("optimizer", {}).get("param_groups", [])
    if not groups or any(float(group["lr"]) != 1e-5 or
                         float(group["weight_decay"]) != 0 for group in groups):
        raise ValueError("源断点必须为lr=1e-5、weight_decay=0")
    saved_args = state.get("args", {})
    if (Path(saved_args.get("data_root", "/missing")).resolve() != data_root.resolve()
            or float(saved_args.get("lr", -1)) != 1e-5):
        raise ValueError("源断点的数据根或学习率不匹配")
    amp = saved_args.get("amp")
    if amp not in {"bf16", "fp16"}:
        raise ValueError("源断点精度须为bf16或fp16")
    saved_profile = state.get("data_profile", {})
    keys = ("rgb_root", "videos", "available_windows", "seed", "frames", "size",
            "mask_ratio_each_side")
    if any(saved_profile.get(key) != profile.get(key) for key in keys):
        raise ValueError("数据池、采样seed、视频形状或mask与源断点不一致")
    if batch["video_id"] != [VIDEO_ID] or int(batch["clip_start"][0]) != CLIP_START:
        raise ValueError("固定片段必须为0fc958cde2/start2")
    return int(state["step"]), amp


def fixed_train_step(components, batch: dict, cache: dict, optimizer, scaler,
                     *, amp: str) -> dict:
    """原P1损失与参数；仅把单片段随机输入冻结，不运行CFG dropout。"""
    device = next(components.unet.parameters()).device
    components.fcnet.train()
    components.propagator.train()
    components.unet.train()
    optimizer.zero_grad(set_to_none=True)
    plan = cache["plan"]
    mask = cache["mask"].to(device)
    flow_gt = tuple(flow.to(device) for flow in cache["flow"])
    pair_masks = static_fcnet_pair_masks(mask, plan)
    flow_pred, _ = checkpoint(components.fcnet.forward_bidirect_flow,
                              flow_gt, pair_masks, use_reentrant=False)
    completed = components.fcnet.combine_flow(flow_gt, flow_pred, pair_masks)
    _, _, condition = propagate_reference_latents(
        components.propagator, cache["cond_latent"].to(device),
        completed[0], completed[1], mask, plan)

    sigma = cache["target_latent"].new_tensor(cache["sigma"]).to(device).reshape(1, 1, 1, 1, 1)
    target = cache["target_latent"].to(device)
    noisy = target + sigma * cache["noise"].to(device)
    model_input = torch.cat((noisy / (1 + sigma.square()).sqrt(), condition), dim=2)
    cast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    with torch.autocast(device.type, dtype=cast_dtype, enabled=device.type == "cuda"):
        predicted = components.unet(model_input, 0.25 * sigma.log().flatten(),
                                    cache["clip"].to(device),
                                    added_time_ids=cache["time_ids"].to(device)).sample
    denoised = (predicted.float() * (-sigma / (1 + sigma.square()).sqrt())
                + noisy / (1 + sigma.square()))
    weight = (1 + sigma.square()) / sigma.square()
    diffusion_loss = (weight * (denoised - target).square()).flatten(1).mean(1).mean()
    flow_l1, ternary_warp = paired_flow_loss(
        components.flow_loss, completed, flow_gt, mask, batch["target_rgb"].to(device),
        list(plan.pairs), warp_chunk=8)
    loss = diffusion_loss + flow_l1 + ternary_warp
    if not torch.isfinite(loss):
        raise FloatingPointError("固定输入容量训练损失非有限值")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = {name: gradient_report(module) for name, module in
                 (("fcnet", components.fcnet), ("propagator", components.propagator),
                  ("svd_temporal", components.unet))}
    if any(row["status"] != "ok" for row in gradients.values()):
        raise RuntimeError(f"固定输入探针缺少有效梯度: {gradients}")
    none_grad = {name: [key for key, parameter in module.named_parameters()
                        if parameter.requires_grad and parameter.grad is None]
                 for name, module in (("fcnet", components.fcnet),
                                      ("propagator", components.propagator),
                                      ("svd_temporal", components.unet))}
    frozen_grad_tensors = sum(
        parameter.grad is not None
        for module in (components.vae, components.image_encoder,
                       components.raft, components.unet)
        for parameter in module.parameters() if not parameter.requires_grad)
    if frozen_grad_tensors:
        raise RuntimeError(f"固定参数出现梯度: {frozen_grad_tensors}")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer_updated = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
    return {"loss": float(loss.detach()), "diffusion": float(diffusion_loss.detach()),
            "flow_l1": float(flow_l1.detach()), "ternary_warp": float(ternary_warp.detach()),
            "gradients": gradients, "none_grad_names": none_grad,
            "frozen_grad_tensors": frozen_grad_tensors,
            "sigma": float(sigma), "cond_sigma": cache["cond_sigma"],
            "optimizer_updated": optimizer_updated}


def save_terminal_state(path: Path, components, optimizer, scheduler, scaler,
                        *, source_checkpoint: Path, source_step: int, updates: int,
                        update_attempts: int,
                        seed: int, input_frames: list[str], profile: dict) -> None:
    state = {"format": PROBE_FORMAT, "source_checkpoint": str(source_checkpoint.resolve()),
             "source_step": source_step, "probe_updates": updates,
             "probe_update_attempts": update_attempts,
             "formal_training_updates": 0, "seed": seed,
             "input_frames": input_frames, "data_profile": profile,
             "models": trainable_state(components), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-steps", type=int, default=25)
    parser.add_argument("--svd", type=Path, default=DEFAULT_SVD)
    parser.add_argument("--raft-weight", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--fcnet-weight", type=Path, default=DEFAULT_FCNET)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.query_steps < 1 or not 1 <= args.updates <= MAX_UPDATES:
        raise ValueError("query-steps须为正，固定输入探针更新限1..64")
    if not torch.cuda.is_available():
        raise RuntimeError("只由主任务明确启动GPU容量探针")
    repo = Path(__file__).resolve().parents[2]
    if args.output_dir.resolve().is_relative_to(repo) or args.output_dir.exists():
        raise ValueError("探针输出须为新的仓库外目录")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_profile = state.get("data_profile", {})
    dataset = YouTubeVOSP1Dataset(args.data_root, seed=int(saved_profile.get("seed", 123)))
    batch = default_collate([dataset[0]])
    source_step, amp = validate_source(state, dataset.profile(), batch,
                                       data_root=args.data_root, updates=args.updates)
    index, start = dataset.choice(0)
    input_frames = [str(path) for path in dataset.videos[index][1][start:start + 25]]
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    components = load_components(args.svd, args.raft_weight, args.fcnet_weight,
                                  args.external, device="cuda")
    load_trainable_state(components, state["models"])
    components.unet.enable_gradient_checkpointing()
    pipeline_class = _official_pipeline_class(args.external)
    pipeline = pipeline_class(vae=components.vae, image_encoder=components.image_encoder,
                              unet=components.unet, scheduler=components.scheduler,
                              feature_extractor=components.feature_extractor,
                              fix_raft=components.raft, vo_flow_complete=components.fcnet,
                              lat_bi_propagator=components.propagator)
    cache = prepare_teacher_cache(components, batch, seed=args.seed,
                                  util=_official_util(args.external))
    torch.save({"teacher_cache": cache, "target_rgb": batch["target_rgb"],
                "visible_rgb": batch["visible_rgb"], "input_frames": input_frames},
               args.output_dir / "fixed_inputs.pt")
    report = {"status": "running", "kind": "fixed_input_training_clip_capacity_only",
              "source_checkpoint": str(args.checkpoint.resolve()), "source_step": source_step,
              "probe_updates": args.updates, "formal_training_updates": 0,
              "video_id": VIDEO_ID, "clip_start": CLIP_START, "input_frames": input_frames,
              "seed": args.seed, "query_seed": args.seed + 10, "query_steps": args.query_steps,
              "optimizer": "Adam", "lr": 1e-5, "weight_decay": 0,
              "amp": amp,
              "teacher_sigma": cache["sigma"], "teacher_cond_sigma": cache["cond_sigma"],
              "teacher_role": "fixed noisy-GT BUILD; full-GT RAFT/CLIP; fps7; no CFG dropout",
              "query_role": "visible-only Gaussian QUERY; separate condition and fps6",
              "formal_eval": False, "generalization_claim": False, "human_verdict": None,
              "reference_indices": list(cache["plan"].refs),
              "flow_pairs": list(cache["plan"].pairs)}
    report_path = args.output_dir / "run.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    before, decoded = measure_teacher(components, cache, amp=amp)
    save_teacher_frames(pipeline, components, decoded, batch,
                        args.output_dir / "teacher_before")
    run_query(components, pipeline, batch, refs=cache["plan"].refs,
              seed=args.seed + 10, steps=args.query_steps, amp=amp,
              output_dir=args.output_dir / "query_before")
    for module in (components.vae, components.image_encoder, components.raft):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("固定输入探针中的观测编码器必须冻结")
        module.eval()
        module.to("cpu")
    for module in (components.fcnet, components.propagator, components.unet):
        module.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    parameters = [parameter for module in (components.unet, components.propagator,
                                           components.fcnet)
                  for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-5, weight_decay=0)
    optimizer.load_state_dict(state["optimizer"])
    from diffusers.optimization import get_scheduler
    scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=500,
                              num_training_steps=100_000)
    scheduler.load_state_dict(state["scheduler"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    scaler.load_state_dict(state["scaler"])
    rows = [{"update": 0, "teacher": before}]
    log = args.output_dir / "updates.jsonl"
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(rows[0]) + "\n")
        for update in range(1, args.updates + 1):
            train = fixed_train_step(components, batch, cache, optimizer, scaler, amp=amp)
            scheduler.step()
            row = {"update": update, "train": train}
            if update in {16, 32, 64, args.updates}:
                row["teacher"], decoded = measure_teacher(components, cache, amp=amp)
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({"event": "fixed_capacity_update", **row}), flush=True)
    save_terminal_state(args.output_dir / "probe_state_final.pt", components, optimizer,
                        scheduler, scaler, source_checkpoint=args.checkpoint,
                        source_step=source_step,
                        updates=sum(bool(row["train"]["optimizer_updated"]) for row in rows[1:]),
                        update_attempts=len(rows) - 1, seed=args.seed,
                        input_frames=input_frames, profile=dataset.profile())
    training_elapsed = time.perf_counter() - training_started
    training_peak_bytes = torch.cuda.max_memory_allocated()
    training_peak_reserved_bytes = torch.cuda.max_memory_reserved()
    save_teacher_frames(pipeline, components, decoded, batch,
                        args.output_dir / "teacher_after")
    run_query(components, pipeline, batch, refs=cache["plan"].refs,
              seed=args.seed + 10, steps=args.query_steps, amp=amp,
              output_dir=args.output_dir / "query_after")
    report.update(status="complete", teacher_before=before, teacher_after=rows[-1]["teacher"],
                  actual_updates=sum(bool(row["train"]["optimizer_updated"])
                                     for row in rows[1:]),
                  update_attempts=len(rows) - 1,
                  training_elapsed_seconds=training_elapsed,
                  total_elapsed_seconds=time.perf_counter() - started,
                  training_peak_cuda_allocated_bytes=training_peak_bytes,
                  training_peak_cuda_reserved_bytes=training_peak_reserved_bytes,
                  updates_log=str(log), fixed_inputs=str(args.output_dir / "fixed_inputs.pt"),
                  probe_state=str(args.output_dir / "probe_state_final.pt"))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
