"""One-case native DriveEditor main U-Net fine-tuning; no LoRA or module fallback."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from finetune_scope import selected


REPO = Path("/root/autodl-tmp/motion_proj_v77")
RUNS = Path("/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929")
ENCODER = RUNS / "r7/encoder_recovery/official_svd_encoder.safetensors"
SCOPE_COUNT = 1_809_579_626
KINDS = {"real_rgb_synthetic_occlusion", "reviewed_pseudo_clean_target"}
MIN_FREE = 18 * 2**30


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_pair(path: Path, size=(320, 576)):
    pair = json.loads(path.read_text(encoding="utf-8"))
    if pair.get("source_case") != "R001" or pair.get("supervision_kind") not in KINDS:
        raise ValueError("Only R001 and the two registered supervision kinds are allowed")
    num_frames = pair.get("num_frames")
    if num_frames not in (1, 10):
        raise ValueError("pair.num_frames must be 1 or 10; normal DELETE training uses 10")
    if pair.get("frame_index") != 5:
        raise ValueError("R001 review anchor must be frame_index: 5")
    if pair.get("qa_pass") is not True or not pair.get("qa_record"):
        raise ValueError("An explicit passed QA record is required")
    record = Path(pair["qa_record"])
    if not record.is_absolute() or not record.is_file():
        raise ValueError("qa_record must be an existing absolute path")
    paths = {}
    for role in ("x", "y", "hole"):
        p = Path(pair[role])
        if not p.is_absolute():
            raise ValueError(f"{role} must be an absolute path")
        if num_frames == 1:
            if p.suffix.lower() != ".png" or not p.is_file():
                raise ValueError(f"{role} must be one existing PNG")
            paths[role] = [p]
        else:
            expected = [p / f"{i:05d}.png" for i in range(10)]
            if not p.is_dir() or any(not frame.is_file() for frame in expected):
                raise ValueError(f"{role} must contain 00000.png through 00009.png")
            if sorted(p.glob("*.png")) != expected:
                raise ValueError(f"{role} has extra or missing PNG frames")
            paths[role] = expected
    def frames(role, mode):
        return np.stack([np.asarray(Image.open(frame).convert(mode)) for frame in paths[role]])
    x, y, raw_hole = frames("x", "RGB"), frames("y", "RGB"), frames("hole", "L")
    if x.shape != y.shape or x.shape[:3] != raw_hole.shape or x.dtype != np.uint8:
        raise ValueError("X, Y and hole shapes/dtypes do not match")
    if not set(np.unique(raw_hole)).issubset({0, 255}):
        raise ValueError("hole PNG must be binary 0/255")
    hole = raw_hole > 0
    if not hole.any() or np.any((x != y).any(axis=-1) & ~hole):
        raise ValueError("hole is empty or X/Y differ outside the hole")
    # r7 clears the hole before resizing; review frame 5 is not the training extent.
    sys.path.insert(0, str(REPO / "scripts/worldsim_v77/target_protected/iteration7"))
    import train_control as train
    prepared = train.resized(y, x, hole, size)
    if any(t.shape[0] != num_frames for t in prepared) or prepared[0].shape != (num_frames, 3, *size):
        raise RuntimeError("Video size contract failed")
    return pair, prepared, train


def init_model(train):
    train.ENCODER = ENCODER
    train.MODULES = "full_main"
    train.selected = selected
    model, params, names = train.model_init()  # strict offline encoder restore, BF16 frozen / FP32 trainables
    if hasattr(model.model.diffusion_model, "multi_prior_branch"):
        raise RuntimeError("r47 branch must not participate in native main training")
    if len(names) != 1647 or sum(p.numel() for p in params) != SCOPE_COUNT:
        raise RuntimeError("Actual trainable scope differs from official checkpoint header census")
    if any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("Trainable native weights lost original FP32 precision")
    return model, params, names


def restore(model, optimizer, names, out: Path, resume: dict):
    weights = load_file(str(out / resume["weights"]), device="cpu")
    if set(weights) != set(names):
        raise RuntimeError("Resume checkpoint has a different native parameter scope")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(weights[name].to(parameters[name].device))
    del weights
    state = torch.load(out / resume["optimizer"], map_location="cpu", weights_only=False)
    if state["step"] != resume["step"]:
        raise RuntimeError("Optimizer and parameter checkpoint steps disagree")
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])


def snapshot(out: Path, step: int, model, names, optimizer, previous: dict | None):
    if shutil.disk_usage(out).free < 12 * 2**30:
        raise RuntimeError("Less than 12 GiB free before FP32 checkpoint + optimizer snapshot")
    stem = f"{step:04d}"
    weight_name, optimizer_name = f"main_{stem}.safetensors", f"optimizer_{stem}.pt"
    parameters = dict(model.named_parameters())
    if any(not torch.isfinite(parameters[name]).all().item() for name in names):
        raise FloatingPointError(f"Non-finite trainable parameter before step {step} snapshot")
    weight_tmp = out / (weight_name + ".tmp")
    save_file({name: parameters[name].detach().cpu().contiguous() for name in names}, str(weight_tmp))
    os.replace(weight_tmp, out / weight_name)
    optimizer_tmp = out / (optimizer_name + ".tmp")
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, optimizer_tmp)
    os.replace(optimizer_tmp, out / optimizer_name)
    pointer = {"step": step, "weights": weight_name, "optimizer": optimizer_name}
    write_json(out / "resume.json", pointer)
    if previous and previous["optimizer"] != optimizer_name:
        (out / previous["optimizer"]).unlink()
    return pointer


def run(args):
    retired = [p / 'TRAINING_INPUTS_RETIRED.json' for p in args.pair.resolve().parents]
    if any(p.is_file() for p in retired):
        raise RuntimeError('这些输入已因矩形任务重复/伪标签接缝退役，仅保留失败复核，禁止续训')
    if args.steps not in (64, 128):
        raise ValueError("Formal training has only fixed 64/128 step budgets")
    pair, prepared, train = load_pair(args.pair, args.size)
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / "train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(out).free < MIN_FREE:
        raise RuntimeError("At least 18 GiB free disk is required before loading the model")
    if args.prepare and args.probe:
        raise ValueError("--prepare and --probe are mutually exclusive")
    if (out / "complete.json").exists() or (args.probe and any(
        (out / name).exists() for name in ("resume.json", "probe.json", "state.json")
    )):
        raise RuntimeError("Refusing to overwrite completed or existing probe output")
    num_frames = pair["num_frames"]
    contract = {"pair": pair, "frame_count": num_frames, "resolution": args.size, "seed": 6201,
                "native_parameter_count": SCOPE_COUNT, "optimizer": "bitsandbytes.AdamW8bit",
                "loss": "official StandardDiffusionLoss / EDM unchanged", "branch": "official only"}
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Existing output has a different pair or training contract")
    else:
        write_json(out / "config.json", contract)
    if args.prepare:
        batch = train.sample(*prepared, 6201)
        if batch["jpg"].shape != (num_frames, 3, *args.size) or batch["cond_frames"].shape != batch["jpg"].shape:
            raise RuntimeError("Official video batch shape failed")
        if batch["num_video_frames"] != num_frames or batch["mask_concat"].shape != (num_frames, 1, args.size[0]//8, args.size[1]//8):
            raise RuntimeError("Official video mask/frame fields failed")
        write_json(out / "prepare.json", {"shape_y": list(prepared[0].shape),
                   "shape_masked_x": list(prepared[1].shape), "shape_hole": list(prepared[2].shape),
                   "batch_jpg": list(batch["jpg"].shape), "batch_mask": list(batch["mask_concat"].shape),
                   "outside_hole_equal": True, "qa_record": pair["qa_record"]})
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for --probe or training; --prepare is CPU-only")
    import bitsandbytes as bnb
    torch.set_num_threads(4)
    random.seed(6201)
    torch.manual_seed(6201)
    torch.cuda.manual_seed_all(6201)
    start = time.monotonic()
    model, params, names = init_model(train)
    optimizer = bnb.optim.AdamW8bit(params, lr=1e-5, weight_decay=0.01)
    resume = json.loads((out / "resume.json").read_text()) if (out / "resume.json").exists() else None
    if resume and not args.probe:
        restore(model, optimizer, names, out, resume)
    first = resume["step"] if resume else 0
    if first >= args.steps:
        raise RuntimeError("Requested steps already have a checkpoint; refusing duplicate training")
    torch.cuda.reset_peak_memory_stats()
    for step in range(first, 1 if args.probe else args.steps):
        optimizer.zero_grad(set_to_none=True)
        value = train.loss(model, prepared, 6201 + step)
        if not torch.isfinite(value):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        value.backward()
        active = [p for p in params if p.grad is not None]
        nonzero = None
        if step == 0:
            bad = sum(not torch.isfinite(p.grad).all().item() for p in active)
            nonzero = sum(bool(p.grad.abs().max().item() > 0) for p in active)
            if bad or not nonzero:
                raise FloatingPointError(f"Invalid gradients: bad={bad}, nonzero={nonzero}")
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
        optimizer.step()  # Probe includes first allocation/update of 8-bit optimizer states.
        if step == 0 and any(not torch.isfinite(p).all().item() for p in active):
            raise FloatingPointError(f"Non-finite updated parameter at step {step}")
        row = {"step": step + 1, "loss": float(value.detach()), "grad_norm": norm,
               "active_grad_tensors": len(active), "nonzero_grad_tensors": nonzero,
               "missing_grad_tensors": len(params) - len(active),
               "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
               "elapsed_seconds": time.monotonic() - start}
        if args.probe:
            write_json(out / "probe.json", row | {"research_checkpoint": False})
            return
        with (out / "steps.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(out / "state.json", row | {"status": "running"})
        if step + 1 in (32, 64, 128):
            resume = snapshot(out, step + 1, model, names, optimizer, resume)
    write_json(out / "complete.json", {"status": "complete", "steps": args.steps,
               "checkpoint": resume["weights"], "supervision_kind": pair["supervision_kind"]})
    write_json(out / "state.json", row | {"status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--size", type=int, nargs=2, default=[320, 576])
    parser.add_argument("--prepare", action="store_true", help="Validate pair and video batch shapes on CPU")
    parser.add_argument("--probe", action="store_true", help="One backward plus optimizer step; no research weights")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.outdir.mkdir(parents=True, exist_ok=True)
        write_json(args.outdir / "error.json", {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "probe": args.probe})
        raise


if __name__ == "__main__":
    main()
