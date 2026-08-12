#!/usr/bin/env python
"""用冻结 SAM2.1 为 S1 evaluation 分区生成 pseudo target。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.semantic_schema import sha256_file
from motion_proj.worldsim_v33.evaluation_partition import (
    manifest_evaluation_partition,
    resolve_evaluation_frames,
)
from scripts.build_worldsim_v32_sam_masks import quality_gate


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--partition", choices=("development", "heldout"), default="heldout"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖 evaluation mask 目录: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prompts = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if prompts.get("config_sha256") != sha256_file(args.config):
        raise RuntimeError("evaluation prompt/config SHA 不一致")
    if not prompts.get("optimization_forbidden"):
        raise RuntimeError("evaluation prompt 未声明 optimization_forbidden")
    evaluation_frames = set(resolve_evaluation_frames(config, args.partition))
    if manifest_evaluation_partition(prompts) != args.partition:
        raise RuntimeError("evaluation partition 漂移")
    if set(int(value) for value in prompts["evaluation_frames"]) != evaluation_frames:
        raise RuntimeError("evaluation frame 集漂移")

    sam = config["sam2_fallback"]
    source_root = Path(sam["source_checkout"])
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != sam["source_commit"]:
        raise RuntimeError("SAM2 source commit 漂移")
    checkpoint = Path(sam["checkpoint"])
    checkpoint_before = sha256_file(checkpoint)
    if checkpoint_before != sam["checkpoint_sha256"]:
        raise RuntimeError("SAM2 checkpoint SHA 漂移")
    runtime_contract = sam["runtime"]
    for name, path_key, hash_key in (
        ("conda explicit", "conda_explicit", "conda_explicit_sha256"),
        ("pip freeze", "pip_freeze", "pip_freeze_sha256"),
    ):
        if sha256_file(runtime_contract[path_key]) != runtime_contract[hash_key]:
            raise RuntimeError(f"SAM2 {name} SHA 漂移")
    import torchvision
    runtime_actual = {
        "environment": str(Path(sys.executable).resolve().parents[1]),
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
    }
    for name in ("environment", "python", "torch", "torchvision", "numpy"):
        if runtime_actual[name] != str(runtime_contract[name]):
            raise RuntimeError(
                f"SAM2 runtime {name} 漂移: expected={runtime_contract[name]} "
                f"actual={runtime_actual[name]}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation SAM2 需要可见 CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.memory_allocated(device) > 2 * 1024**3:
        raise RuntimeError("evaluation SAM2 GPU preflight 非空闲")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from sam2.build_sam import build_sam2_video_predictor

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    predictor = build_sam2_video_predictor(
        sam["model_config"], str(checkpoint), device=str(device), vos_optimized=False
    )
    target_height = int(config["outputs"]["model_native_height"])
    target_width = int(config["outputs"]["model_native_width"])
    threshold = float(sam["mask_logit_threshold"])
    rows: list[dict[str, Any]] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for block in prompts["blocks"]:
            frame = int(block["frame"])
            if frame not in evaluation_frames or len(block["frames"]) != 1:
                raise RuntimeError("evaluation block 不满足 singleton/eval-only 合同")
            video_dir = Path(block["video_dir"])
            if not video_dir.is_absolute():
                video_dir = args.prompt_manifest.parent / video_dir
            state = predictor.init_state(video_path=str(video_dir), offload_video_to_cpu=True)
            predictor.reset_state(state)
            for prompt in block["prompts"]:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(prompt["object_id"]),
                    box=np.asarray(prompt["box_xyxy"], dtype=np.float32),
                )
            outputs = list(predictor.propagate_in_video(state))
            if len(outputs) != 1 or int(outputs[0][0]) != 0:
                raise RuntimeError("SAM2 singleton propagation 输出不合法")
            _, object_ids, logits = outputs[0]
            frame_row = block["frames"][0]
            prompt_by_id = {int(row["object_id"]): row for row in block["prompts"]}
            for offset, object_id in enumerate(object_ids):
                numeric_id = int(object_id)
                prompt = prompt_by_id[numeric_id]
                role = prompt["role"]
                raw_value = logits[offset : offset + 1].float()
                if raw_value.ndim == 3:
                    raw_value = raw_value.unsqueeze(0)
                if raw_value.ndim != 4:
                    raise RuntimeError(
                        f"SAM2 evaluation logits rank 不合法: {raw_value.shape}"
                    )
                value = functional.interpolate(
                    raw_value,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                logits_np = value.cpu().numpy().astype(np.float16)
                raw_binary = logits_np > threshold
                source_box = frame_row["projected_boxes"][role]
                scale = np.asarray(
                    [
                        target_width / float(frame_row["width"]),
                        target_height / float(frame_row["height"]),
                        target_width / float(frame_row["width"]),
                        target_height / float(frame_row["height"]),
                    ]
                )
                projected_box = (np.asarray(source_box) * scale).tolist()
                accepted, reasons, metrics = quality_gate(
                    binary=raw_binary,
                    projected_box=projected_box,
                    previous=None,
                    quality=sam["quality_gate"],
                )
                binary = raw_binary if accepted else np.zeros_like(raw_binary)
                output = args.output_dir / "masks" / role / block["camera_name"] / f"{frame:03d}.npz"
                atomic_npz(output, logits=logits_np, raw_binary=raw_binary, binary=binary)
                rows.append(
                    {
                        "role": role,
                        "instance_token": config["actors"][role]["instance_token"],
                        "object_id": numeric_id,
                        "frame": frame,
                        "camera_id": int(block["camera_id"]),
                        "camera_name": block["camera_name"],
                        "source_image": frame_row["image"],
                        "source_image_sha256": frame_row["image_sha256"],
                        "prompt_box_xyxy": prompt["box_xyxy"],
                        "projected_box_xyxy": projected_box,
                        "mask": str(output),
                        "mask_sha256": sha256_file(output),
                        "height": target_height,
                        "width": target_width,
                        "accepted": accepted,
                        "rejection_reasons": reasons,
                        "quality_metrics": metrics,
                        "positive_pixels": int(binary.sum()),
                    }
                )
            del state
            torch.cuda.empty_cache()
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("SAM2 checkpoint 在 evaluation 推理后发生 mutation")
    manifest = {
        "schema_version": (
            "worldsim_v33_s1_heldout_mask_manifest_v1"
            if args.partition == "heldout"
            else "worldsim_v33_s1_eval_mask_manifest_v2"
        ),
        "task_id": config["task_id"],
        "config_sha256": sha256_file(args.config),
        "prompt_manifest": str(args.prompt_manifest),
        "prompt_manifest_sha256": sha256_file(args.prompt_manifest),
        "source_commit": source_commit,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "evaluation_partition": args.partition,
        "evaluation_frames": sorted(evaluation_frames),
        "optimization_forbidden": True,
        "mask_count": len(rows),
        "accepted_mask_count": sum(bool(row["accepted"]) for row in rows),
        "rejected_mask_count": sum(not bool(row["accepted"]) for row in rows),
        "runtime": {
            **runtime_actual,
            "wall_seconds": time.monotonic() - started,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "conda_explicit_sha256": runtime_contract["conda_explicit_sha256"],
            "pip_freeze_sha256": runtime_contract["pip_freeze_sha256"],
        },
        "masks": sorted(rows, key=lambda row: (row["role"], row["frame"], row["camera_id"])),
    }
    atomic_json(args.output_dir / "mask_manifest.json", manifest)
    print(json.dumps({"status": "done", "mask_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
