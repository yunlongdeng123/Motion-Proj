#!/usr/bin/env python3
"""在隔离进程中执行一个 R8 frozen proposal generator。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_big_lama(root: Path) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    source = root / "source"
    model_root = root / "big-lama"
    # LaMa 推理模块只需要 seed_everything；避免为未使用的训练器引入旧版 Lightning。
    if "pytorch_lightning" not in sys.modules:
        lightning_stub = types.ModuleType("pytorch_lightning")

        def seed_everything(seed: int) -> int:
            torch.manual_seed(seed)
            return seed

        lightning_stub.seed_everything = seed_everything
        sys.modules["pytorch_lightning"] = lightning_stub
    # 官方归档混入了训练期 ModelCheckpoint；以无行为占位类满足安全反序列化。
    checkpoint_type = type(
        "ModelCheckpoint",
        (),
        {"__module__": "pytorch_lightning.callbacks.model_checkpoint"},
    )
    callbacks_stub = types.ModuleType("pytorch_lightning.callbacks")
    model_checkpoint_stub = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")
    model_checkpoint_stub.ModelCheckpoint = checkpoint_type
    callbacks_stub.model_checkpoint = model_checkpoint_stub
    sys.modules.setdefault("pytorch_lightning.callbacks", callbacks_stub)
    sys.modules.setdefault("pytorch_lightning.callbacks.model_checkpoint", model_checkpoint_stub)
    torch.serialization.add_safe_globals(
        [
            checkpoint_type,
            ContainerMetadata,
            Metadata,
            DictConfig,
            ListConfig,
            AnyNode,
            Any,
        ]
    )
    sys.path.insert(0, str(source))
    from saicinpainting.training.modules import make_generator

    # 必须先解析官方配置的字段引用，否则比例字段会保留为插值字符串。
    config = OmegaConf.load(model_root / "config.yaml")
    generator_config = dict(OmegaConf.to_container(config.generator, resolve=True))
    kind = generator_config.pop("kind")
    model = make_generator(None, kind=kind, **generator_config)
    # 只读取 state_dict，避免反序列化 checkpoint 内未参与推理的训练回调对象。
    checkpoint = torch.load(
        model_root / "models/best.ckpt", map_location="cpu", weights_only=True
    )
    state = {
        key[len("generator.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("generator.")
    }
    model.load_state_dict(state, strict=True)
    model.eval().to("cuda")

    def infer(image: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None].cuda()
        model_input = torch.cat([tensor * (1.0 - mask_tensor), mask_tensor], dim=1)
        with torch.inference_mode():
            prediction = model(model_input)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        return prediction[0].permute(1, 2, 0).float().cpu().numpy()

    return infer


def _load_sd15(
    root: Path, prompt: str, steps: int, guidance_scale: float
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    from diffusers import StableDiffusionInpaintPipeline

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        root,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")

    def infer(image: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipeline(
            prompt=prompt,
            image=Image.fromarray(image, mode="RGB"),
            mask_image=Image.fromarray(mask.astype(np.uint8) * 255, mode="L"),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=image.shape[0],
            width=image.shape[1],
        ).images[0]
        return np.asarray(result, dtype=np.float32) / 255.0

    return infer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=["big_lama", "sd15_inpainting"])
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--big-lama-root", type=Path)
    parser.add_argument("--sd15-root", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repeat-count", required=True, type=int)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.monotonic()
    if args.candidate == "big_lama":
        if args.big_lama_root is None:
            raise RuntimeError("big_lama root 未设置")
        infer = _load_big_lama(args.big_lama_root)
    else:
        if args.sd15_root is None:
            raise RuntimeError("sd15 root 未设置")
        infer = _load_sd15(
            args.sd15_root, args.prompt, args.inference_steps, args.guidance_scale
        )
    load_seconds = time.monotonic() - load_started
    rows = []
    for input_path in sorted(args.input_dir.glob("*.npz")):
        with np.load(input_path, allow_pickle=False) as archive:
            image = np.asarray(archive["image"], dtype=np.uint8)
            mask = np.asarray(archive["mask"], dtype=bool)
        case_rows = []
        for repeat_index in range(args.repeat_count):
            torch.cuda.synchronize()
            started = time.monotonic()
            raw = infer(image, mask, args.seed)
            torch.cuda.synchronize()
            latency = time.monotonic() - started
            finite = bool(np.all(np.isfinite(raw)))
            raw_uint8 = np.rint(np.clip(raw, 0.0, 1.0) * 255.0).astype(np.uint8)
            overlay = image.copy()
            overlay[mask] = raw_uint8[mask]
            outside_exact = bool(np.array_equal(overlay[~mask], image[~mask]))
            masked_change = float(
                np.mean(
                    np.abs(
                        raw_uint8[mask].astype(np.float32)
                        - image[mask].astype(np.float32)
                    )
                )
                / 255.0
            )
            output_path = args.output_dir / f"{input_path.stem}__repeat{repeat_index + 1}.npy"
            np.save(output_path, overlay, allow_pickle=False)
            case_rows.append(
                {
                    "repeat_index": repeat_index + 1,
                    "seed": args.seed,
                    "latency_seconds": latency,
                    "finite": finite,
                    "masked_change": masked_change,
                    "outside_mask_exact": outside_exact,
                    "output": output_path.name,
                    "output_sha256": _sha256(output_path),
                }
            )
        rows.append(
            {
                "case_id": input_path.stem,
                "repeats": case_rows,
                "repeat_sha_exact": len({row["output_sha256"] for row in case_rows}) == 1,
            }
        )
    all_repeats = [repeat for row in rows for repeat in row["repeats"]]
    result = {
        "schema_version": "worldsim_v6.r8_worker_result.v1",
        "candidate": args.candidate,
        "case_count": len(rows),
        "repeat_count": args.repeat_count,
        "load_seconds": load_seconds,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_reserved() / (1024**2)),
        "median_latency_seconds": float(
            np.median([row["latency_seconds"] for row in all_repeats])
        ),
        "all_repeats_successful": len(all_repeats) == len(rows) * args.repeat_count,
        "all_repeat_sha_exact": all(row["repeat_sha_exact"] for row in rows),
        "all_finite": all(row["finite"] for row in all_repeats),
        "all_nonzero_masked_change": all(row["masked_change"] > 0.0 for row in all_repeats),
        "all_outside_mask_exact": all(row["outside_mask_exact"] for row in all_repeats),
        "cases": rows,
    }
    _write_json(args.output_dir / "WORKER_RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
