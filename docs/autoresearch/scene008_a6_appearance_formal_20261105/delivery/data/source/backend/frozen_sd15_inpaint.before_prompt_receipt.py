"""Offline, frozen SD1.5 inpainting with optional BUILD a6 IP-Adapter reference.

Inputs are RGB, a white=inpaint mask, and an optional PVD silhouette.
The model is never trained. Exact input pixels are restored outside the effective
mask. Batch candidates are fixed by an explicit manifest, with no adaptive tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODEL_HOME = Path("/root/autodl-tmp/models/worldsim_v82_v42_freeinpaint_20261024/hub")
SD15 = MODEL_HOME / "models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
IPADAPTER = MODEL_HOME / "models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729"
PARAMETERS = {
    "scheduler": "DDIMScheduler",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "strength": 1.0,
    "eta": 0.0,
    "ip_adapter_scale": 0.65,
    "negative_prompt": "blurry, low quality, distorted vehicle, extra wheels, duplicate car, text, watermark",
    "dtype": "float16",
    "variant": "fp16",
    "generator_device": "cpu",
    "mask_white_means": "generate",
    "mask_threshold": 128,
    "resize": "none; dimensions must be multiples of 8",
    "lcm_lora": False,
    "training_updates": 0,
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_record(path):
    p = Path(path).resolve(strict=True)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}


def model_files(use_reference):
    required = [
        SD15 / "model_index.json",
        SD15 / "unet/config.json",
        SD15 / "unet/diffusion_pytorch_model.fp16.safetensors",
        SD15 / "vae/config.json",
        SD15 / "vae/diffusion_pytorch_model.fp16.safetensors",
        SD15 / "text_encoder/config.json",
        SD15 / "text_encoder/model.fp16.safetensors",
        SD15 / "scheduler/scheduler_config.json",
    ]
    required += sorted((SD15 / "tokenizer").glob("*"))
    if use_reference:
        required += [
            IPADAPTER / "models/ip-adapter_sd15.bin",
            IPADAPTER / "models/image_encoder/config.json",
            IPADAPTER / "models/image_encoder/model.safetensors",
        ]
    return [file_record(p) for p in required if p.is_file() or not p.exists()]


def load_pipeline(use_reference, device):
    from importlib import metadata
    import torch
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    from diffusers.utils import logging as diffusers_logging
    from transformers.utils import logging as transformers_logging

    diffusers_logging.disable_progress_bar()
    transformers_logging.disable_progress_bar()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(SD15), torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, local_files_only=True,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if use_reference:
        pipe.load_ip_adapter(
            str(IPADAPTER), subfolder="models", weight_name="ip-adapter_sd15.bin",
            local_files_only=True, image_encoder_folder="image_encoder",
        )
        pipe.set_ip_adapter_scale(PARAMETERS["ip_adapter_scale"])
    components = {}
    for name, module in pipe.components.items():
        if isinstance(module, torch.nn.Module):
            module.eval().requires_grad_(False)
            components[name] = {
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "training": module.training,
            }
    if any(v["trainable_parameters"] or v["training"] for v in components.values()):
        raise RuntimeError("All components must be frozen and in eval mode")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    if device != "cpu":
        pipe = pipe.to(device)
    versions = {name: metadata.version(name) for name in (
        "torch", "diffusers", "transformers", "accelerate", "safetensors", "Pillow", "numpy")}
    return pipe, {
        "versions": versions, "python": sys.version, "platform": platform.platform(),
        "device": device, "components": components,
        "scheduler_config": dict(pipe.scheduler.config),
        "sd15_snapshot": str(SD15), "ip_adapter_snapshot": str(IPADAPTER) if use_reference else None,
    }


def prepare_job(job):
    import numpy as np
    from PIL import Image

    image = Image.open(job["image"]).convert("RGB")
    mask = Image.open(job["mask"]).convert("L")
    if image.size != mask.size or any(s % 8 for s in image.size):
        raise ValueError("image/mask must match exactly and have dimensions divisible by 8")
    original = np.asarray(image).copy()
    original_mask = np.asarray(mask) >= PARAMETERS["mask_threshold"]
    effective_mask = original_mask.copy()
    records = {"image": file_record(job["image"]), "mask": file_record(job["mask"])}
    outside_silhouette = 0
    if job.get("silhouette"):
        silhouette_image = Image.open(job["silhouette"]).convert("L")
        if silhouette_image.size != image.size:
            raise ValueError("silhouette must have the exact image dimensions")
        silhouette = np.asarray(silhouette_image) >= PARAMETERS["mask_threshold"]
        outside_silhouette = int((effective_mask & ~silhouette).sum())
        effective_mask &= silhouette
        records["silhouette"] = file_record(job["silhouette"])
    protected_overlap = 0
    if job.get("known_mask"):
        known_image = Image.open(job["known_mask"]).convert("L")
        if known_image.size != image.size:
            raise ValueError("known_mask must have the exact image dimensions")
        known = np.asarray(known_image) >= PARAMETERS["mask_threshold"]
        protected_overlap = int((effective_mask & known).sum())
        effective_mask &= ~known
        records["known_mask"] = file_record(job["known_mask"])
    if not effective_mask.any():
        raise ValueError("effective inpainting mask is empty")
    reference = None
    if job.get("reference"):
        contract = job.get("reference_contract", {})
        if not (contract.get("split") == "BUILD" and contract.get("actor_id") == "a6"
                and contract.get("held_rgb_used") is False
                and contract.get("user_reference_used") is False and contract.get("source")):
            raise ValueError("reference requires BUILD/a6 provenance, held_rgb_used=false, user_reference_used=false and source")
        reference = Image.open(job["reference"]).convert("RGB")
        records["reference"] = file_record(job["reference"])
    return image, Image.fromarray(effective_mask.astype("uint8") * 255), original, effective_mask, reference, {
        "files": records, "width": image.width, "height": image.height,
        "original_mask_pixels": int(original_mask.sum()),
        "effective_mask_pixels": int(effective_mask.sum()),
        "mask_pixels_clipped_by_silhouette": outside_silhouette,
        "mask_pixels_protected_by_known_mask": protected_overlap,
        "reference_contract": job.get("reference_contract"),
    }


def run_job(pipe, job, prepared, root, manifest_hash):
    import numpy as np
    from PIL import Image
    import torch

    image, mask, original, effective_mask, reference, inputs = prepared
    destination = root / job["id"]
    destination.mkdir(parents=True, exist_ok=False)
    mask.save(destination / "effective_mask.png")
    generator = torch.Generator(device="cpu").manual_seed(int(job["seed"]))
    kwargs = dict(
        prompt=job["prompt"], negative_prompt=PARAMETERS["negative_prompt"],
        image=image, mask_image=mask, width=image.width, height=image.height,
        num_inference_steps=PARAMETERS["num_inference_steps"],
        guidance_scale=PARAMETERS["guidance_scale"], strength=PARAMETERS["strength"],
        eta=PARAMETERS["eta"], generator=generator, output_type="np",
    )
    if reference is not None:
        kwargs["ip_adapter_image"] = reference
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        generated_float = pipe(**kwargs).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not np.isfinite(generated_float).all():
        raise RuntimeError("Non-finite model output")
    raw = np.rint(np.clip(generated_float, 0, 1) * 255).astype("uint8")
    generated = Image.fromarray(raw)
    if raw.shape != original.shape:
        raise RuntimeError("pipeline output shape mismatch")
    composed = np.where(effective_mask[..., None], raw, original).astype("uint8")
    delta = np.abs(composed.astype("int16") - original.astype("int16"))
    if delta[~effective_mask].any():
        raise RuntimeError("preserved pixels changed")
    generated.save(destination / "generated_raw.png")
    Image.fromarray(composed).save(destination / "composed.png")
    receipt = {
        "status": "FROZEN_INFERENCE_COMPLETE", "job": job,
        "parameters": PARAMETERS, "inputs": inputs,
        "model_manifest_sha256": manifest_hash,
        "seconds": elapsed, "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "training_updates": 0, "ranking_or_selection_performed": False,
        "finite_output": True,
        "known_pixels_preserved_exactly": True,
        "outside_effective_mask_max_abs_rgb_u8": int(delta[~effective_mask].max()) if (~effective_mask).any() else 0,
        "generated_image_is_inferred_appearance": True,
        "outputs": {name: file_record(destination / name) for name in ("generated_raw.png", "composed.png", "effective_mask.png")},
    }
    write_json(destination / "receipt.json", receipt)
    print(json.dumps({"job": job["id"], "seconds": elapsed, "output": str(destination / "composed.png")}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-manifest", help="JSON with jobs list; each job has id/image/mask/prompt/seed and optional silhouette/known_mask/reference/reference_contract")
    parser.add_argument("--image")
    parser.add_argument("--mask")
    parser.add_argument("--silhouette")
    parser.add_argument("--known-mask")
    parser.add_argument("--reference")
    parser.add_argument("--reference-contract", help="JSON file proving reference BUILD/a6 provenance")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--job-id", default="smoke")
    parser.add_argument("--check-model", action="store_true", help="CPU offline model load and frozen-parameter check; no inference")
    parser.add_argument("--check-with-reference", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "model_manifest.json").exists():
        raise FileExistsError("Use a fresh output folder to preserve prior run evidence")
    if args.check_model:
        jobs, prepared = [], []
        use_reference = args.check_with_reference
    else:
        if args.batch_manifest:
            batch = json.loads(Path(args.batch_manifest).read_text())
            jobs = batch["jobs"]
        else:
            if not (args.image and args.mask and args.prompt):
                parser.error("--image, --mask and --prompt are required without --batch-manifest")
            job = dict(id=args.job_id, image=args.image, mask=args.mask, silhouette=args.silhouette,
                       known_mask=args.known_mask, reference=args.reference, prompt=args.prompt, seed=args.seed)
            if args.reference_contract:
                job["reference_contract"] = json.loads(Path(args.reference_contract).read_text())
            jobs = [job]
        if not jobs or len({j["id"] for j in jobs}) != len(jobs):
            raise ValueError("Non-empty jobs with unique ids required")
        if any(not j["id"] or Path(j["id"]).name != j["id"] for j in jobs):
            raise ValueError("job ids must be simple folder names")
        use_reference = any(j.get("reference") for j in jobs)
        if use_reference and not all(j.get("reference") for j in jobs):
            raise ValueError("A batch must consistently enable or disable IP-Adapter")
        prepared = [prepare_job(j) for j in jobs]
    file_hashes = model_files(use_reference)
    pipe, environment = load_pipeline(use_reference, "cpu" if args.check_model else "cuda")
    manifest = {
        "status": "OFFLINE_FROZEN_MODEL_LOADED", "files": file_hashes,
        "source": file_record(__file__), "parameters": PARAMETERS,
        "environment": environment, "training_updates": 0, "downloads": 0,
    }
    write_json(root / "model_manifest.json", manifest)
    manifest_hash = sha256(root / "model_manifest.json")
    if args.check_model:
        print(json.dumps({"status": manifest["status"], "manifest": str(root / "model_manifest.json")}), flush=True)
        return
    write_json(root / "batch_manifest.json", {"jobs": jobs})
    receipts = [run_job(pipe, job, p, root, manifest_hash) for job, p in zip(jobs, prepared)]
    write_json(root / "batch_receipt.json", {
        "status": "FROZEN_BATCH_COMPLETE", "candidates": len(receipts),
        "training_updates": 0, "downloads": 0,
        "model_manifest_sha256": manifest_hash,
        "jobs": [{"id": r["job"]["id"], "seconds": r["seconds"], "output": r["outputs"]["composed.png"]} for r in receipts],
    })


if __name__ == "__main__":
    main()
