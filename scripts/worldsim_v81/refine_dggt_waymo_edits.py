#!/usr/bin/env python3
"""Offline, framewise official Difix refinement of DGGT Gaussian edit PNGs.

Inputs are the already saved noop/delete/move/optional insert_copy PNGs. This is an appearance
refinement, not object removal or inpainting. The author model math and the
"remove degradation" prompt are retained; one model instance is reused for
all frames. The only source compatibility edit is replacing the hard-coded
sd-turbo repository ID with its complete local checkpoint directory in memory.
The upstream files on disk are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path


BASE_BRANCHES = ("noop", "delete", "move")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external", type=Path, required=True, help="Official DGGT root")
    p.add_argument("--edits-dir", type=Path, required=True, help="Gaussian edit output with noop/delete/move PNG subdirectories")
    p.add_argument("--output-dir", type=Path, required=True, help="New result directory; existing path is rejected")
    p.add_argument("--difix-checkpoint", type=Path, help="Author Difix weight file; default external/pretrained/diffusion_model.pth")
    p.add_argument("--sequence-length", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234, help="Use the same per-frame VAE noise across branches")
    p.add_argument("--check-only", action="store_true", help="CPU file contract only; no torch import, CUDA, or output")
    return p.parse_args()


def _inspect_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, tuple[str, ...], list[list[Path]], list[int]]:
    from PIL import Image

    external = args.external.resolve()
    edits = args.edits_dir.resolve()
    out = args.output_dir.resolve()
    if args.sequence_length < 1:
        raise ValueError("sequence-length must be positive")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    if not edits.is_dir():
        raise FileNotFoundError(edits)
    model_source = external / "third_party" / "difix" / "src" / "model.py"
    infer_source = external / "third_party" / "difix" / "infer.py"
    if not model_source.is_file() or not infer_source.is_file():
        raise FileNotFoundError("official third_party/difix/{infer.py,src/model.py} required")
    if model_source.read_text(encoding="utf-8").count('"stabilityai/sd-turbo"') != 5:
        raise RuntimeError("official Difix model source has changed; inspect sd-turbo load sites before running")
    sd_turbo = external / "pretrained" / "sd-turbo"
    expected = ("model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                "tokenizer/tokenizer_config.json", "unet/config.json", "vae/config.json")
    missing = [name for name in expected if not (sd_turbo / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete local sd-turbo checkpoint: {missing}")
    checkpoint = (args.difix_checkpoint or external / "pretrained" / "diffusion_model.pth").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    names = BASE_BRANCHES + (("insert_copy",) if (edits / "insert_copy").is_dir() else ())
    branches = []
    dimensions = []
    for name in names:
        folder = edits / name
        paths = [folder / f"{i:03d}.png" for i in range(args.sequence_length)]
        if not folder.is_dir() or any(not p.is_file() for p in paths):
            raise FileNotFoundError(f"{name}: expected contiguous PNG frames 000..{args.sequence_length - 1:03d} in {folder}")
        actual = sorted(folder.glob("*.png"))
        if actual != paths:
            raise ValueError(f"{folder}: expected exactly {args.sequence_length} numbered PNGs")
        for path in paths:
            with Image.open(path) as im:
                if im.mode not in ("RGB", "RGBA"):
                    raise ValueError(f"{path}: expected RGB or RGBA PNG")
                dimensions.append(im.size)
        branches.append(paths)
    if len(set(dimensions)) != 1:
        raise ValueError(f"input branch/frame dimensions differ: {sorted(set(dimensions))}")
    return external, edits, checkpoint, names, branches, list(dimensions[0])


def _load_offline_author_model(external: Path, checkpoint: Path):
    """Load upstream Difix class after replacing only its sd-turbo repo literal."""
    local_sd = external / "pretrained" / "sd-turbo"
    model_path = external / "third_party" / "difix" / "src" / "model.py"
    source = model_path.read_text(encoding="utf-8")
    original = '"stabilityai/sd-turbo"'
    replacements = source.count(original)
    if replacements != 5:
        raise RuntimeError(f"expected five author sd-turbo literals, found {replacements}; inspect upstream source before running")
    patched = source.replace(original, json.dumps(str(local_sd)))
    # The source itself remains untouched. Register the module under the
    # official package name so class/module references keep upstream identity.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    sys.path.insert(0, str(external))
    module_name = "third_party.difix.src.model"
    module = types.ModuleType(module_name)
    module.__file__ = str(model_path)
    module.__package__ = "third_party.difix.src"
    sys.modules[module_name] = module
    try:
        exec(compile(patched, str(model_path), "exec"), module.__dict__)
        model = module.Difix(pretrained_path=str(checkpoint), timestep=199, mv_unet=False)
        model.set_eval()
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return model, replacements


def _refine_one(path: Path, model, torch, transforms, F):
    """Official third_party/difix/infer.py process math, with model reuse."""
    from PIL import Image

    with Image.open(path) as im:
        img_tensor = transforms.ToTensor()(im.convert("RGB"))
    _, original_h, original_w = img_tensor.shape
    new_h = ((original_h + 7) // 8) * 8
    new_w = ((original_w + 7) // 8) * 8
    if new_h != original_h or new_w != original_w:
        img_pil = transforms.ToPILImage()(img_tensor)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tensor = transforms.ToTensor()(img_pil)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    input_pil = transforms.ToPILImage()(img_tensor.cpu())
    model.sched.set_timesteps(1, device=device)
    model.sched.timesteps = torch.tensor([model.timesteps.item()], device=device)
    output_pil = model.sample(input_pil, width=new_w, height=new_h, prompt="remove degradation")
    output_tensor = transforms.ToTensor()(output_pil)
    if new_h != original_h or new_w != original_w:
        output_tensor = F.interpolate(output_tensor.unsqueeze(0), size=(original_h, original_w),
                                      mode="bilinear", align_corners=False).squeeze(0)
    return transforms.ToPILImage()(output_tensor.cpu().clamp(0, 1))


def main() -> int:
    args = _args()
    external, edits, checkpoint, names, inputs, dimensions = _inspect_inputs(args)
    if args.check_only:
        print(json.dumps({"status": "cpu_contract_ok", "input_frames_per_branch": args.sequence_length,
                          "branches": list(names), "size_wh": dimensions, "difix_checkpoint": str(checkpoint),
                          "sd_turbo": str(external / "pretrained" / "sd-turbo")}, ensure_ascii=False))
        return 0

    # Heavy imports and CUDA use begin only after the CPU-only file contract.
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms

    if not torch.cuda.is_available():
        raise RuntimeError("Author Difix model.py requires CUDA; use --check-only for CPU contract")
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model, replacements = _load_offline_author_model(external, checkpoint)
    load_seconds = time.perf_counter() - load_start
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    branch_stats = {}
    with torch.no_grad():
        for name, paths in zip(names, inputs):
            folder = out / name
            folder.mkdir()
            frames = []
            elapsed = []
            for index, path in enumerate(paths):
                # Author VAE draws a latent sample. Pair the same frame across
                # edits with a fixed seed; this is an experiment control.
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                start = time.perf_counter()
                image = _refine_one(path, model, torch, transforms, F)
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - start)
                image.save(folder / f"{index:03d}.png")
                frames.append(np.asarray(image))
            imageio.mimwrite(out / f"{name}.mp4", frames, fps=8, codec="libx264")
            branch_stats[name] = {"frame_seconds": elapsed, "total_seconds": sum(elapsed)}
    manifest = {
        "status": "completed", "method": "official_difix_framewise_offline_reuse_one_model",
        "source_edits_dir": str(edits), "source_pngs": {name: [str(p) for p in paths] for name, paths in zip(names, inputs)},
        "upstream_infer": str(external / "third_party" / "difix" / "infer.py"),
        "upstream_model": str(external / "third_party" / "difix" / "src" / "model.py"),
        "compatibility": {"only_in_memory_change": "replace five hard-coded stabilityai/sd-turbo literals with local sd-turbo path",
                          "replacements": replacements, "sd_turbo": str(external / "pretrained" / "sd-turbo"),
                          "offline_environment": True, "upstream_files_modified": False},
        "difix_checkpoint": str(checkpoint), "model_loads": 1, "timestep": 199,
        "mv_unet": False, "prompt": "remove degradation", "size_wh": dimensions,
        "paired_noise_seed_per_frame": args.seed, "model_load_seconds": load_seconds,
        "branch_stats": branch_stats, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "limits": ["This is framewise Difix appearance refinement, not object removal or inpainting.",
                   "Input PNG quantization precedes Difix; this entrypoint does not refine in-memory Gaussian renderer tensors.",
                   "The author infer.py normally instantiates Difix per frame; this entrypoint reuses one loaded model across all branches."]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output_dir": str(out), "model_loads": 1,
                      "model_load_seconds": load_seconds, "branch_stats": branch_stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
