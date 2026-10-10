#!/usr/bin/env python3
"""Bounded DGGT Difix reference-image diagnostic for existing Waymo edits.

The GPU phase replays eight prior single-view frames and renders sixteen
two-view frames. --check-only performs CPU imports, file checks, and weight
shape checks without making any output or touching CUDA.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path


BRANCHES = ("noop", "delete")
CONDITIONS = ("official_public_single", "official_mv_self", "official_mv_legal_ref")
PROMPT = "remove degradation"
TIMESTEP = 199
FRAMES = 4
MAX_OUTPUT_FRAMES = 24
SEED_BASE = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, required=True, help="Official DGGT checkout")
    parser.add_argument("--edits-dir", type=Path, required=True, help="Existing gaussian_edits directory")
    parser.add_argument("--prior-dir", type=Path, help="Existing difix_refined directory; defaults to edits-dir sibling")
    parser.add_argument("--output-dir", type=Path, required=True, help="New diagnostics/reference_refinement_r1 directory")
    parser.add_argument("--reference-rgb", type=Path, help="Existing official input 000_0 RGB; default from edit manifest")
    parser.add_argument("--difix-checkpoint", type=Path, help="Downloaded author checkpoint; default external/pretrained/diffusion_model.pth")
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--check-only", action="store_true", help="CPU-only contract/import/weight-shape check; no GPU output")
    parser.add_argument("--check-report", type=Path, help="Optional new JSON path for --check-only evidence")
    parser.add_argument("--resume", action="store_true", help="Continue a matching incomplete diagnostic from its run.json")
    return parser.parse_args()


def _cgroup(name: str) -> str | None:
    path = Path("/sys/fs/cgroup") / name
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def inspect_inputs(args: argparse.Namespace) -> dict:
    from PIL import Image

    external = args.external.resolve()
    edits = args.edits_dir.resolve()
    prior = (args.prior_dir or edits.parent / "difix_refined").resolve()
    out = args.output_dir.resolve()
    checkpoint = (args.difix_checkpoint or external / "pretrained" / "diffusion_model.pth").resolve()
    model_path = external / "third_party/difix/src/model.py"
    mv_path = external / "third_party/difix/src/mv_unet.py"
    public_single_path = external / "third_party/difix/infer.py"
    public_ref_path = external / "third_party/difix/src/inference_difix.py"
    sd_turbo = external / "pretrained/sd-turbo"
    for path in (edits, prior, external, sd_turbo):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (checkpoint, model_path, mv_path, public_single_path, public_ref_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_sd = (
        "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
        "tokenizer/tokenizer_config.json", "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
    )
    missing = [name for name in expected_sd if not (sd_turbo / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete local sd-turbo: {missing}")
    source = model_path.read_text(encoding="utf-8")
    if source.count('"stabilityai/sd-turbo"') != 5:
        raise RuntimeError("official model.py changed; expected five sd-turbo literals")
    if "from mv_unet import UNet2DConditionModel" not in source:
        raise RuntimeError("official model.py no longer exposes mv_unet=True implementation")
    if "vae.load_state_dict(_sd_vae)" not in source or "unet.load_state_dict(_sd_unet)" not in source:
        raise RuntimeError("official strict weight-load sites changed")
    ref_source = public_ref_path.read_text(encoding="utf-8")
    if "mv_unet=True if args.ref_image is not None else False" not in ref_source or "ref_image=ref_image" not in ref_source:
        raise RuntimeError("official inference_difix.py reference interface changed")
    mv_source = mv_path.read_text(encoding="utf-8")
    if "num_views = 2" not in mv_source or 'b (v n) d' not in mv_source:
        raise RuntimeError("official two-view attention contract changed")

    edit_manifest_path = edits / "manifest.json"
    prior_manifest_path = prior / "manifest.json"
    edit_manifest = json.loads(edit_manifest_path.read_text(encoding="utf-8"))
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    if edit_manifest.get("frames") != FRAMES or prior_manifest.get("status") != "completed":
        raise ValueError("expected completed four-frame edit and prior Difix run")
    if prior_manifest.get("mv_unet") is not False or prior_manifest.get("timestep") != TIMESTEP:
        raise ValueError("prior Difix is not official public single-view timestep 199")
    if args.seed != SEED_BASE or prior_manifest.get("paired_noise_seed_per_frame") != SEED_BASE:
        raise ValueError("protocol requires the prior Difix seed 1234+i for every frame")
    if prior_manifest.get("prompt") != PROMPT:
        raise ValueError("prior Difix prompt differs from the fixed protocol prompt")
    if Path(prior_manifest.get("source_edits_dir", "")).resolve() != edits:
        raise ValueError("prior Difix source edits directory differs from this run")
    if Path(prior_manifest.get("upstream_infer", "")).resolve() != public_single_path.resolve():
        raise ValueError("prior Difix used a different official single-view entrypoint")
    if Path(prior_manifest.get("upstream_model", "")).resolve() != model_path.resolve():
        raise ValueError("prior Difix used a different official model source")
    if Path(prior_manifest["difix_checkpoint"]).resolve() != checkpoint:
        raise ValueError("prior Difix used a different checkpoint")
    ref_from_manifest = Path(edit_manifest["rgb_paths"][0]).resolve()
    ref = (args.reference_rgb or ref_from_manifest).resolve()
    if ref != ref_from_manifest or not ref.is_file() or ref.name != "000_0.jpg":
        raise ValueError("legal reference must be the edit run's original official 000_0.jpg input")
    if any(Path(p).resolve() != ref for p in edit_manifest["rgb_paths"][:1]):
        raise ValueError("reference path differs from official input list")

    paths = {}
    sizes = []
    for branch in BRANCHES:
        paths[branch] = {"raw": [], "prior": []}
        for index in range(FRAMES):
            for role, folder in (("raw", edits), ("prior", prior)):
                path = folder / branch / f"{index:03d}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as image:
                    if image.mode not in ("RGB", "RGBA"):
                        raise ValueError(f"{path}: expected RGB/RGBA")
                    sizes.append(image.size)
                paths[branch][role].append(path)
    with Image.open(ref) as image:
        reference_size = image.size
    if len(set(sizes)) != 1:
        raise ValueError(f"raw/prior PNG sizes differ: {sorted(set(sizes))}")
    if prior_manifest.get("size_wh") != list(sizes[0]):
        raise ValueError("prior Difix manifest dimensions differ from actual PNGs")
    source_pngs = prior_manifest.get("source_pngs", {})
    for branch in BRANCHES:
        if [Path(path).resolve() for path in source_pngs.get(branch, [])] != paths[branch]["raw"]:
            raise ValueError(f"prior Difix source PNG sequence differs for {branch}")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing nonempty output: {out}")
    if args.resume and not (out / "run.json").is_file():
        raise FileNotFoundError(f"--resume requires an existing run.json: {out}")
    return {
        "external": external, "edits": edits, "prior": prior, "out": out,
        "checkpoint": checkpoint, "sd_turbo": sd_turbo, "model_path": model_path,
        "mv_path": mv_path, "public_single_path": public_single_path,
        "public_ref_path": public_ref_path, "edit_manifest": edit_manifest,
        "prior_manifest": prior_manifest, "reference_rgb": ref,
        "reference_size_wh": list(reference_size), "size_wh": list(sizes[0]),
        "paths": paths,
    }


def import_official_mv(src_dir: Path, sd_turbo: Path) -> tuple[types.ModuleType, dict]:
    """Import upstream mv_unet with guarded import-only compatibility names."""
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    evidence = {"old_unet_2d_blocks_alias_applied": False, "alias_target": None,
                "missing_position_net_sentinel_applied": False}
    import diffusers.models.embeddings as embeddings

    if not hasattr(embeddings, "PositionNet"):
        config = json.loads((sd_turbo / "unet/config.json").read_text(encoding="utf-8"))
        attention_type = config.get("attention_type", "default")
        source = (src_dir / "mv_unet.py").read_text(encoding="utf-8")
        gated_site = 'if attention_type in ["gated", "gated-text-image"]:'
        if attention_type != "default" or source.count("PositionNet(") != 1 or gated_site not in source:
            raise RuntimeError("PositionNet compatibility guard requires default attention and one gated-only construction")

        class UnavailablePositionNet:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("PositionNet sentinel was constructed; stop instead of changing gated attention math")

        embeddings.PositionNet = UnavailablePositionNet
        evidence.update(missing_position_net_sentinel_applied=True,
                        position_net_config_attention_type=attention_type,
                        position_net_official_source=f"{src_dir / 'mv_unet.py'}:744-754",
                        position_net_behavior="import name only; construction raises RuntimeError")
    try:
        module = importlib.import_module("mv_unet")
    except ModuleNotFoundError as exc:
        if exc.name != "diffusers.models.unet_2d_blocks":
            raise
        new_name = "diffusers.models.unets.unet_2d_blocks"
        sys.modules[exc.name] = importlib.import_module(new_name)
        sys.modules.pop("mv_unet", None)
        module = importlib.import_module("mv_unet")
        evidence.update(old_unet_2d_blocks_alias_applied=True, alias_target=new_name)
    if Path(module.__file__).resolve() != (src_dir / "mv_unet.py").resolve():
        raise RuntimeError("mv_unet resolved outside official DGGT checkout")
    evidence["official_mv_path"] = str(Path(module.__file__).resolve())
    return module, evidence


def _checkpoint_metadata(contract: dict, torch, mv_module) -> dict:
    """Inspect checkpoint/base keys without instantiating multi-GB CPU models."""
    from diffusers import UNet2DConditionModel
    from safetensors import safe_open

    checkpoint = torch.load(contract["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    for name in ("state_dict_unet", "state_dict_vae", "rank_vae", "vae_lora_target_modules"):
        if name not in checkpoint:
            raise KeyError(f"author Difix checkpoint lacks {name}")
    unet_ckpt = checkpoint["state_dict_unet"]
    vae_ckpt = checkpoint["state_dict_vae"]
    if not unet_ckpt or not vae_ckpt:
        raise ValueError("Difix checkpoint has empty UNet or VAE weights")

    # The published mv_unet changes attention math, while retaining UNet keys.
    base_file = contract["sd_turbo"] / "unet/diffusion_pytorch_model.safetensors"
    with safe_open(str(base_file), framework="pt", device="cpu") as base:
        base_shapes = {key: tuple(base.get_slice(key).get_shape()) for key in base.keys()}
    checkpoint_shapes = {key: tuple(value.shape) for key, value in unet_ckpt.items()}
    mismatched_base = [key for key, shape in checkpoint_shapes.items() if base_shapes.get(key) != shape]
    if mismatched_base:
        raise ValueError(f"UNet checkpoint/base keys or shapes differ: {mismatched_base[:12]}")

    # A meta-device construction catches old constructor API drift while using
    # no real model-weight RAM and never executing a forward pass.
    config = json.loads((contract["sd_turbo"] / "unet/config.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        standard = UNet2DConditionModel.from_config(config)
        multi_view = mv_module.UNet2DConditionModel.from_config(config)
    standard_shapes = {key: tuple(value.shape) for key, value in standard.state_dict().items()}
    mv_shapes = {key: tuple(value.shape) for key, value in multi_view.state_dict().items()}
    if standard_shapes != base_shapes:
        raise ValueError("standard meta UNet key/shape set differs from local sd-turbo weights")
    if mv_shapes != standard_shapes:
        extra = sorted(set(mv_shapes) - set(standard_shapes))
        missing = sorted(set(standard_shapes) - set(mv_shapes))
        changed = sorted(k for k in set(mv_shapes) & set(standard_shapes) if mv_shapes[k] != standard_shapes[k])
        raise ValueError(f"official mv_unet meta key/shape mismatch extra={extra[:8]} missing={missing[:8]} changed={changed[:8]}")
    del standard, multi_view
    gc.collect()

    if not isinstance(checkpoint["rank_vae"], int) or checkpoint["rank_vae"] < 1:
        raise ValueError("invalid VAE LoRA rank in author checkpoint")
    if not checkpoint["vae_lora_target_modules"]:
        raise ValueError("empty VAE LoRA target list in author checkpoint")
    if not hasattr(mv_module, "new_forward") or not hasattr(mv_module, "UNet2DConditionModel"):
        raise ValueError("official mv_unet lacks two-view forward or model class")
    result = {
        "checkpoint_unet_key_count": len(checkpoint_shapes),
        "checkpoint_vae_key_count": len(vae_ckpt),
        "base_unet_key_count": len(base_shapes),
        "standard_meta_unet_key_count": len(standard_shapes),
        "mv_meta_unet_key_count": len(mv_shapes),
        "unet_checkpoint_base_shapes_match": True,
        "mv_and_standard_meta_key_shapes_equal": True,
        "vae_lora_rank": checkpoint["rank_vae"],
        "vae_lora_target_count": len(checkpoint["vae_lora_target_modules"]),
        "mv_unet_class_and_two_view_forward_imported": True,
        "mv_unet_model_instantiated_on_meta_only": True,
        "full_weight_materialization_and_strict_load_pending_gpu": True,
        "checkpoint_mmap_cpu": True,
    }
    del checkpoint
    gc.collect()
    return result


def cpu_check(contract: dict) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("--check-only requires CUDA_VISIBLE_DEVICES='' or -1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    import torch
    import diffusers

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mv_module, compatibility = import_official_mv(contract["mv_path"].parent, contract["sd_turbo"])
    before = {"cpu_max": _cgroup("cpu.max"), "memory_max": _cgroup("memory.max"),
              "memory_current": _cgroup("memory.current")}
    weights = _checkpoint_metadata(contract, torch, mv_module)
    return {
        "status": "cpu_check_only_passed", "gpu_executed": False,
        "frames_planned": {"official_public_single": 8, "official_mv_self": 8, "official_mv_legal_ref": 8},
        "maximum_output_frames": MAX_OUTPUT_FRAMES,
        "inputs": {"edit_size_wh": contract["size_wh"], "legal_reference_rgb": str(contract["reference_rgb"]),
                   "legal_reference_size_wh": contract["reference_size_wh"],
                   "checkpoint": str(contract["checkpoint"]), "prior": str(contract["prior"]),
                   "output_dir": str(contract["out"])},
        "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__},
        "resources": {"cpu_threads": 1, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                      "cgroup_before": before, "cgroup_after_memory_current": _cgroup("memory.current"),
                      "prior_single_peak_cuda_allocated_bytes": contract["prior_manifest"].get("peak_cuda_allocated_bytes"),
                      "mv_expected_memory": "Above the 6.32 GB single-view allocated peak is plausible; two-view attention can grow superlinearly. Exact peak unknown until GPU run. No precision/resolution fallback."},
        "compatibility": compatibility, "weights": weights,
        "human_verdict": None,
        "limits": ["This checks checkpoint/base UNet metadata and constructs both UNets on meta; real weight loading, VAE key-shape checks, and forward remain for the GPU phase.",
                   "The legal reference contains the original vehicle; later outputs require explicit reintroduction review."],
    }


def _load_sibling_refiner():
    path = Path(__file__).with_name("refine_dggt_waymo_edits.py")
    spec = importlib.util.spec_from_file_location("existing_dggt_refiner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mv_author_model(contract: dict, torch):
    src_dir = contract["mv_path"].parent
    _, compatibility = import_official_mv(src_dir, contract["sd_turbo"])
    source = contract["model_path"].read_text(encoding="utf-8")
    local_sd = str(contract["sd_turbo"])
    patched = source.replace('"stabilityai/sd-turbo"', json.dumps(local_sd))
    if patched == source or source.count('"stabilityai/sd-turbo"') != 5:
        raise RuntimeError("official sd-turbo load sites changed")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", DIFFUSERS_OFFLINE="1")
    if str(contract["external"]) not in sys.path:
        sys.path.insert(0, str(contract["external"]))
    name = "third_party.difix.src.model"
    module = types.ModuleType(name)
    module.__file__ = str(contract["model_path"])
    module.__package__ = "third_party.difix.src"
    sys.modules[name] = module
    try:
        exec(compile(patched, str(contract["model_path"]), "exec"), module.__dict__)
        model = module.Difix(pretrained_path=str(contract["checkpoint"]), timestep=TIMESTEP, mv_unet=True)
        model.set_eval()
    except Exception:
        sys.modules.pop(name, None)
        raise
    checkpoint = torch.load(contract["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    unet_shapes = {key: tuple(value.shape) for key, value in model.unet.state_dict().items()}
    vae_shapes = {key: tuple(value.shape) for key, value in model.vae.state_dict().items()}
    for ckpt_key, shapes in (("state_dict_unet", unet_shapes), ("state_dict_vae", vae_shapes)):
        bad = [key for key, value in checkpoint[ckpt_key].items() if shapes.get(key) != tuple(value.shape)]
        if bad:
            raise ValueError(f"loaded {ckpt_key} does not match all author keys/shapes: {bad[:12]}")
    check = {"mv_unet_keys": len(unet_shapes), "checkpoint_unet_keys": len(checkpoint["state_dict_unet"]),
             "checkpoint_vae_keys": len(checkpoint["state_dict_vae"]), "all_author_keys_shapes_match": True}
    del checkpoint
    return model, compatibility, check, unet_shapes


def _prepare_png(path: Path, transforms):
    from PIL import Image

    with Image.open(path) as image:
        tensor = transforms.ToTensor()(image.convert("RGB"))
    _, h, w = tensor.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    if (new_h, new_w) != (h, w):
        image = transforms.ToPILImage()(tensor).resize((new_w, new_h), Image.Resampling.LANCZOS)
        tensor = transforms.ToTensor()(image)
    return transforms.ToPILImage()(tensor.cpu()), (h, w), (new_h, new_w)


def _run_mv_one(input_pil, ref_pil, original_hw, resized_hw, model, torch, transforms, F):
    device = torch.device("cuda")
    model.sched.set_timesteps(1, device=device)
    model.sched.timesteps = torch.tensor([model.timesteps.item()], device=device)
    h, w = resized_hw
    output = model.sample(input_pil, width=w, height=h, ref_image=ref_pil, prompt=PROMPT)
    tensor = transforms.ToTensor()(output)
    if resized_hw != original_hw:
        tensor = F.interpolate(tensor.unsqueeze(0), size=original_hw, mode="bilinear", align_corners=False).squeeze(0)
    return transforms.ToPILImage()(tensor.cpu().clamp(0, 1))


def _capture_unet_inputs(model, torch):
    seen = []

    def hook(_module, args, kwargs):
        latent = args[0]
        timestep = args[1]
        text = kwargs["encoder_hidden_states"]
        if latent.shape[0] != 2 or text.shape[0] != 2:
            raise RuntimeError(f"expected two-view UNet batch, got {tuple(latent.shape)} and {tuple(text.shape)}")
        seen.append({"latent": latent.detach().cpu().clone(),
                     "latent_shape": list(latent.shape),
                     "timestep": timestep.detach().cpu().clone(),
                     "text": text.detach().cpu().clone()})

    handle = model.unet.register_forward_pre_hook(hook, with_kwargs=True)
    return seen, handle


def _image_diff(a, b) -> dict:
    import numpy as np

    aa, bb = np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))
    delta = np.abs(aa.astype(np.int16) - bb.astype(np.int16))
    return {"max_abs_u8": int(delta.max()), "mean_abs_u8": float(delta.mean()),
            "same_png_pixels": bool(np.array_equal(aa, bb))}


def _vehicle_mask(contract: dict, branch: str, index: int, size_wh: list[int]):
    from PIL import Image
    import numpy as np

    path = Path(contract["edit_manifest"]["selector_paths"][index])
    with Image.open(path) as image:
        if image.mode not in ("P", "L", "I", "I;16"):
            image = image.convert("L")
        target_h = round(image.height * (518 / image.width) / 14) * 14
        image = image.resize((518, target_h), Image.Resampling.NEAREST)
        if target_h > 518:
            top = (target_h - 518) // 2
            image = image.crop((0, top, 518, top + 518))
        arr = np.asarray(image) > 0
    if arr.shape != (size_wh[1], size_wh[0]):
        raise ValueError("vehicle mask differs from edit output size")
    return arr


def _masked_mae(a, b, mask) -> float:
    import numpy as np

    aa = np.asarray(a.convert("RGB"), dtype=np.float32)
    bb = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.abs(aa[mask] - bb[mask]).mean())


def _write_report(out: Path, report: dict) -> None:
    path = out / "run.json"
    temporary = out / ".run.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_paired_inputs(out: Path, branch: str, index: int, pair: dict, torch) -> Path:
    folder = out / "paired_inputs" / branch
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{index:03d}.pt"
    temporary = folder / f".{index:03d}.pt.tmp"
    torch.save(pair, temporary)
    os.replace(temporary, path)
    return path


def _prior_case(report: dict, section: str, branch: str, index: int) -> dict | None:
    matches = [item for item in report.get(section, {}).get(branch, []) if item.get("index") == index]
    if len(matches) > 1:
        raise ValueError(f"duplicate {section} cases for {branch}/{index}")
    return matches[0] if matches else None


def _normalize_seed_metadata(parameters: dict) -> dict:
    """Keep legacy partial-run recovery compatible with the actual seed rule."""
    normalized = dict(parameters)
    if "seed_per_frame" in normalized:
        legacy = normalized.pop("seed_per_frame")
        if "seed_base" in normalized and normalized["seed_base"] != legacy:
            raise ValueError("conflicting legacy seed metadata")
        normalized.setdefault("seed_base", legacy)
        normalized.setdefault("per_frame_rule", "seed_base + frame_index")
    return normalized


def run_gpu(contract: dict, args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("GPU phase requires CUDA; --check-only is the CPU path")
    if not args.resume:
        if contract["out"].exists():
            contract["out"].rmdir()  # only an empty caller-created directory is allowed
        contract["out"].mkdir(parents=True, exist_ok=False)
    out = contract["out"]
    report = {
        "status": "running", "gpu_executed": True,
        "method": "bounded_official_public_single_vs_two_view_reference_probe",
        "conditions": {
            "official_public_single": "Existing released infer.py path; mv_unet=False, no reference, same offline loader.",
            "official_mv_self": "Published inference_difix.py two-view path; reference is the rendered frame itself.",
            "official_mv_legal_ref": "Same two-view path; reference is existing original 000_0.jpg Waymo input containing the vehicle.",
        },
        "source": {"external": str(contract["external"]), "reference_interface": str(contract["public_ref_path"]),
                   "single_interface": str(contract["public_single_path"]), "checkpoint": str(contract["checkpoint"]),
                   "legal_reference_rgb": str(contract["reference_rgb"]), "edits": str(contract["edits"]),
                   "prior": str(contract["prior"]), "edit_frames": contract["edit_manifest"]["frames"],
                   "raw_pngs": {branch: [str(path) for path in contract["paths"][branch]["raw"]] for branch in BRANCHES},
                   "prior_pngs": {branch: [str(path) for path in contract["paths"][branch]["prior"]] for branch in BRANCHES}},
        "parameters": {"seed_base": args.seed, "per_frame_rule": "seed_base + frame_index",
                       "timestep": TIMESTEP, "prompt": PROMPT,
                       "width_height": contract["size_wh"], "precision": "FP32", "frames_per_condition": 8,
                       "max_output_frames": MAX_OUTPUT_FRAMES},
        "compatibility": {}, "weight_checks": {}, "baseline_replay": {}, "paired_checks": {},
        "baseline_replay_verified": False,
        "vehicle_reintroduction": {"reference_contains_original_vehicle": True,
                                   "human_verdict": None,
                                   "assessment": "pending visual review; masked RGB distances are diagnostics, not identity verdict"},
        "human_verdict": None,
        "limitations": ["PNG quantization precedes all paths, matching the prior offline edit refinement input.",
                        "This probe does not retrain the published checkpoint or change Gaussian edits.",
                        "Masked pixel similarity cannot prove object identity or scene-correct background restoration."],
    }
    if args.resume:
        prior_report = json.loads((out / "run.json").read_text(encoding="utf-8"))
        if prior_report.get("status") == "completed":
            raise ValueError("completed diagnostic must not be resumed")
        prior_parameters = _normalize_seed_metadata(prior_report.get("parameters", {}))
        if prior_report.get("source") != report["source"] or prior_parameters != report["parameters"]:
            raise ValueError("resume source or parameters differ from existing run.json")
        if prior_report.get("conditions") != report["conditions"]:
            raise ValueError("resume condition definitions differ from existing run.json")
        report = prior_report
        report["parameters"] = prior_parameters
        report.setdefault("attempt_history", []).append({
            "resumed_at_unix": time.time(), "previous_status": report.get("status"),
            "previous_error": report.get("error"),
        })
        report["status"] = "running"
        report.pop("error", None)
    for condition in CONDITIONS:
        for branch in BRANCHES:
            (out / condition / branch).mkdir(parents=True, exist_ok=args.resume)
    _write_report(out, report)
    try:
        refiner = _load_sibling_refiner()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        single_model, replacements = refiner._load_offline_author_model(contract["external"], contract["checkpoint"])
        if next(single_model.unet.parameters()).dtype != torch.float32 or next(single_model.vae.parameters()).dtype != torch.float32:
            raise RuntimeError("public single-view model is not FP32")
        report["compatibility"]["local_sd_turbo_literal_replacements"] = replacements
        report["model_load_seconds_single"] = time.perf_counter() - load_start
        single_shapes = {key: tuple(value.shape) for key, value in single_model.unet.state_dict().items()}
        with torch.no_grad():
            for branch in BRANCHES:
                report["baseline_replay"].setdefault(branch, [])
                for index, raw_path in enumerate(contract["paths"][branch]["raw"]):
                    previous = _prior_case(report, "baseline_replay", branch, index)
                    if previous is not None:
                        if not Path(previous["output"]).is_file():
                            raise FileNotFoundError(f"recorded baseline output missing: {previous['output']}")
                        continue
                    target = out / "official_public_single" / branch / f"{index:03d}.png"
                    if target.exists():
                        raise FileExistsError(f"unrecorded baseline output needs manual inspection: {target}")
                    torch.manual_seed(args.seed + index)
                    torch.cuda.manual_seed_all(args.seed + index)
                    image = refiner._refine_one(raw_path, single_model, torch, transforms, F)
                    image.save(target)
                    with Image.open(contract["paths"][branch]["prior"][index]) as old:
                        comparison = _image_diff(image, old)
                    comparison.update(index=index, source=str(raw_path), previous=str(contract["paths"][branch]["prior"][index]), output=str(target))
                    report["baseline_replay"][branch].append(comparison)
                    _write_report(out, report)
        report["baseline_replay_verified"] = all(
            item["same_png_pixels"] for branch in BRANCHES for item in report["baseline_replay"][branch]
        ) and sum(len(report["baseline_replay"][branch]) for branch in BRANCHES) == 8
        _write_report(out, report)
        if not report["baseline_replay_verified"]:
            raise RuntimeError("public single-view replay differs from prior PNG pixels; inspect the saved baseline evidence before two-view inference")
        torch.cuda.synchronize()
        report["peak_cuda_allocated_single_bytes"] = torch.cuda.max_memory_allocated()
        del single_model
        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        mv_model, alias_evidence, mv_weights, mv_shapes = _load_mv_author_model(contract, torch)
        if next(mv_model.unet.parameters()).dtype != torch.float32 or next(mv_model.vae.parameters()).dtype != torch.float32:
            raise RuntimeError("official multi-view model is not FP32")
        if mv_shapes != single_shapes:
            raise RuntimeError("mv_unet and single-view UNet parameter key/shape sets differ")
        report["model_load_seconds_mv"] = time.perf_counter() - load_start
        report["compatibility"]["mv_import"] = alias_evidence
        report["weight_checks"] = {**mv_weights, "mv_and_single_unet_key_shapes_equal": True}
        with Image.open(contract["reference_rgb"]) as reference_file:
            legal_reference = reference_file.convert("RGB").copy()
        with torch.no_grad():
            for branch in BRANCHES:
                report["paired_checks"].setdefault(branch, [])
                for index, raw_path in enumerate(contract["paths"][branch]["raw"]):
                    previous = _prior_case(report, "paired_checks", branch, index)
                    if previous is not None:
                        if not all(previous.get(key) is True for key in (
                                "same_batch_shape", "view0_vae_latent_torch_equal",
                                "text_embeddings_torch_equal", "timestep_torch_equal")):
                            raise RuntimeError(f"recorded paired input failure must be inspected: {branch}/{index}")
                        saved = [previous["paired_unet_inputs"], *previous["outputs"].values()]
                        if any(not Path(path).is_file() for path in saved):
                            raise FileNotFoundError(f"recorded paired evidence missing for {branch}/{index}: {saved}")
                        continue
                    input_pil, original_hw, resized_hw = _prepare_png(raw_path, transforms)
                    self_path = out / "official_mv_self" / branch / f"{index:03d}.png"
                    legal_path = out / "official_mv_legal_ref" / branch / f"{index:03d}.png"
                    trace_path = out / "paired_inputs" / branch / f"{index:03d}.pt"
                    orphaned = [path for path in (self_path, legal_path, trace_path) if path.exists()]
                    if orphaned:
                        raise FileExistsError(f"unrecorded paired evidence needs manual inspection: {orphaned}")
                    seen, handle = _capture_unet_inputs(mv_model, torch)
                    try:
                        torch.manual_seed(args.seed + index)
                        torch.cuda.manual_seed_all(args.seed + index)
                        self_image = _run_mv_one(input_pil, input_pil, original_hw, resized_hw,
                                                 mv_model, torch, transforms, F)
                        self_image.save(self_path)
                        torch.manual_seed(args.seed + index)
                        torch.cuda.manual_seed_all(args.seed + index)
                        legal_image = _run_mv_one(input_pil, legal_reference, original_hw, resized_hw,
                                                  mv_model, torch, transforms, F)
                        legal_image.save(legal_path)
                    finally:
                        handle.remove()
                    if len(seen) != 2:
                        raise RuntimeError(f"expected two UNet prehook captures, found {len(seen)}")
                    first, second = seen
                    trace_path = _save_paired_inputs(out, branch, index, {
                        "branch": branch, "frame_index": index, "seed": args.seed + index,
                        "raw_render": str(raw_path), "self_reference": str(raw_path),
                        "legal_reference": str(contract["reference_rgb"]),
                        "self": first, "legal": second,
                    }, torch)
                    equal_shape = first["latent_shape"] == second["latent_shape"]
                    latent_equal = torch.equal(first["latent"][0], second["latent"][0])
                    timestep_equal = torch.equal(first["timestep"], second["timestep"])
                    text_equal = torch.equal(first["text"], second["text"])
                    case = {
                        "index": index, "source": str(raw_path),
                        "outputs": {"official_mv_self": str(self_path), "official_mv_legal_ref": str(legal_path)},
                        "paired_unet_inputs": str(trace_path),
                        "same_batch_shape": equal_shape, "batch_shape": first["latent_shape"],
                        "view0_vae_latent_torch_equal": latent_equal,
                        "text_embeddings_torch_equal": text_equal,
                        "timestep_torch_equal": timestep_equal,
                        "prompt": PROMPT, "timestep": TIMESTEP,
                        "self_vs_legal": _image_diff(self_image, legal_image),
                    }
                    report["paired_checks"][branch].append(case)
                    _write_report(out, report)
                    if not all((equal_shape, latent_equal, timestep_equal, text_equal)):
                        raise RuntimeError(f"paired input contract failed for {branch}/{index}: "
                                           f"shape={equal_shape} latent={latent_equal} t={timestep_equal} text={text_equal}")
                    mask = _vehicle_mask(contract, branch, index, contract["size_wh"])
                    with Image.open(contract["paths"]["noop"]["raw"][index]) as noop_raw:
                        case["vehicle_mask_similarity"] = {
                            "self_mask_mae_to_raw_noop_u8": _masked_mae(self_image, noop_raw, mask),
                            "legal_mask_mae_to_raw_noop_u8": _masked_mae(legal_image, noop_raw, mask),
                        }
                    case["vehicle_mask_pixels"] = int(mask.sum())
                    _write_report(out, report)
        torch.cuda.synchronize()
        report["peak_cuda_allocated_mv_bytes"] = torch.cuda.max_memory_allocated()
        actual = sum(len(list((out / condition / branch).glob("*.png")))
                     for condition in CONDITIONS for branch in BRANCHES)
        if actual != MAX_OUTPUT_FRAMES:
            raise RuntimeError(f"expected exactly {MAX_OUTPUT_FRAMES} outputs, found {actual}")
        report["output_frames"] = actual
        report["status"] = "completed"
        return report
    except Exception as exc:
        report["status"] = "failed_gpu_probe"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        _write_report(out, report)


def main() -> int:
    args = parse_args()
    if args.check_report is not None and not args.check_only:
        raise ValueError("--check-report is only valid with --check-only")
    if args.resume and args.check_only:
        raise ValueError("--resume and --check-only are mutually exclusive")
    contract = inspect_inputs(args)
    if args.check_only:
        result = cpu_check(contract)
        if args.check_report is not None:
            report_path = args.check_report.resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    result = run_gpu(contract, args)
    print(json.dumps({"status": result["status"], "out": str(contract["out"]),
                      "output_frames": result["output_frames"],
                      "peak_cuda_allocated_mv_bytes": result["peak_cuda_allocated_mv_bytes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
