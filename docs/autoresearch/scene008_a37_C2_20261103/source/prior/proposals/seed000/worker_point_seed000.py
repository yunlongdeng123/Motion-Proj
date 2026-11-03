"""One frozen Omni point-conditioned proposal; original evidence stays separate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_runtime_model(official_root, dino_root, output):
    runtime = output / "runtime_model"
    for name in ["model", "vae", "cond_encoder", "scheduler", "image_processor"]:
        target = runtime / name
        target.mkdir(parents=True, exist_ok=True)
        config = json.loads((official_root / name / "config.json").read_text())
        if name == "cond_encoder":
            config["params"]["image_encoder"]["params"]["version"] = str(dino_root)
        dump(target / "config.json", config)
        weight = official_root / name / "pytorch_model.bin"
        if weight.exists() and not (target / weight.name).exists():
            (target / weight.name).symlink_to(weight)
    shutil.copy2(official_root / "config.json", runtime / "config.json")
    return runtime


def surface_samples(vertices, faces, count, seed):
    import numpy as np
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise RuntimeError("Generated mesh has no finite positive-area surface")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(faces), size=count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    sample_tri = tri[ids]
    return sample_tri[:, 0] + uv[:, :1] * (sample_tri[:, 1] - sample_tri[:, 0]) + uv[:, 1:] * (sample_tri[:, 2] - sample_tri[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, default=Path("/root/autodl-tmp/models/c2_hunyuan_omni_70e803"))
    ap.add_argument("--dino-root", type=Path, default=Path("/root/autodl-tmp/models/c2_dinov2_large_47b73"))
    ap.add_argument("--official-repo", type=Path, default=Path("/root/autodl-tmp/third_party/hunyuan3d-omni-v61-me2"))
    ap.add_argument("--download-verification", type=Path, default=Path(__file__).with_name("download_verified.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=4.5)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--point-count", type=int, default=2048)
    ap.add_argument("--resume-latent", type=Path)
    ap.add_argument("--cpu-probe", action="store_true")
    ap.add_argument("--prepare-inputs-only", action="store_true")
    ns = ap.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if ns.cpu_probe or ns.prepare_inputs_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    overlay = Path(__file__).parent / "python_overlay"
    sys.path.insert(0, str(overlay))
    sys.path.insert(0, str(ns.official_repo))
    import numpy as np
    import torch
    from hy3dshape.models.conditioners.omni_encoder import OmniEncoder
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    from hy3dshape.models.autoencoders.shape_vae_pl import ShapeVAEPL
    from hy3dshape.models.diffusion.transport import create_transport, Sampler
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from hy3dshape.pipelines.utils import export_to_trimesh
    from hy3dshape.preprocessors import ImageProcessorV2
    from scipy.spatial import cKDTree
    from PIL import Image
    import trimesh

    ns.output.mkdir(parents=True, exist_ok=True)
    if ns.cpu_probe:
        scheduler = Sampler(create_transport(path_type="Linear", prediction="velocity"))
        image_processor = ImageProcessorV2()
        result = dict(torch=torch.__version__, cuda=torch.version.cuda,
                      classes_imported=[c.__name__ for c in [OmniEncoder, HunYuanDiTPlain, ShapeVAEPL,
                                                          Hunyuan3DOmniSiTFlowMatchingPipeline]],
                      scheduler_instantiated=type(scheduler).__name__,
                      image_processor_instantiated=type(image_processor).__name__,
                      overlay=str(overlay), gpu_initialized=torch.cuda.is_initialized())
        dump(ns.output / "cpu_probe.json", result)
        print(json.dumps(result), flush=True)
        return

    if not ns.prepare_inputs_only:
        verification = json.loads(ns.download_verification.read_text())
        if not verification.get("all_ok"):
            raise RuntimeError("Pinned prior downloads have not passed SHA256 verification")
        for row in verification["files"]:
            path = Path(row["path"])
            if not path.is_file() or path.stat().st_size != row["bytes"]:
                raise RuntimeError(f"Verified prior file is missing or changed: {path}")

    contract = json.loads(ns.contract.read_text(encoding="utf-8"))
    points_path = Path(contract["points_npz"])
    image_path = Path(contract["conditioning_image"])
    for key, path in [("points_sha256", points_path), ("conditioning_image_sha256", image_path)]:
        if key in contract and sha(path) != contract[key]:
            raise RuntimeError(f"Original observation contract hash mismatch: {key}")
    data = np.load(points_path, allow_pickle=False)
    point_key = contract.get("point_key", "observed_points_actor_native_m")
    size_key = contract.get("size_key", "size_lwh_m")
    points_native = np.asarray(data[point_key], dtype=np.float64)
    size_lwh = np.asarray(data[size_key] if size_key in data else contract["size_lwh_m"], dtype=np.float64)
    if points_native.ndim != 2 or points_native.shape[1] != 3 or not np.isfinite(points_native).all():
        raise RuntimeError("Expected finite original actor-local Nx3 LiDAR observations")
    half_span = float(size_lwh.max() / 2)
    if "normalization_half_span_m" in contract and abs(half_span - float(contract["normalization_half_span_m"])) > 1e-4:
        raise RuntimeError("Full actor box normalization disagrees with observation contract")
    points_omni = points_native[:, [0, 2, 1]] / half_span
    if not len(points_omni):
        raise RuntimeError("No valid observed a6 points")
    rng = np.random.default_rng(ns.seed)
    ids = rng.choice(len(points_omni), ns.point_count, replace=len(points_omni) < ns.point_count)
    conditioning = points_omni[ids].astype(np.float32)
    image = Image.open(image_path).convert("RGBA")
    true_visible_pixels = int((np.asarray(image)[..., 3] > 0).sum())
    if not true_visible_pixels:
        raise RuntimeError("Conditioning crop contains no original visible actor pixels")
    if "true_visible_pixel_count" in contract and true_visible_pixels != contract["true_visible_pixel_count"]:
        raise RuntimeError("Visible pixel count drifted from observation contract")
    np.savez_compressed(ns.output / "input_points_and_transform.npz", points_native_m=points_native,
                        size_lwh_m=size_lwh, omni_half_span_m=half_span, sampled_indices=ids,
                        conditioning_points_omni=conditioning)
    dump(ns.output / "input_contract_frozen.json", dict(contract=contract, seed=ns.seed,
         point_count=ns.point_count, observed_count=len(points_native), full_size_lwh_m=size_lwh.tolist(),
         half_span_m=half_span, axis_native_to_omni=[0, 2, 1], true_visible_pixels=true_visible_pixels,
         image_sha256=sha(image_path), points_sha256=sha(points_path), contract_sha256=sha(ns.contract),
         sampled_point_max_abs=float(np.max(np.abs(conditioning))), steps=ns.steps,
         guidance_scale=ns.guidance_scale, octree_resolution=ns.resolution))
    if ns.prepare_inputs_only:
        result = dict(observed_point_count=len(points_native), conditioning_point_count=len(conditioning),
                      size_lwh_m=size_lwh.tolist(), half_span_m=half_span,
                      visible_pixels=true_visible_pixels, input_sha256=sha(ns.contract),
                      sampled_point_max_abs=float(np.max(np.abs(conditioning))),
                      gpu_initialized=torch.cuda.is_initialized())
        dump(ns.output / "input_probe.json", result)
        print(json.dumps(result), flush=True)
        return

    started = time.monotonic()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(ns.seed)
    np.random.seed(ns.seed)
    runtime = make_runtime_model(ns.model_root, ns.dino_root, ns.output)
    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(str(runtime), fast_decode=False,
               device="cuda", dtype=torch.float16)
    for name in ["model", "vae", "cond_encoder"]:
        module = getattr(pipeline, name)
        module.requires_grad_(False)
        module.eval()
    trainable_prior_parameters = sum(p.numel() for name in ["model", "vae", "cond_encoder"]
                                     for p in getattr(pipeline, name).parameters() if p.requires_grad)
    if trainable_prior_parameters:
        raise RuntimeError("Pretrained proposal parameters must stay frozen")
    print("C2_OMNI_FROZEN_READY", flush=True)
    generation_started = time.monotonic()
    if ns.resume_latent:
        saved = torch.load(ns.resume_latent, map_location="cpu", weights_only=False)
        if saved["input_contract_sha256"] != sha(ns.contract):
            raise RuntimeError("Latent resume input contract mismatch")
        latents = saved["latents"].to(device=pipeline.device, dtype=pipeline.dtype)
        generated_steps = 0
    else:
        with torch.no_grad():
            result = pipeline(image=str(image_path), point=torch.from_numpy(conditioning[None]).to(
                device=pipeline.device, dtype=pipeline.dtype), num_inference_steps=ns.steps,
                guidance_scale=ns.guidance_scale, generator=torch.Generator("cuda").manual_seed(ns.seed),
                output_type="latent", fast_decode=False)
        latents = result["shapes"][0]
        if not torch.isfinite(latents).all():
            raise RuntimeError("Frozen prior produced nonfinite latent state")
        generated_steps = ns.steps
        torch.save(dict(latents=latents.detach().cpu(), input_contract_sha256=sha(ns.contract),
                        scale_factor=float(pipeline.scale_factor), seed=ns.seed, steps=ns.steps,
                        model_revision="70e803bfb4e127d534049d8ab8c8cb511780d485",
                        dino_revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                        verification_sha256=sha(ns.download_verification)),
                   ns.output / "candidate_latents.pt")
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - generation_started
    decode_started = time.monotonic()
    with torch.no_grad():
        decoded = pipeline.vae.decode((latents / float(pipeline.scale_factor)).to(
            next(pipeline.vae.parameters()).dtype), octree_depth=8,
            bounds=[-1.01] * 3 + [1.01] * 3, mc_level=0.0, num_chunks=8000,
            octree_resolution=ns.resolution, mc_mode="mc")
        meshes = export_to_trimesh(decoded)
    if len(meshes) != 1 or meshes[0] is None:
        raise RuntimeError("Expected a single finite learned prior mesh")
    mesh = meshes[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise RuntimeError("Learned prior mesh is empty or nonfinite")
    mesh.export(ns.output / "candidate_omni.glb")
    vertices_native = vertices[:, [0, 2, 1]] * half_span
    # Axis swap has determinant -1: reverse winding to preserve outward normals.
    mesh_native = trimesh.Trimesh(vertices=vertices_native, faces=faces[:, [0, 2, 1]], process=False)
    mesh_native.export(ns.output / "candidate_actor_native.ply")
    samples = surface_samples(vertices_native, np.asarray(mesh_native.faces), 65536, ns.seed)
    distances, _ = cKDTree(samples).query(points_native, k=1, workers=4)
    np.savez_compressed(ns.output / "candidate_surface_samples.npz", points_actor_native_m=samples,
                        observed_to_proposal_distance_m=distances)
    torch.cuda.synchronize()
    report = dict(schema_version="c2.frozen_omni_proposal.v1", prior_type="pretrained_learned_3d_shape",
                  model_repo="tencent/Hunyuan3D-Omni",
                  model_revision="70e803bfb4e127d534049d8ab8c8cb511780d485",
                  dino_revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                  worker_sha256=sha(__file__), verification_sha256=sha(ns.download_verification),
                  seed=ns.seed, new_diffusion_steps=generated_steps, training_steps=0,
                  frozen=True, trainable_prior_parameters=trainable_prior_parameters,
                  control="point", point_count=ns.point_count,
                  observed_count=len(points_native), vertices=len(vertices), faces=len(faces),
                  mesh_bounds_native_m=mesh_native.bounds.tolist(),
                  observed_to_surface_approx_m=dict(mean=float(distances.mean()),
                    p50=float(np.quantile(distances,.5)),p90=float(np.quantile(distances,.9)),
                    p99=float(np.quantile(distances,.99))),
                  generation_seconds=generation_seconds, decode_seconds=time.monotonic()-decode_started,
                  total_wall_seconds=time.monotonic()-started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,cuda=torch.version.cuda,
                  input_contract_sha256=sha(ns.contract), output=str(ns.output),
                  quality_status="proposal only; original RGB/LiDAR optimization and matched renders remain required",
                  resumed_from=str(ns.resume_latent) if ns.resume_latent else None)
    dump(ns.output / "proposal_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
