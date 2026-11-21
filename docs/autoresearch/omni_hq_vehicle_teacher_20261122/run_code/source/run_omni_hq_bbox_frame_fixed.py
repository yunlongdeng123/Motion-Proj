"""唯一工程重放：将已知车辆 LWH 映射为独立确认的 raw XYZ 框轴。"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

RUN_ID = "omni_hq_vehicle_teacher_20261122_r1"
OFFICIAL = Path("/root/autodl-tmp/third_party/hunyuan3d-omni-v61-me2")
MODEL = Path("/root/autodl-tmp/models/c2_hunyuan_omni_70e803")
DINO = Path("/root/autodl-tmp/models/c2_dinov2_large_47b73")
OVERLAY = Path("/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1/source/prior/python_overlay")
VERIFICATION = Path("/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1/source/prior/download_verified.json")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def imports(cpu=False):
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(OVERLAY))
    sys.path.insert(0, str(OFFICIAL))
    import numpy as np
    import torch
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from hy3dshape.preprocessors import ImageProcessorV2
    torch.set_num_threads(2)
    return np, torch, Image, Hunyuan3DOmniSiTFlowMatchingPipeline, ImageProcessorV2


def interface_probe(output):
    np, torch, Image, Pipeline, Processor = imports(cpu=True)
    processor = Processor()
    pipeline = Pipeline(image_processor=processor, vae=SimpleNamespace(latent_shape=(4, 8)),
                        scheduler=SimpleNamespace(), device="cpu", dtype=torch.float32)
    a = np.zeros((72, 96, 4), np.uint8)
    a[12:60, 12:84] = [100, 120, 150, 255]
    b = a.copy(); b[..., :3] = b[..., :3] // 2
    one, _ = pipeline.prepare_image([Image.fromarray(a)], None)
    two, _ = pipeline.prepare_image([Image.fromarray(a), Image.fromarray(b)], None)
    latent = pipeline.prepare_latents(None, two.shape[0], torch.float32, torch.device("cpu"),
                                     torch.Generator("cpu").manual_seed(101))
    source = inspect.getsource(Pipeline)
    assert tuple(one.shape) == (1, 3, 512, 512)
    assert tuple(two.shape) == (2, 3, 512, 512)
    assert tuple(latent.shape) == (2, 4, 8)
    assert "batch_size = image.shape[0]" in source
    report = dict(schema="omni_hq.interface_probe.v1", run_id=RUN_ID,
        official_repo=str(OFFICIAL), official_commit=subprocess.check_output(
            ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip(),
        official_web_source="https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/blob/main/hy3dshape/pipelines/pipeline_generation_sit_omni.py",
        single_image_tensor_shape=list(one.shape), list_two_image_tensor_shape=list(two.shape),
        prepare_latents_two_batch_shape=list(latent.shape),
        native_joint_multiview_supported=False, list_semantics="independent samples in batch dimension 0",
        decision="single rear-three-quarter crop; second image is reference only",
        image_processor_class=type(processor).__name__, torch=torch.__version__,
        cuda_initialized=torch.cuda.is_initialized(), cpu_threads=torch.get_num_threads(),
        bbox_semantics="(B,1,3) normalized [length,height,width], official encoder lines 407-451",
        fusion_added=False, model_weights_loaded=False)
    dump(output / "interface_probe.json", report)
    (output / "official_prepare_image_source.txt").write_text(inspect.getsource(Pipeline.prepare_image))
    (output / "official_pipeline_call_source.txt").write_text(inspect.getsource(Pipeline.__call__))
    print(json.dumps(report), flush=True)


def runtime_model(output):
    runtime = output / "runtime_model"
    for name in ["model", "vae", "cond_encoder", "scheduler", "image_processor"]:
        target = runtime / name; target.mkdir(parents=True, exist_ok=True)
        config = json.loads((MODEL / name / "config.json").read_text())
        if name == "cond_encoder":
            config["params"]["image_encoder"]["params"]["version"] = str(DINO)
        dump(target / "config.json", config)
        weight = MODEL / name / "pytorch_model.bin"
        if weight.exists() and not (target / weight.name).exists():
            (target / weight.name).symlink_to(weight)
    shutil.copy2(MODEL / "config.json", runtime / "config.json")
    return runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--contract", type=Path)
    ap.add_argument("--interface-probe", action="store_true")
    ap.add_argument("--prepare-inputs-only", action="store_true")
    ap.add_argument("--resume-latent", type=Path)
    ns = ap.parse_args(); ns.output.mkdir(parents=True, exist_ok=True)
    if ns.interface_probe:
        return interface_probe(ns.output)
    if ns.contract is None:
        raise ValueError("--contract is required")
    np, torch, Image, Pipeline, Processor = imports(cpu=ns.prepare_inputs_only)
    contract = json.loads(ns.contract.read_text(encoding="utf-8-sig"))
    if contract["run_id"] != RUN_ID:
        raise ValueError("Wrong run ID")
    image_path = Path(contract["conditioning_image"])
    if sha(image_path) != contract["conditioning_image_sha256"]:
        raise ValueError("Conditioning image changed")
    if contract["input_role"] != "high_quality_vehicle_class_reference_not_a20_truth":
        raise ValueError("Reference image role must be explicit")
    if contract["view_role"] != "rear_three_quarter_single_primary":
        raise ValueError("Only frozen single rear-three-quarter primary is authorized")
    size_lwh = np.asarray(contract["declared_size_lwh_m"], float)
    if size_lwh.shape != (3,) or not np.allclose(size_lwh, [5., 2.5, 2.08], atol=1e-7):
        raise ValueError("Known a20 whole-box LWH changed")
    mapping = contract["actor_to_raw_bbox_mapping"]
    if mapping["actor_size_indices_for_raw_xyz"] != [1, 2, 0] or contract["candidate_id"] != "bbox_frame_fixed":
        raise ValueError("Expected the single frozen coordinate correction")
    # 首候选独立灰模确认 raw -Z 为车头、+Y 为上；仅修正输入框的轴顺序。
    bbox_lhw = (size_lwh[[1, 2, 0]] / size_lwh.max()).astype(np.float32)
    image = Image.open(image_path).convert("RGBA")
    alpha = np.asarray(image)[..., 3]
    if not (alpha > 0).any():
        raise ValueError("Empty image")
    processor = Processor()
    prepared, mask = Pipeline(image_processor=processor, device="cpu", dtype=torch.float32).prepare_image(str(image_path), None)
    if tuple(prepared.shape) != (1, 3, 512, 512):
        raise ValueError("Expected exactly one official image sample")
    preview = ((prepared[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(preview).save(ns.output / "official_preprocessed_condition.png")
    frozen = dict(schema="omni_hq.input_frozen.v1", contract=contract, seed=101, steps=50,
        octree_resolution=256, guidance_scale=4.5, official_bbox_lhw=bbox_lhw.tolist(),
        official_bbox_shape=[1, 1, 3], official_bbox_definition="ordered raw XYZ extents = [actor W,actor H,actor L] / max(LWH)",
        official_bbox_raw_xyz=bbox_lhw.tolist(), actor_to_raw_bbox_mapping=mapping,
        official_preprocessed_shape=list(prepared.shape), conditioning_image_sha256=sha(image_path),
        model_image_count=1, point_condition_used=False, lidar_read_during_build=False,
        holdout_read=False, second_reference_image_passed_to_model=False,
        output_coordinate_contract="raw normalized Omni; front/longitudinal axis not assumed",
        input_contract_sha256=sha(ns.contract), worker_sha256=sha(__file__))
    dump(ns.output / "input_contract_frozen.json", frozen)
    if ns.prepare_inputs_only:
        print(json.dumps(frozen), flush=True); return
    if (ns.output / "candidate_latents.pt").exists() and ns.resume_latent is None:
        raise RuntimeError("Latent already exists; second stochastic sampling is prohibited")
    verification = json.loads(VERIFICATION.read_text())
    if not verification.get("all_ok"):
        raise RuntimeError("Pinned model verification is invalid")
    for row in verification["files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Pinned model file missing or changed: {path}")
    free_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
    if int(free_rows.splitlines()[0]) < 18 * 1024 or compute:
        raise RuntimeError(f"GPU preflight failed: free={free_rows!r}, active={compute!r}")
    dump(ns.output / "gpu_preflight.json", dict(free_mib=int(free_rows.splitlines()[0]), compute_apps=compute,
         required_free_mib=18*1024, cuda_memory_fraction=.45, cpu_threads=2))
    started = time.monotonic(); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.45, device=0); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(101); np.random.seed(101); random.seed(101)
    pipeline = Pipeline.from_pretrained(str(runtime_model(ns.output)), fast_decode=False, device="cuda", dtype=torch.float16)
    for name in ["model", "vae", "cond_encoder"]:
        getattr(pipeline, name).requires_grad_(False); getattr(pipeline, name).eval()
    audit = {}; original_encode = pipeline.encode_cond
    def audited_encode(*args, **kwargs):
        bound = inspect.signature(original_encode).bind(*args, **kwargs).arguments
        if bound.get("point") is not None or tuple(bound["bbox"].shape) != (1, 1, 3):
            raise RuntimeError("Official single-image bbox-only contract violated")
        encoded, sampled = original_encode(*args, **kwargs)
        tokens = encoded["cond"]; count = pipeline.cond_encoder.image_encoder.patch_nums
        if tokens.shape[0] != 2 or tokens.shape[1] - count != 11:
            raise RuntimeError("Wrong official CFG batch/token layout")
        image_zero = float(tokens[1:, :count].abs().max().item())
        image_abs = float(tokens[:1, :count].abs().max().item())
        bbox_diff = float((tokens[:1, count:] - tokens[1:, count:]).abs().max().item())
        if image_zero != 0 or image_abs <= 0 or bbox_diff != 0 or not torch.isfinite(tokens).all():
            raise RuntimeError("Nonfinite or invalid official image+bbox CFG")
        audit.update(image_tokens=count, bbox_and_type_tokens=11, model_image_count=1,
            conditional_image_max_abs=image_abs, unconditional_image_max_abs=image_zero,
            conditional_unconditional_bbox_max_diff=bbox_diff,
            actual_bbox_lhw=bound["bbox"].detach().float().cpu().tolist(),
            sampled_bbox_corners=sampled.detach().float().cpu().tolist(), point_condition_used=False)
        dump(ns.output / "actual_image_bbox_mechanism.json", audit)
        print("OFFICIAL_IMAGE_BBOX_AUDIT " + json.dumps(audit), flush=True)
        return encoded, sampled
    pipeline.encode_cond = audited_encode
    forward_audit = dict(calls=0, all_finite=True)
    def audit_forward(module, inputs, result):
        forward_audit["calls"] += 1
        if not torch.isfinite(result).all():
            forward_audit["all_finite"] = False; raise RuntimeError("Nonfinite shape model output")
    pipeline.model.register_forward_hook(audit_forward)
    generation_started = time.monotonic()
    if ns.resume_latent:
        saved = torch.load(ns.resume_latent, map_location="cpu", weights_only=False)
        if saved["input_contract_sha256"] != sha(ns.contract):
            raise RuntimeError("Resume contract mismatch")
        latents = saved["latents"].to(device=pipeline.device, dtype=pipeline.dtype); generated_steps = 0
    else:
        with torch.no_grad():
            result = pipeline(image=str(image_path), bbox=torch.from_numpy(bbox_lhw[None, None]).to(
                device=pipeline.device, dtype=pipeline.dtype), num_inference_steps=50, guidance_scale=4.5,
                generator=torch.Generator("cuda").manual_seed(101), output_type="latent", fast_decode=False)
        latents = result["shapes"][0]; generated_steps = 50
        if tuple(latents.shape[:1]) != (1,) or not torch.isfinite(latents).all():
            raise RuntimeError("Expected one finite latent")
        torch.save(dict(latents=latents.detach().cpu(), input_contract_sha256=sha(ns.contract),
            seed=101, steps=50, guidance_scale=4.5, scale_factor=float(pipeline.scale_factor),
            official_bbox_lhw=bbox_lhw.tolist(), input_role=contract["input_role"],
            model_revision="70e803bfb4e127d534049d8ab8c8cb511780d485"), ns.output / "candidate_latents.pt")
    torch.cuda.synchronize(); generation_seconds = time.monotonic() - generation_started
    decode_started = time.monotonic()
    from hy3dshape.pipelines.utils import export_to_trimesh
    with torch.no_grad():
        decoded = pipeline.vae.decode((latents / float(pipeline.scale_factor)).to(next(pipeline.vae.parameters()).dtype),
            octree_depth=8, bounds=[-1.01]*3+[1.01]*3, mc_level=0., num_chunks=8000, octree_resolution=256, mc_mode="mc")
        meshes = export_to_trimesh(decoded)
    if len(meshes) != 1 or meshes[0] is None:
        raise RuntimeError("Expected one decoded mesh")
    mesh = meshes[0]; vertices = np.asarray(mesh.vertices); faces = np.asarray(mesh.faces)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise RuntimeError("Empty or nonfinite raw mesh")
    for ext in ["glb", "ply", "obj"]:
        mesh.export(ns.output / ("candidate_raw_omni." + ext))
    np.savez_compressed(ns.output / "candidate_raw_geometry.npz", vertices=vertices, faces=faces)
    torch.cuda.synchronize()
    report = dict(schema="omni_hq.proposal_report.v1", run_id=RUN_ID,
        model_repo="tencent/Hunyuan3D-Omni", model_revision="70e803bfb4e127d534049d8ab8c8cb511780d485",
        dino_revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", seed=101, steps=50,
        new_sampling_steps=generated_steps, octree_resolution=256, guidance_scale=4.5, training_steps=0,
        model_image_count=1, input_role=contract["input_role"], official_bbox_lhw=bbox_lhw.tolist(),
        candidate_id="bbox_frame_fixed", candidate_role=contract["candidate_role"],
        official_bbox_raw_xyz=bbox_lhw.tolist(), actor_to_raw_bbox_mapping=mapping,
        bbox_mechanism=audit, forward_audit=forward_audit, point_condition_used=False, holdout_read=False,
        lidar_read_during_build=False, raw_bounds_xyz=mesh.bounds.tolist(), raw_span_xyz=np.ptp(vertices, axis=0).tolist(),
        vertices=len(vertices), faces=len(faces), raw_mesh_modified=False, actor_frame_mapping_applied=False,
        heading_semantics="pending independent review of raw mesh views", quality_status="raw grey-mesh proposal; review pending",
        generation_seconds=generation_seconds, decode_seconds=time.monotonic()-decode_started,
        total_wall_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, resource_limits=dict(cuda_fraction=.45, cpu_threads=2),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        worker_sha256=sha(__file__), input_contract_sha256=sha(ns.contract), conditioning_image_sha256=sha(image_path),
        resumed_from=str(ns.resume_latent) if ns.resume_latent else None)
    dump(ns.output / "proposal_report.json", report); print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
