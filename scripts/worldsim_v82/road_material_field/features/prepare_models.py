#!/usr/bin/env python3
"""核验原生冻结特征权重，并只读登记 TRELLIS.2 纹理化前置。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def import_superpoint():
    path = Path(__file__).parent / "vendor" / "superpoint.py"
    spec = importlib.util.spec_from_file_location("official_superpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SuperPoint


def road_image(path: str):
    return Image.open(path).convert("RGB"), str(path)


def normalized_rgb(image, size=(224, 224)):
    arr = np.asarray(image.resize(size, Image.Resampling.BILINEAR)).copy()
    x = torch.from_numpy(arr).permute(2, 0, 1).float()[None] / 255
    return (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    image, image_path = road_image(args.image)
    import transformers
    from transformers import AutoModel

    started = time.monotonic()
    model = AutoModel.from_pretrained(args.dino, local_files_only=True).eval().requires_grad_(False).to(args.device)
    x = normalized_rgb(image).to(args.device)
    with torch.inference_mode():
        output = model(pixel_values=x).last_hidden_state
    skip = 1 + model.config.num_register_tokens
    patches = output[:, skip:]
    assert patches.shape == (1, 196, 768), patches.shape
    assert torch.isfinite(patches).all()
    assert not any(p.requires_grad for p in model.parameters())
    dino = {
        "status": "native_weights_loaded_real_image_forward_passed",
        "model_type": model.config.model_type,
        "path": args.dino,
        "safetensors_bytes": (Path(args.dino) / "model.safetensors").stat().st_size,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "frozen": True,
        "parameter_updates": 0,
        "device": args.device,
        "input_hw": list(x.shape[-2:]),
        "output_all_tokens_shape": list(output.shape),
        "output_patch_tokens_shape": list(patches.shape),
        "finite": True,
        "patch_l2_mean": patches.norm(dim=-1).mean().item(),
        "seconds_load_and_forward": time.monotonic() - started,
        "runtime_python": sys.executable,
        "transformers": transformers.__version__,
        "preprocessing": {"mean": [.485,.456,.406], "std": [.229,.224,.225], "patch_size":16, "exclude_cls_and_register_tokens":skip},
        "official_model_origin": "https://github.com/facebookresearch/dinov3",
    }
    del model, output, patches, x
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    started = time.monotonic()
    SuperPoint = import_superpoint()
    sp = SuperPoint({"max_keypoints": 512, "weights_path":args.superpoint_weights}).eval().requires_grad_(False).to(args.device)
    gray = np.asarray(image.convert("L").resize((320, 208), Image.Resampling.BILINEAR)).copy()
    gray = torch.from_numpy(gray).float()[None, None].to(args.device) / 255
    with torch.inference_mode():
        pred = sp({"image": gray})
    desc = pred["descriptors"][0]
    assert torch.isfinite(desc).all()
    assert not any(p.requires_grad for p in sp.parameters())
    superpoint = {
        "status": "official_weights_loaded_real_image_forward_passed",
        "source": "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superpoint.py",
        "weights_source": "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superpoint_v1.pth",
        "weights_path": str(args.superpoint_weights),
        "weights_bytes": Path(args.superpoint_weights).stat().st_size,
        "frozen": True, "parameter_updates": 0, "device": args.device,
        "parameter_count": sum(p.numel() for p in sp.parameters()),
        "input_hw": list(gray.shape[-2:]),
        "keypoints": len(pred["keypoints"][0]),
        "descriptor_shape": list(desc.shape),
        "finite": True,
        "seconds_load_and_forward": time.monotonic() - started,
        "max_keypoints": 512, "nms_radius": 4, "keypoint_threshold": .005,
    }

    probes = {}
    for name in ["trellis2", "o_voxel", "cumesh", "nvdiffrast", "flex_gemm", "flash_attn", "trimesh"]:
        try:
            found = importlib.util.find_spec(name)
            probes[name] = bool(found)
        except (ImportError, ModuleNotFoundError, ValueError):
            probes[name] = False
    # 官方 preprocess_mesh: (p-center)*scale, 再 [x,-z,y]；postprocess 撤销轴旋转。
    vertices = np.array([[10., -4., 2.], [15.12, -4., 2.], [10., 1.12, 2.], [15.12, 1.12, 2.]])
    center = (vertices.min(0) + vertices.max(0)) / 2
    scale = .99999 / np.ptp(vertices, axis=0).max()
    normalized = (vertices - center) * scale
    internal = normalized[:, [0,2,1]] * np.array([1,-1,1])
    output_axes = internal[:, [0,2,1]] * np.array([1,1,-1])
    restored = output_axes / scale + center
    error = np.abs(restored - vertices).max().item()
    assert error < 1e-12
    trellis = {
        "status": "read_only_preflight_not_executed",
        "official_source": "https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_texturing.py",
        "official_config": "https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/texturing_pipeline.json",
        "pipeline": "Trellis2TexturingPipeline",
        "config_file": "texturing_pipeline.json",
        "module_probes_in_this_feature_runtime": probes,
        "required_models": ["shape_slat_encoder", "tex_slat_decoder", "tex_slat_flow_model_512", "tex_slat_flow_model_1024"],
        "additional_models": ["facebook/dinov3-vitl16-pretrain-lvd1689m", "briaai/RMBG-2.0"],
        "input": "metric road mesh + legal masked RGB appearance prompt; geometry stays fixed",
        "preprocess_image": False,
        "preprocess_image_reason": "官方默认会去背景并裁物体，背景道路需使用明确构造的合法RGB条件",
        "mesh_transform": {"center_world_m":center.tolist(),"scale_normalized_per_m":float(scale),"preprocess_axes":"[x,-z,y]", "postprocess_axes":"[x,z,-y]", "output_to_world":"vertices_world = vertices_output / scale + center", "roundtrip_max_abs_m":error},
        "resolution": 512, "texture_size": 1024, "seed": 42,
        "sampler": {"steps":12,"guidance_strength":1.0,"guidance_rescale":0.0,"guidance_interval":[.6,.9],"rescale_t":3.0},
        "budget_contract": {"exploration_samples":1, "gpu_seconds_cap_if_later_run":600, "training_steps":0, "current_large_model_downloads":0, "current_gpu_inference_seconds":0},
        "requirements": "Linux, NVIDIA >=24GB, compiler CUDA toolkit (official recommends12.4); native CUDA dependencies",
        "road_adaptation_boundary": "官方生成PBR纹理不提供观测像素硬约束；必须保留visible材质并仅将生成结果写入hole，不能把世界几何重建称为完成",
    }
    receipt = {"run_id":args.run_id, "created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), "image_path": image_path,"image_native_size":image.size,"torch":torch.__version__,"platform":platform.platform(),"dino_v3":dino,"superpoint":superpoint,"trellis2":trellis}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino", default=os.environ.get("DINO_V3_MODEL"), required=not os.environ.get("DINO_V3_MODEL"))
    parser.add_argument("--superpoint-weights", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", default="model_preflight")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", default="model_preparation_receipt.json")
    run(parser.parse_args())
