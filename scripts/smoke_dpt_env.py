"""Depth Anything V2 环境冒烟测试：加载 Large checkpoint，对一张 nuScenes 前视图做单图推理。

对应 DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md 8.1 第 6 步。
"""
import argparse
import glob
import hashlib
import os
import sys
import time

import cv2
import numpy as np
import torch

DPT_ROOT = os.environ.get("DPT_ROOT", "/root/autodl-tmp/third_party/Depth-Anything-V2")
CKPT = os.environ.get(
    "DPT_CKPT", "/root/autodl-tmp/checkpoints/depth_anything_v2/depth_anything_v2_vitl.pth")
sys.path.insert(0, DPT_ROOT)

MODEL_CONFIGS = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--encoder", default="vitl")
    parser.add_argument("--outdir", default="/root/autodl-tmp/motion_proj/tmp/dpt_smoke")
    args = parser.parse_args()

    print("python  ", sys.version.split()[0])
    print("torch   ", torch.__version__, "| cuda", torch.version.cuda)
    print("device  ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("opencv  ", cv2.__version__)

    image_path = args.image
    if image_path is None:
        candidates = sorted(glob.glob(
            "/root/autodl-tmp/data/nuscenes/samples/CAM_FRONT/*.jpg"))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(DPT_ROOT, "assets/examples/*.jpg")))
        image_path = candidates[0]
    print("image   ", image_path)

    print("ckpt    ", CKPT)
    print("ckpt sha256", sha256(CKPT))

    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True), None
    model = model.cuda().eval()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"model   {args.encoder} loaded | params {n_param/1e6:.1f}M")

    raw = cv2.imread(image_path)
    assert raw is not None, f"读不到图像 {image_path}"
    print("input   ", raw.shape)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t0 = time.time()
        depth = model.infer_image(raw)  # HxW, float32, 相对逆深度
        torch.cuda.synchronize()
        dt = time.time() - t0

    assert depth.shape == raw.shape[:2], f"深度图尺寸不匹配 {depth.shape} vs {raw.shape[:2]}"
    assert np.isfinite(depth).all(), "深度图含 NaN/Inf"
    assert depth.max() > depth.min(), "深度图是常数，模型没有真正推理"
    print(f"output   {depth.shape} dtype={depth.dtype} "
          f"range=[{depth.min():.3f}, {depth.max():.3f}] mean={depth.mean():.3f}")
    print(f"latency  {dt*1000:.0f} ms | peak GPU mem "
          f"{torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

    os.makedirs(args.outdir, exist_ok=True)
    norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    out_vis = os.path.join(args.outdir, "depth_vis.png")
    out_raw = os.path.join(args.outdir, "depth_raw.npy")
    cv2.imwrite(out_vis, np.concatenate([raw, color], axis=0))
    np.save(out_raw, depth.astype(np.float32))
    print("saved   ", out_vis)
    print("saved   ", out_raw)

    print("\nDPT SMOKE PASSED")


if __name__ == "__main__":
    main()
