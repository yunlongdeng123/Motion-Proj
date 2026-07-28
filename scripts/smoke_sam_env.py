"""Grounded-SAM-2 环境冒烟测试：GroundingDINO 文本检测 + SAM2 分割，跑一张 nuScenes 前视图。

对应 DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md 8.1 第 6 步。
必须在 Grounded-SAM-2 仓库根目录下运行（hydra config 走相对路径）。
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

SAM_ROOT = os.environ.get("SAM_ROOT", "/root/autodl-tmp/third_party/Grounded-SAM-2")
SAM2_CKPT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_CFG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# 动态驾驶重建关注的可动类别
TEXT_PROMPT = "car. truck. bus. pedestrian. bicycle. motorcycle."


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
    parser.add_argument("--outdir", default="/root/autodl-tmp/motion_proj/tmp/sam_smoke")
    args = parser.parse_args()

    # 官方 demo 以 `grounding_dino.groundingdino.*` 形式导入，依赖仓库根目录在 sys.path 中
    os.chdir(SAM_ROOT)
    sys.path.insert(0, SAM_ROOT)
    print("python  ", sys.version.split()[0])
    print("torch   ", torch.__version__, "| cuda", torch.version.cuda)
    print("device  ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from torchvision.ops import box_convert

    try:
        import sam2._C  # noqa: F401
        print("sam2 CUDA 扩展 (_C)  已编译可用")
    except Exception as exc:
        print("sam2 CUDA 扩展 (_C)  不可用 ->", type(exc).__name__, exc)
    try:
        from grounding_dino.groundingdino import _C  # noqa: F401
        print("groundingdino CUDA 扩展 (_C)  已编译可用")
    except Exception as exc:
        print("groundingdino CUDA 扩展 (_C)  不可用 ->", type(exc).__name__, exc)

    image_path = args.image
    if image_path is None:
        candidates = sorted(glob.glob(
            "/root/autodl-tmp/data/nuscenes/samples/CAM_FRONT/*.jpg"))
        if not candidates:
            candidates = [os.path.join(SAM_ROOT, "notebooks/images/truck.jpg")]
        image_path = candidates[0]
    print("image   ", image_path)
    print("sam2 ckpt sha256 ", sha256(SAM2_CKPT))
    print("gdino ckpt sha256", sha256(GDINO_CKPT))

    device = "cuda"
    sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(model_config_path=GDINO_CFG,
                                 model_checkpoint_path=GDINO_CKPT, device=device)
    print("models loaded")

    image_source, image = load_image(image_path)
    sam2_predictor.set_image(image_source)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    boxes, confidences, labels = predict(
        model=grounding_model, image=image, caption=TEXT_PROMPT,
        box_threshold=0.35, text_threshold=0.25, device=device)
    torch.cuda.synchronize()
    t_dino = time.time() - t0

    h, w, _ = image_source.shape
    print(f"input    {image_source.shape}")
    print(f"prompt   '{TEXT_PROMPT}'")
    print(f"gdino    {len(boxes)} boxes in {t_dino*1000:.0f} ms | labels {labels}")
    assert len(boxes) > 0, "GroundingDINO 在一张典型驾驶图上没有检出任何目标"

    boxes_px = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None, point_labels=None,
            box=input_boxes, multimask_output=False)
    torch.cuda.synchronize()
    t_sam = time.time() - t0

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    print(f"sam2     masks {masks.shape} in {t_sam*1000:.0f} ms")
    assert masks.shape[0] == len(boxes), "mask 数量与 box 数量不一致"
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    assert (areas > 0).all(), "存在空 mask，SAM2 没有真正分割"
    print(f"mask 面积占比 {[round(float(a)/(h*w), 4) for a in areas]}")
    print(f"peak GPU mem {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

    os.makedirs(args.outdir, exist_ok=True)
    vis = image_source.copy()[:, :, ::-1].copy()
    rng = np.random.default_rng(0)
    for i, (mask, box, label, conf) in enumerate(
            zip(masks, input_boxes, labels, confidences)):
        color = rng.integers(64, 255, size=3).tolist()
        vis[mask.astype(bool)] = (0.5 * vis[mask.astype(bool)] +
                                  0.5 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{label} {float(conf):.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out = os.path.join(args.outdir, "grounded_sam2_vis.png")
    cv2.imwrite(out, vis)
    print("saved   ", out)

    print("\nSAM SMOKE PASSED")


if __name__ == "__main__":
    main()
