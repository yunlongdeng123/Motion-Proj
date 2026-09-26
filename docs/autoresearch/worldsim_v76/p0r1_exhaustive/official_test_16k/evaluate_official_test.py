"""使用未改动的相机 0–4 时间切分、is_val 和 actor 插值评估。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0], "--config", args.config, "mode", "evaluate"]
    from lib.config import cfg
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer
    from lib.utils.loss_utils import psnr, ssim
    from lib.utils.lpipsPyTorch.modules.lpips import LPIPS

    torch.set_num_threads(2)
    cfg.loaded_iter = args.iteration
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    assert scene.loaded_iter == args.iteration
    cameras = scene.getTestCameras()
    train_keys = {(int(c.meta["cam"]), int(c.meta["frame"])) for c in scene.getTrainCameras()}
    keys = {(int(c.meta["cam"]), int(c.meta["frame"])) for c in cameras}
    assert not (keys & train_keys)
    assert len(keys) == len(cameras) == 75
    assert keys == {(c, f) for c in range(5) for f in range(4, 61, 4)}
    assert all(c.meta["is_val"] for c in cameras)
    renderer = StreetGaussianRenderer()
    lpips_metric = LPIPS("alex", "0.1").cuda().eval()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for camera in cameras:
            result = renderer.render(camera, gaussians)
            for name in ("rgb", "depth", "acc"):
                assert torch.isfinite(result[name]).all(), (camera.image_name, name, "nonfinite renderer output")
            rgb = result["rgb"].clamp(0, 1)
            gt = camera.original_image.cuda()
            assert rgb.shape == gt.shape
            # 官方 render.py 保存 PNG，metrics.py 从 PNG 读取；保留相同 8-bit 量化。
            rgb8 = (rgb * 255).round().to(torch.uint8)
            gt8 = (gt * 255).round().to(torch.uint8)
            pred, target = rgb8.float()/255, gt8.float()/255
            Image.fromarray(rgb8.permute(1,2,0).cpu().numpy()).save(out/f"{camera.image_name}_rgb.png")
            Image.fromarray(gt8.permute(1,2,0).cpu().numpy()).save(out/f"{camera.image_name}_gt.png")
            acc = result["acc"].squeeze(0)
            low = acc < 0.1
            white = (rgb8 > 250).all(dim=0)
            row = {"image": camera.image_name, "frame": int(camera.meta["frame"]),
                "camera": int(camera.meta["cam"]), "is_val": bool(camera.meta["is_val"]),
                "timestamp": float(camera.meta["timestamp"]),
                "psnr": float(psnr(pred, target).mean()), "ssim": float(ssim(pred, target).mean()),
                "lpips": float(lpips_metric(pred.unsqueeze(0), target.unsqueeze(0)).mean()),
                "float_psnr": float(psnr(rgb, gt).mean()),
                "acc_mean": float(acc.mean()), "acc_lt_0_1_fraction": float(low.float().mean()),
                "rgb_white_fraction": float(white.float().mean()),
                "white_and_low_acc_fraction": float((white & low).float().mean())}
            rows.append(row)
            if row["frame"] in (4,20,60):
                Image.fromarray((acc.clamp(0,1)*255).round().byte().cpu().numpy()).save(out/f"{camera.image_name}_acc.png")
            print(json.dumps(row), flush=True)
    names = ("psnr", "ssim", "lpips", "float_psnr", "acc_mean", "acc_lt_0_1_fraction", "rgb_white_fraction", "white_and_low_acc_fraction")
    def aggregate(subset):
        return {"view_count": len(subset), **{k: float(np.mean([r[k] for r in subset])) for k in names}}
    report = {"task_id": f"{Path(cfg.model_path).name}-TEST-{args.iteration}", "checkpoint_iteration": int(scene.loaded_iter),
        "config": args.config, "model_path": str(cfg.model_path), "seed": None,
        "seed_note": "Deterministic evaluation; no random seed explicitly set",
        "evaluation_kind": "official_config_temporal_test_cameras_0_to_4",
        "camera_source": "Scene.getTestCameras(); original is_val and actor pose interpolation preserved",
        "split_test": int(cfg.data.split_test), "split_train": int(cfg.data.split_train),
        "test_frames": sorted({f for _, f in keys}), "cameras": list(cfg.data.cameras),
        "gradient_train_view_count": len(train_keys), "gradient_train_overlap": 0,
        "finite_raw_rgb_depth_acc_views": len(rows),
        "metric_protocol": "full image, PNG-equivalent uint8 round-trip, official PSNR/SSIM and LPIPS alex v0.1",
        "input_role_note": "按官方配置留出 RGB 梯度训练帧；初始化使用全部候选帧的 RGB/LiDAR/COLMAP/先验，因此不是完全未见数据。",
        "mean": aggregate(rows), "per_camera": {str(c): aggregate([r for r in rows if r["camera"] == c]) for c in range(5)},
        "per_view": rows, "failure_ledger_refs": [], "failure_ledger_delta": "none"}
    (out/"metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "per_view"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
