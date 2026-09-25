"""将固定留出帧 20 的五个相机并排展示，不筛选最好案例。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.input_dir)
metrics = json.loads((root / "metrics.json").read_text())
fig, axes = plt.subplots(5, 3, figsize=(14, 14.5))
for cam in range(5):
    row = next(r for r in metrics["per_view"] if r["frame"] == 20 and r["camera"] == cam)
    for col, suffix in enumerate(("gt", "rgb", "acc")):
        im = Image.open(root / f"020_{cam}_{suffix}.png")
        axes[cam,col].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[cam,col].axis("off")
    axes[cam,0].set_title(f"Camera {cam} | GT frame 20", fontsize=11)
    axes[cam,1].set_title(f"{metrics['checkpoint_iteration']/1000:g}k render | PSNR {row['psnr']:.2f} dB", fontsize=11)
    axes[cam,2].set_title("Gaussian acc | white = 1", fontsize=11)
fig.suptitle(f"Official temporal test split, cameras 0-4\n{Path(metrics['model_path']).name}", fontsize=15)
fig.tight_layout(rect=(0,0,1,0.955))
fig.savefig(args.output, dpi=130, facecolor="white")
