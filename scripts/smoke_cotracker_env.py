#!/usr/bin/env python3
"""CoTracker3 固定仓库与固定权重的 GPU 冒烟测试。"""

import argparse
import glob
import hashlib
import subprocess
import sys
import time

import cv2
import numpy as np
import torch


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="/root/autodl-tmp/third_party/co-tracker",
    )
    parser.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/checkpoints/cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    from cotracker.predictor import CoTrackerPredictor

    images = sorted(glob.glob(
        "/root/autodl-tmp/data/nuscenes/samples/CAM_FRONT/*.jpg"
    ))[:args.frames]
    if len(images) != args.frames:
        raise RuntimeError("没有足够的 nuScenes CAM_FRONT 图像用于 CoTracker3 smoke")

    frames = []
    for path in images:
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError("无法读取图像: {}".format(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 256), interpolation=cv2.INTER_AREA)
        frames.append(image)

    video_np = np.stack(frames, axis=0)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().cuda()

    print("python  ", sys.version.split()[0])
    print("torch   ", torch.__version__, "| cuda", torch.version.cuda)
    print("device  ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("repo    ", args.repo)
    commit = subprocess.check_output(
        ["git", "-C", args.repo, "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    print("commit  ", commit)
    print("ckpt    ", args.checkpoint)
    print("ckpt sha256", sha256(args.checkpoint))
    print("video   ", tuple(video.shape), "range", float(video.min()), float(video.max()))

    torch.cuda.reset_peak_memory_stats()
    model = CoTrackerPredictor(
        checkpoint=args.checkpoint,
        offline=True,
        window_len=60,
        v2=False,
    ).cuda().eval()
    torch.cuda.synchronize()
    started = time.time()
    with torch.no_grad():
        tracks, visibility = model(video, grid_size=args.grid_size)
    torch.cuda.synchronize()
    elapsed = time.time() - started

    assert tracks.shape[:2] == (1, args.frames)
    assert visibility.shape[:3] == tracks.shape[:3]
    assert torch.isfinite(tracks).all(), "CoTracker3 tracks 含 NaN/Inf"
    assert visibility.any(), "CoTracker3 没有任何可见轨迹"
    print("tracks  ", tuple(tracks.shape), "visibility", tuple(visibility.shape))
    print("visible ", int(visibility.sum()), "/", visibility.numel())
    print("latency ", round(elapsed, 3), "s")
    print("peak GPU mem", round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1), "MB")
    print("\nCOTRACKER3 SMOKE PASSED")


if __name__ == "__main__":
    main()
