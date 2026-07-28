#!/usr/bin/env python3
"""AD-GS semantic.py 所用 Grounding DINO HF 快照的固定版本 smoke。"""

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

from PIL import Image
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


def snapshot_fingerprint(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                digest.update(block)
        rows.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    snapshot = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_files_only=True,
    )
    fingerprint, files = snapshot_fingerprint(Path(snapshot))
    print("model   ", args.model_id)
    print("revision", args.revision)
    print("snapshot", snapshot)
    print("snapshot fingerprint", fingerprint)
    print("snapshot files", len(files), "bytes", sum(row["bytes"] for row in files))

    image_path = args.image
    if image_path is None:
        candidates = sorted(glob.glob(
            "/root/autodl-tmp/data/nuscenes/samples/CAM_FRONT/*.jpg"
        ))
        if not candidates:
            raise RuntimeError("找不到 nuScenes CAM_FRONT smoke 图像")
        image_path = candidates[0]
    image = Image.open(image_path).convert("RGB")
    text = "car. truck. bus. pedestrian. bicycle. motorcycle."

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=True,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=True,
    ).cuda().eval()
    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )
    torch.cuda.synchronize()
    boxes = results[0]["boxes"]
    labels = results[0]["labels"]
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert len(boxes) > 0, "Grounding DINO HF 模型未检出任何驾驶目标"
    assert torch.isfinite(boxes).all(), "Grounding DINO boxes 含 NaN/Inf"
    print("image   ", image_path, image.size)
    print("boxes   ", len(boxes), "labels", labels)
    print("latency ", round(time.time() - started, 3), "s")
    print("peak GPU mem", round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1), "MB")
    print("\nGROUNDING DINO HF SMOKE PASSED")


if __name__ == "__main__":
    main()
