#!/usr/bin/env python3
"""Generate actual SegFormer Cityscapes sky masks for a prepared FRONT pilot."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from mmseg.apis import init_segmentor, inference_segmentor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    model = init_segmentor(str(args.config), str(args.checkpoint), device=args.device)
    model.eval()
    image_dir = args.scene / "image/image_0"
    sky_dir = args.scene / "sky/sky_0"
    sky_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for image_path in sorted(image_dir.glob("*.png")):
        target = sky_dir / image_path.name
        if target.exists():
            reused = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            assert reused is not None and set(np.unique(reused)).issubset({0,255})
            summaries.append({"frame": image_path.stem,"sky_fraction":float(np.mean(reused>0)),"reused_verified_existing":True})
            continue
        result = inference_segmentor(model, str(image_path))
        mask = (result[0] == 10).astype(np.uint8) * 255
        target = sky_dir / image_path.name
        assert cv2.imwrite(str(target), mask)
        summary = {"frame": image_path.stem, "sky_fraction": float(np.mean(mask > 0))}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    manifest_path = args.scene / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(summaries) == len(manifest["frames"])
    manifest["sky_masks_present"] = len(summaries)
    manifest["sky_segmentation"] = {"model": "SegFormer B5 Cityscapes", "class_id": 10,
                                    "config": str(args.config), "checkpoint": str(args.checkpoint),
                                    "frame_statistics": summaries}
    manifest["status"] = "static_inputs_prepared"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "sky_masks": len(summaries)}), flush=True)


if __name__ == "__main__":
    main()
