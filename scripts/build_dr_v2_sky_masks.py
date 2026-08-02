#!/usr/bin/env python
"""Create DriveStudio sky masks with the matching Cityscapes SegFormer-B5 model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--model-id", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("/root/autodl-tmp/hf_cache/hub"))
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    info = HfApi(endpoint=os.environ.get("HF_ENDPOINT") or None).model_info(
        args.model_id, revision=args.revision
    )
    revision = info.sha
    snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=revision,
            cache_dir=args.cache_dir,
        )
    )
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        snapshot, local_files_only=True
    ).to(args.device)
    model.eval()
    id_to_label = {int(key): str(value) for key, value in model.config.id2label.items()}
    sky_ids = [key for key, value in id_to_label.items() if value.strip().lower() == "sky"]
    if len(sky_ids) != 1:
        raise RuntimeError(f"expected exactly one Cityscapes sky class, got {sky_ids}: {id_to_label}")
    sky_id = sky_ids[0]

    image_paths = sorted(
        path
        for path in (args.scene_root / "images").glob("*.jpg")
        if int(path.stem.rsplit("_", 1)[1]) in set(args.cameras)
    )
    if not image_paths:
        raise RuntimeError(f"no matching images under {args.scene_root / 'images'}")
    output_root = args.scene_root / "sky_masks"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, image_path in enumerate(image_paths, 1):
        output = output_root / f"{image_path.stem}.png"
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            mask = logits.argmax(dim=1)[0].eq(sky_id).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(output)
        if output.stat().st_size == 0:
            raise RuntimeError(f"empty sky mask: {output}")
        rows.append(
            {
                "image": image_path.name,
                "mask": output.name,
                "mask_bytes": output.stat().st_size,
                "mask_sha256": sha256_file(output),
                "sky_fraction": float((mask > 0).mean()),
            }
        )
        if index % 10 == 0 or index == len(image_paths):
            print(f"sky masks {index}/{len(image_paths)}", flush=True)

    siblings = list(info.siblings or [])
    source_files = []
    for sibling in siblings:
        path = snapshot / sibling.rfilename
        if path.is_file() and path.stat().st_size > 0:
            source_files.append(
                {
                    "path": sibling.rfilename,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    card_data = info.card_data.to_dict() if info.card_data is not None else {}
    manifest = {
        "schema_version": 1,
        "status": "done",
        "model_id": args.model_id,
        "resolved_revision": revision,
        "license": card_data.get("license"),
        "model_snapshot": str(snapshot),
        "source_files": source_files,
        "scene_root": str(args.scene_root),
        "cameras": args.cameras,
        "sky_class_id": sky_id,
        "sky_class_label": id_to_label[sky_id],
        "image_count": len(image_paths),
        "mask_count": len(rows),
        "mean_sky_fraction": float(np.mean([row["sky_fraction"] for row in rows])),
        "files": rows,
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps({"status": "done", "masks": len(rows), "revision": revision}, sort_keys=True))


if __name__ == "__main__":
    main()
