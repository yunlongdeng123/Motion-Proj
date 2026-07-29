#!/usr/bin/env python3
"""把冻结 AD-GS 六场景窗口 staging 为 DGGT 官方 loader 的输入结构。"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_ROOT = Path(
    "/root/autodl-tmp/data/dynamic_recon/processed/adgs_nuscenes_v1"
)
SCENES = [
    "scene-0230",
    "scene-0242",
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]
WINDOWS = [
    [10, 11, 12, 13],
    [34, 35, 36, 37],
    [66, 67, 68, 69],
]
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]
EXPECTED_SIZE = (1600, 900)


def now():
    return dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat()


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(path))


def load_mask(path):
    mask = np.load(path, allow_pickle=False)
    if mask.shape != (EXPECTED_SIZE[1], EXPECTED_SIZE[0]):
        raise RuntimeError("{} shape 非预期: {}".format(path, mask.shape))
    if not np.isfinite(mask).all():
        raise RuntimeError("{} 含 NaN/Inf".format(path))
    return mask


def write_binary_mask(source, target):
    mask = load_mask(source)
    image = Image.fromarray((mask != 0).astype(np.uint8) * 255, mode="L")
    image.save(target, format="PNG", optimize=False)
    with Image.open(target) as check:
        if check.size != EXPECTED_SIZE or check.mode != "L":
            raise RuntimeError("{} staging mask 校验失败".format(target))
    return {
        "source_nonzero": int(np.count_nonzero(mask)),
        "source_unique_count": int(np.unique(mask).size),
        "staged_nonzero": int(np.count_nonzero(np.asarray(image))),
    }


def stage(output_root):
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("输出目录非空，禁止覆盖: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    pseudo_scene_map = {}
    pseudo_index = 0

    for scene in SCENES:
        source_scene = SOURCE_ROOT / scene
        for window in WINDOWS:
            pseudo_scene = "{:03d}".format(pseudo_index)
            pseudo_index += 1
            pseudo_scene_map[pseudo_scene] = {
                "source_scene": scene,
                "raw_frames": window,
            }
            target_scene = output_root / pseudo_scene
            image_dir = target_scene / "images"
            sky_dir = target_scene / "sky_masks"
            dynamic_dir = target_scene / "fine_dynamic_masks/all"
            for directory in [image_dir, sky_dir, dynamic_dir]:
                directory.mkdir(parents=True)

            for staged_frame, raw_frame in enumerate(window):
                processed_frame = raw_frame - 10
                for view, camera in enumerate(CAMERAS):
                    source_index = processed_frame * len(CAMERAS) + view
                    source_image = (
                        source_scene / "image/{:06d}.png".format(source_index)
                    )
                    source_sky = (
                        source_scene
                        / "sky/mask_{:06d}.npy".format(source_index)
                    )
                    source_dynamic = (
                        source_scene
                        / "semantic/mask_{:06d}.npy".format(source_index)
                    )
                    for source in [
                        source_image, source_sky, source_dynamic
                    ]:
                        if not source.is_file() or source.stat().st_size == 0:
                            raise RuntimeError(
                                "源文件缺失或为空: {}".format(source)
                            )

                    target_name = "{:03d}_{}.png".format(
                        staged_frame, view
                    )
                    target_image = image_dir / target_name
                    os.link(str(source_image), str(target_image))
                    with Image.open(target_image) as check:
                        if check.size != EXPECTED_SIZE or check.mode != "RGB":
                            raise RuntimeError(
                                "{} image 校验失败: {} {}".format(
                                    target_image, check.size, check.mode
                                )
                            )
                    target_sky = sky_dir / target_name
                    target_dynamic = dynamic_dir / target_name
                    sky_stats = write_binary_mask(source_sky, target_sky)
                    dynamic_stats = write_binary_mask(
                        source_dynamic, target_dynamic
                    )
                    rows.append({
                        "pseudo_scene": pseudo_scene,
                        "source_scene": scene,
                        "raw_frame": raw_frame,
                        "processed_frame": processed_frame,
                        "staged_frame": staged_frame,
                        "view": view,
                        "camera": camera,
                        "source_index": source_index,
                        "source_image": str(source_image),
                        "source_image_bytes": source_image.stat().st_size,
                        "source_image_sha256": sha256_file(source_image),
                        "staged_image": str(target_image),
                        "staged_image_kind": "hardlink",
                        "staged_image_sha256": sha256_file(target_image),
                        "source_sky": str(source_sky),
                        "source_sky_sha256": sha256_file(source_sky),
                        "staged_sky": str(target_sky),
                        "staged_sky_sha256": sha256_file(target_sky),
                        "sky_stats": sky_stats,
                        "source_dynamic": str(source_dynamic),
                        "source_dynamic_sha256": sha256_file(source_dynamic),
                        "staged_dynamic": str(target_dynamic),
                        "staged_dynamic_sha256": sha256_file(target_dynamic),
                        "dynamic_stats": dynamic_stats,
                    })

    mapping_path = output_root / "raw_to_staged.jsonl"
    with mapping_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    mapping_sha256 = sha256_file(mapping_path)
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "source_root": str(SOURCE_ROOT),
        "output_root": str(output_root),
        "official_scenes": SCENES,
        "windows": WINDOWS,
        "cameras_in_view_order": CAMERAS,
        "source_resolution_wh": list(EXPECTED_SIZE),
        "staged_pseudo_scenes": pseudo_scene_map,
        "pseudo_scene_count": len(pseudo_scene_map),
        "frame_camera_count": len(rows),
        "expected_frame_camera_count": (
            len(SCENES) * len(WINDOWS) * 4 * len(CAMERAS)
        ),
        "image_staging": "same-filesystem hardlink, byte-preserving",
        "sky_staging": "uint16 mask != 0 -> uint8 PNG",
        "dynamic_staging": "uint16 instance mask != 0 -> uint8 PNG",
        "poses_provided_to_dggt": False,
        "intrinsics_provided_to_dggt": False,
        "per_scene_optimization": False,
        "dggt_loader_resize": {
            "source": [900, 1600],
            "actual_model_input": [294, 518],
            "rule": "width=518, aspect-preserving height rounded to multiple of 14",
        },
        "mapping": {
            "path": str(mapping_path),
            "bytes": mapping_path.stat().st_size,
            "sha256": mapping_sha256,
        },
    }
    if manifest["frame_camera_count"] != manifest[
        "expected_frame_camera_count"
    ]:
        raise RuntimeError("staging 行数不完整")
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest = stage(Path(args.output_root))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
