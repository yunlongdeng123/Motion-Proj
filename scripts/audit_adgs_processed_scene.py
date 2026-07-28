#!/usr/bin/env python3
"""审计 AD-GS scene-0230 预处理、伪监督、flow 与 COLMAP 产物。"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


EXPECTED_IMAGES = 180
EXPECTED_VAL_REL_FRAMES = set(range(4, 60, 4))


def fail(result, message):
    result["failures"].append(message)


def finite(array):
    return bool(np.isfinite(array).all())


def count_and_validate_images(scene, result):
    image_dir = scene / "image"
    paths = sorted(image_dir.glob("*.png"))
    expected_names = ["{:06d}.png".format(idx) for idx in range(EXPECTED_IMAGES)]
    if [path.name for path in paths] != expected_names:
        fail(result, "image 文件名或计数不等于 000000..000179.png")
    bad = 0
    for path in paths:
        try:
            with Image.open(str(path)) as image:
                if image.size != (1600, 900):
                    bad += 1
                image.verify()
        except Exception:
            bad += 1
    if bad:
        fail(result, "{} 张 image 无法解析或尺寸异常".format(bad))
    return len(paths)


def validate_meta(scene, result):
    path = scene / "meta.npz"
    if not path.is_file():
        fail(result, "缺少 meta.npz")
        return {}
    meta = np.load(str(path), allow_pickle=False)
    expected_shapes = {
        "K": (180, 3, 3),
        "R": (180, 3, 3),
        "T": (180, 3),
        "time_stamps": (180,),
        "is_val_list": (180,),
    }
    shapes = {}
    for key, shape in expected_shapes.items():
        if key not in meta:
            fail(result, "meta.npz 缺少 {}".format(key))
            continue
        value = meta[key]
        shapes[key] = list(value.shape)
        if value.shape != shape:
            fail(result, "meta.{} shape {} != {}".format(key, value.shape, shape))
        if key != "is_val_list" and not finite(value):
            fail(result, "meta.{} 含 NaN/Inf".format(key))

    if "time_stamps" in meta:
        expected_time = np.repeat(np.arange(60, dtype=np.float32), 3)
        if not np.array_equal(meta["time_stamps"], expected_time):
            fail(result, "meta.time_stamps 与 60 帧×3 相机顺序不一致")
    if "is_val_list" in meta:
        expected_val = np.array([
            rel_frame in EXPECTED_VAL_REL_FRAMES
            for rel_frame in np.repeat(np.arange(60), 3)
        ])
        if not np.array_equal(meta["is_val_list"], expected_val):
            fail(result, "meta.is_val_list 与 upstream every-4 协议不一致")
    return {
        "shapes": shapes,
        "n_val_images": int(meta["is_val_list"].sum())
        if "is_val_list" in meta else None,
    }


def validate_ply(path, result, label, require_object=False):
    if not path.is_file() or path.stat().st_size <= 0:
        fail(result, "缺少或为空: {}".format(path))
        return {}
    try:
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data
    except Exception as exc:
        fail(result, "{} PLY 解析失败: {}".format(label, exc))
        return {}
    names = list(vertex.dtype.names or [])
    if len(vertex) == 0:
        fail(result, "{} PLY 没有顶点".format(label))
    for key in ["x", "y", "z"]:
        if key not in names or not finite(vertex[key]):
            fail(result, "{} PLY {} 缺失或含 NaN/Inf".format(label, key))
    if require_object and "obj" not in names:
        fail(result, "{} PLY 缺少 obj 属性".format(label))
    return {"vertices": len(vertex), "properties": names}


def validate_dense_arrays(scene, folder, result, integer):
    root = scene / folder
    prefix = "mask_" if integer else ""
    paths = sorted(root.glob("{}*.npy".format(prefix)))
    expected_names = [
        "{}{:06d}.npy".format(prefix, idx) for idx in range(EXPECTED_IMAGES)
    ]
    if [path.name for path in paths] != expected_names:
        fail(result, "{} 文件名或计数不完整".format(folder))
    bad_shape = 0
    bad_value = 0
    nonempty = 0
    for path in paths:
        value = np.load(str(path), allow_pickle=False)
        if value.shape not in [(900, 1600), (900, 1600, 1)]:
            bad_shape += 1
        if not finite(value):
            bad_value += 1
        if integer and not np.issubdtype(value.dtype, np.integer):
            bad_value += 1
        if np.any(value > 0):
            nonempty += 1
    if bad_shape:
        fail(result, "{} 有 {} 个 shape 异常".format(folder, bad_shape))
    if bad_value:
        fail(result, "{} 有 {} 个 dtype/有限性异常".format(folder, bad_value))
    return {"count": len(paths), "nonempty": nonempty}


def validate_flow(scene, result):
    meta = np.load(str(scene / "meta.npz"), allow_pickle=False)
    is_val = meta["is_val_list"]
    semantic_dir = scene / "semantic"
    expected = []
    for idx in range(EXPECTED_IMAGES):
        mask = np.load(
            str(semantic_dir / "mask_{:06d}.npy".format(idx)),
            allow_pickle=False,
        )
        if not is_val[idx] and np.any(mask > 0):
            expected.append(idx)
    flow_dir = scene / "flow"
    paths = sorted(flow_dir.glob("*.npz"))
    actual = [int(path.stem) for path in paths]
    if actual != expected:
        fail(result, "flow coverage 与 non-val 非空 object mask 不一致")
    bad = 0
    for path in paths:
        payload = np.load(str(path), allow_pickle=True)["flow"]
        for item in payload:
            for value in item:
                if isinstance(value, np.ndarray) and not finite(value):
                    bad += 1
    if bad:
        fail(result, "flow 中存在 {} 个 NaN/Inf 数组".format(bad))
    return {
        "count": len(paths),
        "expected": len(expected),
        "coverage": len(paths) / max(len(expected), 1),
    }


def validate_colmap(scene, result):
    model = scene / "colmap/triangulated/sparse/model"
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    for name in required:
        path = model / name
        if not path.is_file() or path.stat().st_size <= 0:
            fail(result, "COLMAP model 缺少 {}".format(name))
    analysis = ""
    if all((model / name).is_file() for name in required):
        proc = subprocess.run(
            [
                "/root/autodl-tmp/envs/adgs/bin/colmap",
                "model_analyzer",
                "--path",
                str(model),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        analysis = proc.stdout
        if proc.returncode != 0:
            fail(result, "COLMAP model_analyzer 失败")
    return {"model_analyzer": analysis}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    scene = Path(args.scene)
    result = {
        "schema_version": 1,
        "scene": str(scene),
        "failures": [],
    }
    result["images"] = {"count": count_and_validate_images(scene, result)}
    result["meta"] = validate_meta(scene, result)
    result["points3d"] = validate_ply(
        scene / "points3d.ply", result, "points3d", require_object=True
    )
    result["depth"] = validate_dense_arrays(scene, "depth", result, integer=False)
    result["sky"] = validate_dense_arrays(scene, "sky", result, integer=True)
    result["semantic"] = validate_dense_arrays(
        scene, "semantic", result, integer=True
    )
    result["flow"] = validate_flow(scene, result)
    result["colmap"] = validate_colmap(scene, result)
    result["colmap_ply"] = validate_ply(
        scene / "colmap.ply", result, "colmap", require_object=False
    )
    result["status"] = "done" if not result["failures"] else "blocked"

    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".partial")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(output))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "done" else 2)


if __name__ == "__main__":
    main()
