"""Read three existing scene frames; quantify current sparse-depth last-write targets."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def functions_only(path, names):
    parsed = ast.parse(Path(path).read_text())
    funcs = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, scene = Path(args.source_root), Path(args.scene)
    spec = importlib.util.spec_from_file_location("calib_cpu", root / "scene/waymo_calibration.py")
    calib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calib)
    loader = functions_only(root / "scene/waymo_loader2.py", {"pad_poses", "unpad_poses", "transform_poses_pca"})
    camera_poses = []
    frames = sorted(p.stem for p in (scene / "calib").glob("*.txt"))
    for frame in frames:
        _, l2c = calib.read_named_camera_calibration(scene / "calib" / (frame + ".txt"), [0])
        ego = np.loadtxt(scene / "pose" / (frame + ".txt"))
        camera_poses.append((ego @ np.linalg.inv(l2c[0]))[:3])
    _, _, scale = loader["transform_poses_pca"](np.stack(camera_poses), fix_radius=0)
    records = []
    for frame in (frames[0], frames[10], frames[-1]):
        k, l2c = calib.read_named_camera_calibration(scene / "calib" / (frame + ".txt"), [0])
        width, height = Image.open(scene / "image/image_0" / (frame + ".png")).size
        width, height = int(width / 4), int(height / 4)
        ks = k[0].copy()
        ks[:2] /= 4
        pointfile = scene / "velodyne" / (frame + ".bin")
        points = np.fromfile(pointfile, dtype=np.float32).reshape(-1, 6)[:, :3]
        camera = (np.pad(points, ((0, 0), (0, 1)), constant_values=1) @ l2c[0].T)[:, :3] * scale
        uvz = camera @ ks.T
        uvz = uvz[uvz[:, 2] > 0]
        uvz[:, :2] /= uvz[:, 2:]
        uvz = uvz[(uvz[:, 0] >= 0) & (uvz[:, 0] < width)
                  & (uvz[:, 1] >= 0) & (uvz[:, 1] < height)]
        uv = uvz[:, :2].astype(int)
        ids, z = uv[:, 1] * width + uv[:, 0], uvz[:, 2]
        n = height * width
        count = np.bincount(ids, minlength=n)
        current, reverse = np.zeros(n), np.zeros(n)
        current[ids] = z
        reverse[ids[::-1]] = z[::-1]
        zmin, zmax = np.full(n, np.inf), np.zeros(n)
        np.minimum.at(zmin, ids, z)
        np.maximum.at(zmax, ids, z)
        populated = count > 0
        duplicate = count > 1
        farther = populated & (current > zmin + 1e-6 * scale)
        order_changed = populated & (np.abs(current - reverse) > 1e-6 * scale)
        span_m = (zmax[duplicate] - zmin[duplicate]) / scale
        error_m = (current[populated] - zmin[populated]) / scale
        mask_file = scene / "dynamic_masks/dynamic_mask_0" / (frame + ".png")
        mask = np.array(Image.open(mask_file).convert("L").resize((width, height), Image.Resampling.NEAREST)) > 0
        inverse_delta = np.abs(1 / (current[populated] + 1e-5) - 1 / (zmin[populated] + 1e-5))
        record = {
            "frame": frame, "camera": 0, "render_shape": [height, width],
            "point_file_sha256": hashlib.sha256(pointfile.read_bytes()).hexdigest(),
            "projected_point_count": len(z), "populated_depth_pixels": int(populated.sum()),
            "populated_fraction_full_image": float(populated.mean()),
            "duplicate_pixels": int(duplicate.sum()), "duplicate_fraction_of_depth_pixels": float(duplicate.sum() / populated.sum()),
            "last_write_not_nearest_pixels": int(farther.sum()),
            "last_write_not_nearest_fraction_depth_pixels": float(farther.sum() / populated.sum()),
            "target_changes_after_point_order_reversal_pixels": int(order_changed.sum()),
            "depth_pixels_with_collision_span_gt_0_5m": int((span_m > .5).sum()),
            "depth_pixels_with_collision_span_gt_2m": int((span_m > 2).sum()),
            "collision_span_m_quantiles": {str(q): float(np.quantile(span_m, q)) for q in (.5, .9, .99, 1.)},
            "last_write_minus_nearest_m_quantiles": {str(q): float(np.quantile(error_m, q)) for q in (.5, .9, .99, 1.)},
            "inverse_depth_target_difference_sum_divided_full_image": float(inverse_delta.sum() / n),
            "weighted_eq16_difference_scale_lambda_0_1": float(.1 * inverse_delta.sum() / n),
            "collision_last_farther_inside_dynamic_mask_pixels": int((farther.reshape(height, width) & mask).sum()),
            "collision_last_farther_outside_dynamic_mask_pixels": int((farther.reshape(height, width) & ~mask).sum()),
        }
        records.append(record)
    result = {
        "scope": "CPU read-only three existing front-camera frames; no CUDA, model or checkpoint",
        "source_root": str(root), "scene": str(scene), "world_meter_to_model_scale": float(scale),
        "normalization": "Exact frozen transform_poses_pca AST function,49 front camera poses,fix_radius0",
        "reference": "Eq16 PDF7-8/printed1702-1703; frozen camera_utils.py117-133 last-write depth projection",
        "caveat": "The paper does not describe a z-buffer. Order dependence is confirmed; selecting nearest in each coarse pixel is a conventional visibility choice and must be marked explicitly. Cause of observed collapse not established.",
        "records": records,
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"model_scale": float(scale), "out": str(out)}))
    for record in records:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
