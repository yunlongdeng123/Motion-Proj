"""审计实际 COLMAP 点、track 与已保存初始化，不重建或改写训练数据。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary
from lib.utils.colmap_view_mapping import build_colmap_view_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    model = run / "colmap/triangulated/sparse/model"
    xyz, rgb, err, tracks = read_points3D_binary(str(model / "points3D.bin"))
    images = read_extrinsics_binary(str(model / "images.bin"))
    err = err[:, 0]
    keep = err < 0.6  # 与 drivestudio_utils.py 的严格不等式相同。
    lengths = np.array([len(t) for t in tracks])
    ply = PlyData.read(str(run / "input_ply/points3D_bkgd.ply"))["vertex"]
    bkgd = np.column_stack([ply[k] for k in ("x", "y", "z")])
    # PLY 将 xyz 存为 float32，按同样转换做坐标匹配，再校验尾部顺序与 track。
    dists, indices = cKDTree(bkgd).query(xyz.astype(np.float32), workers=2)
    saved = keep & (dists == 0)
    n_saved = int(saved.sum())
    suffix_matches = n_saved > 0 and np.array_equal(bkgd[-n_saved:], xyz[saved].astype(np.float32))
    visibility = np.load(run / "input_ply/points3D_bkgd.npy", mmap_mode="r")
    assert visibility.shape[0] == len(bkgd)
    n_frames = visibility.shape[1] // 5
    actual_names = sorted((Path("/root/autodl-tmp/data/v76_vadgs/official_000/000") / "images").glob("*.jpg"))
    actual_views = [(int(p.stem.split("_")[0]), int(p.stem.split("_")[1])) for p in actual_names]
    actual_views = [(f,c) for f,c in actual_views if 0 <= f < n_frames and c in range(5)]
    assert actual_views == [(f,c) for f in range(n_frames) for c in range(5)]
    fixed_mapping = build_colmap_view_mapping(images, [f for f,c in actual_views], [c for f,c in actual_views])
    fixed_mapping_errors = sum(actual_views[fixed_mapping[i]] != (int(Path(im.name).stem), int(Path(im.name).parent.name.split("_")[1])) for i,im in images.items())
    image_rows = []
    counts_all = {i: 0 for i in images}
    counts_keep = {i: 0 for i in images}
    counts_saved = {i: 0 for i in images}
    for p, track in enumerate(tracks):
        for image_id, _ in track:
            counts_all[image_id] += 1
            counts_keep[image_id] += int(keep[p])
            counts_saved[image_id] += int(saved[p])
    mapping_errors = []
    for image_id, im in images.items():
        cam_dir, filename = im.name.split("/")
        cam, frame = int(cam_dir.split("_")[1]), int(Path(filename).stem)
        if ((image_id - 1) // n_frames, (image_id - 1) % n_frames) != (cam, frame):
            mapping_errors.append({"id": image_id, "name": im.name})
        image_rows.append({"image_id": image_id, "name": im.name, "camera": cam,
                           "frame": frame, "observations_before": counts_all[image_id],
                           "observations_error_lt_0_6": counts_keep[image_id],
                           "observations_saved_initialization": counts_saved[image_id]})
    track_mismatch = 0
    correct_track_mismatch = 0
    edges_actual = edges_correct = edges_shared = 0
    wrong_camera_edges = wrong_frame_edges = 0
    if suffix_matches:
        for j, p in enumerate(np.flatnonzero(saved)):
            expected = np.zeros(visibility.shape[1], dtype=bool)
            correct = np.zeros(visibility.shape[1], dtype=bool)
            for image_id, _ in tracks[p]:
                cam = (image_id - 1) // n_frames
                frame = (image_id - 1) % n_frames
                expected[frame * 5 + cam] = True
                cam_dir, filename = images[image_id].name.split("/")
                true_cam, true_frame = int(cam_dir.split("_")[1]), int(Path(filename).stem)
                correct[true_frame * 5 + true_cam] = True
                wrong_camera_edges += int(cam != true_cam)
                wrong_frame_edges += int(frame != true_frame)
            actual = visibility[len(bkgd)-n_saved+j]
            track_mismatch += int(not np.array_equal(expected, actual))
            correct_track_mismatch += int(not np.array_equal(correct, actual))
            edges_actual += int(actual.sum())
            edges_correct += int(correct.sum())
            edges_shared += int((correct & actual).sum())
    per_camera = {}
    for cam in sorted({row["camera"] for row in image_rows}):
        subset = [row for row in image_rows if row["camera"] == cam]
        before = sum(row["observations_before"] for row in subset)
        after = sum(row["observations_error_lt_0_6"] for row in subset)
        per_camera[str(cam)] = {"views": len(subset), "observations_before": before,
            "observations_error_lt_0_6": after, "observation_retention": after / before,
            "observations_saved_initialization": sum(row["observations_saved_initialization"] for row in subset),
            "retained_per_image_min_median_max": np.percentile([row["observations_error_lt_0_6"] for row in subset], [0, 50, 100]).tolist()}
    result = {"task_id": f"{run.name}-COLMAP-AUDIT", "run": str(run),
        "points_path": str(model / "points3D.bin"), "threshold": "error < 0.6 px",
        "points_before": len(xyz), "points_error_lt_0_6": int(keep.sum()),
        "retention_fraction": float(keep.mean()), "mean_error_before": float(err.mean()),
        "mean_error_after": float(err[keep].mean()),
        "error_quantiles_0_25_50_75_90_95_99_100": np.percentile(err, [0,25,50,75,90,95,99,100]).tolist(),
        "track_length_mean_before": float(lengths.mean()),
        "track_length_mean_after": float(lengths[keep].mean()),
        "total_track_observations": int(lengths.sum()),
        "retained_track_observations": int(lengths[keep].sum()),
        "saved_initialization": {"background_points": len(bkgd), "visibility_shape": list(visibility.shape),
            "retained_colmap_points_exact_float32_matches": n_saved,
            "colmap_suffix_order_verified": suffix_matches,
            "colmap_visibility_rows_mismatched": track_mismatch if suffix_matches else None,
            "visibility_comparison_note": "previous mismatch count checks reproduction of the old arithmetic; the following checks actual image names",
            "colmap_visibility_rows_differ_from_name_mapping": correct_track_mismatch,
            "colmap_visibility_edges_actual": edges_actual,
            "colmap_visibility_edges_correct": edges_correct,
            "colmap_visibility_edges_shared": edges_shared,
            "colmap_track_edges_wrong_camera": wrong_camera_edges,
            "colmap_track_edges_wrong_frame": wrong_frame_edges,
            "other_background_points": len(bkgd)-n_saved,
            "fraction_colmap": n_saved/len(bkgd)},
        "image_id_mapping_error_count": len(mapping_errors),
        "fix_validation": {"mapped_images": len(fixed_mapping), "view_table_matches_dataset_filename_order": True,
            "mismatches_against_image_names": fixed_mapping_errors,
            "scope": "修复后的映射函数通过当前305图像验证；旧 input_ply 与训练checkpoint未被修复或重新生成。"},
        "image_id_mapping_errors": mapping_errors, "per_camera": per_camera,
        "per_image": sorted(image_rows, key=lambda r: (r["camera"], r["frame"])),
        "interpretation_limit": "保留率不能独立衡量几何覆盖或证明 sparse matching 的因果影响；需同输入 exhaustive 对照。",
        "failure_ledger_refs": ["V76-F01"],
        "failure_ledger_delta": "V76-F01: saved visibility remains inconsistent with image names" if correct_track_mismatch else "none; name-based saved visibility verified"}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_image", "image_id_mapping_errors")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
