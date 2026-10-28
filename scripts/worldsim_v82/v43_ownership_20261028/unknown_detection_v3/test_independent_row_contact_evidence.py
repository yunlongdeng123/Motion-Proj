"""CPU真实数据回放和当前U交集审计；不读取native编辑RGB。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "2"

import argparse
import json
from pathlib import Path

import numpy as np

from build_shadow_evidence import (DEFAULT_ROOT, RawClip, plane_height,
                                   project, sample, visibility)


def background_audit(directory, native, clip):
    contract = json.loads((directory / "contract.json").read_text())
    row_export = Path(contract["row_export"])
    rows = []
    for raw in contract["query_raws"]:
        with np.load(directory / ("%03d.npz" % raw)) as f:
            a = {k: f[k] for k in f.files}
        ids = a["ids"]
        with np.load(native / ("%03d.npz" % raw)) as d:
            role = np.zeros(len(ids), np.uint8)
            index = np.searchsorted(ids, d["ids"])
            assert np.array_equal(ids[index], d["ids"])
            role[index] = d["static_role"]
        p, mixed = a["background_protection_candidate"], a["mixed_color_mismatch"]
        assert not a["background_certified"].any()
        assert not (p & mixed).any()
        assert np.all(a["source_count"][p] >= 2)
        assert np.all(a["source_sh_match_count"][p] >= 2)
        assert np.all(a["source_sh_match_fraction"][p] >= 0.8)
        assert np.all(a["independent_time_span_s"][p] >= 0.499)
        assert np.all(a["independent_camera_baseline_m"][p] >= 1)
        assert np.all(a["independent_view_angle_deg"][p] >= 1)
        assert np.all(a["query_sh_rgb_mean_abs_error"][p] <= 0.060001)
        assert np.all(a["original_rgb_median_deviation"][p] <= 0.035001)
        assert np.all(a["original_rgb_p90_deviation"][p] <= 0.060001)
        query_column = np.flatnonzero(a["source_raws"] == raw)[0]
        assert not a["source_geometry_valid"][:, query_column].any()
        report = json.loads((directory / ("%03d.json" % raw)).read_text())
        replayed = 0
        for si, sr in enumerate(a["source_raws"]):
            source_rows = np.flatnonzero(a["source_geometry_valid"][:, si])
            if not len(source_rows):
                continue
            frame = clip.get(int(sr))
            gain = np.array(report["exposure_sources"][str(int(sr))]["gain_rgb_source_to_query"])
            rgb = sample(frame["rgb"], a["source_actual_surface_uv"][source_rows, si]) * gain
            assert np.allclose(rgb, a["source_original_rgb_exposure_matched"][source_rows, si], atol=2e-6)
            with np.load(row_export / ("%03d.npz" % sr)) as exported:
                native_rgb = exported["rgb"][source_rows] * gain
            error = np.abs(native_rgb - rgb)
            match = (error.mean(1) <= 0.060) & (error.max(1) <= 0.120)
            assert np.array_equal(match, a["source_sh_match"][source_rows, si])
            replayed += len(source_rows)
        current_u = role >= 2
        protected_u = current_u & ~p
        unchanged = np.array_equal(current_u, protected_u)
        rows.append({"raw": raw, "protection_candidates": int(p.sum()),
                     "protection_conflicts_with_current_u": int((p & current_u).sum()),
                     "background_geometry_witness": int(a["background_geometry_witness"].sum()),
                     "geometry_conflicts_with_current_u": int((a["background_geometry_witness"] & current_u).sum()),
                     "mixed_color_mismatch": int(mixed.sum()), "mixed_conflicts_with_current_u": int((mixed & current_u).sum()),
                     "unknown": int(a["unknown"].sum()), "original_source_rgb_observations_replayed": replayed,
                     "native_removal_set_unchanged": unchanged, "all_invariants_passed": True})
    intersection = {"schema": "independent-background-current-u-intersection-v1", "queries": rows,
                    "native_evidence": str(native), "native_set_unchanged": all(r["native_removal_set_unchanged"] for r in rows),
                    "native_rerender_skipped": True,
                    "reason": "native集合不变，跳过重复GPU；mixed不升级为颜色匹配保护，也不自动判car",
                    "ablation": "only color-matched background_protection_candidate veto; baseline/actor identity unchanged",
                    "certified_background_rows": 0, "human_GT": None}
    (directory / "intersection_summary.json").write_text(json.dumps(intersection, indent=2), encoding="utf-8")
    (directory / "verification.json").write_text(json.dumps({"all_invariants_passed": True, "queries": rows,
          "meaning": "all original source RGB values replayed and proposal invariants checked; not accuracy or shadow-free certification"}, indent=2), encoding="utf-8")
    return rows


def contact_audit(directory, clip):
    contract = json.loads((directory / "contract.json").read_text())
    strict = Path(contract["strict_input"])
    rows = []
    for raw in contract["query_raws"]:
        with np.load(directory / ("%03d.npz" % raw)) as f:
            a = {k: f[k] for k in f.files}
        frame = clip.get(raw)
        effect = a["contact_ground_photo_effect"]
        y, x = np.where(effect)
        assert not (effect & ~a["actual_near_ground_return"]).any()
        assert not (effect & ~a["contact_ground_photo_effect_valid"]).any()
        assert np.all(np.abs(a["actual_return_height_m"][effect]) <= 0.050001)
        assert np.all(a["contact_ground_photo_effect_score"][effect] <= 0.49)
        assert np.all(a["source_count"][effect] >= 3)
        assert np.all(a["luminance_ratio"][effect] <= 0.840001)
        assert np.all(a["luminance_ratio"][effect] >= 0.249999)
        assert np.all(a["luminance_drop"][effect] >= 0.034999)
        assert np.all(a["attenuation_rgb_residual"][effect] <= 0.035001)
        assert np.array_equal(a["body_overlap_contact_effect"], effect & frame["body"])
        assert not bool(a["full_shadow_segmentation"])
        actual = np.fromfile(clip.source / "velodyne" / (frame["meta"]["frame_id"] + ".bin"), np.float32).reshape(-1, 6)[:, :3].astype(np.float64)
        local = actual[a["actual_lidar_row_index"][effect]]
        world = local @ frame["pose"][:3, :3].T + frame["pose"][:3, 3]
        assert np.allclose(world, a["actual_return_world"][effect], atol=1e-8)
        uv, _ = project(world, frame)
        assert np.array_equal(np.floor(uv).astype(int), np.column_stack([x, y]))
        qrgb = sample(frame["rgb"], uv)
        assert np.allclose(qrgb, a["original_rgb_at_return_uv"][effect], atol=1e-7)
        for sr in np.unique(a["source_raw"][effect]):
            if sr < 0:
                continue
            indices = np.any(a["source_raw"][effect] == sr, axis=1)
            valid, _, _ = visibility(world[indices], clip.get(int(sr)), baseline=True)
            assert valid.all(), (raw, int(sr))
        assert not (a["source_raw"][effect] == raw).any()
        rows.append({"raw": raw, "contact_effect_candidates": len(x),
                     "inside_body_candidates": int((effect & frame["body"]).sum()),
                     "actual_original_lidar_and_source_visibility_replayed": True,
                     "all_invariants_passed": True})
    (directory / "verification.json").write_text(json.dumps({"all_invariants_passed": True, "queries": rows,
        "meaning": "direct original LiDAR and source visibility replay; near-ground may still be tire contact, not human shadow GT"}, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--contact", type=Path, required=True)
    parser.add_argument("--native-evidence", type=Path, required=True)
    args = parser.parse_args()
    clip = RawClip(args.root)
    print(json.dumps({"background": background_audit(args.background, args.native_evidence, clip),
                      "contact": contact_audit(args.contact, clip)}, indent=2), flush=True)
