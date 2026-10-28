"""CPU：原始 RGB/LiDAR 的行级正背景见证；不读取任何删除/补景结果。

缺背景见证不构成 car/U 正例。颜色与真实稳定表面不符时标 mixed。
当前源阴影缺少独立完整负标签，因此保护候选不升级为 certified。
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "2"

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter
from scipy.spatial import cKDTree

from build_shadow_evidence import (CTX, DEFAULT_ROOT, RawClip, actor_distances,
                                   fit_exposure, footprint_distance, plane_height,
                                   project, sample)

CONFIG = {
    "schema": "independent-raw-row-background-witness-v1",
    "max_sigma_for_protection_m": 0.20,
    "min_effective_opacity": 0.05,
    "max_row_world_motion_m": 0.15,
    "actual_surface_match_radius_m": "max(0.08,min(0.20,2*sigma_max_m))",
    "max_actual_surface_projection_error_px": 2.0,
    "target_footprint_clearance_m": 6.0,
    "other_actor_footprint_clearance_m": 5.0,
    "known_shadow_exclusion_dilation_px": 3,
    "minimum_semantic_confidence": 0.70,
    "max_patch_rgb_std": 0.045,
    "max_raw_rgb_median_deviation": 0.035,
    "max_raw_rgb_p90_deviation": 0.060,
    "max_sh_rgb_mean_abs_error": 0.060,
    "max_sh_rgb_channel_abs_error": 0.120,
    "minimum_source_sh_match_fraction": 0.80,
    "min_independent_time_seconds": 0.5,
    "min_independent_camera_baseline_m": 1.0,
    "min_independent_view_angle_deg": 1.0,
    "minimum_independent_original_views": 2,
    "source_shadow_state_unknown": True,
    "background_certified": False,
    "geometry_model": "visible actual LiDAR point near same original-row world center; conservative sphere, not Gaussian ray intersection",
    "forbidden_inputs": ["deleted RGB", "inpainted RGB", "model repair RGB", "suppression feedback"],
    "missing_evidence_semantics": "unknown; never positive car/U",
    "training_updates": 0, "GPU_calls": 0, "human_GT": None,
}
SOURCE_REASONS = {
    "same_query_not_independent": 1, "row_inactive": 2,
    "row_too_wide_for_center_surface_test": 4, "row_world_position_changed": 8,
    "not_in_front_camera": 16, "no_close_actual_visible_lidar_surface": 32,
    "foreground_depth_occlusion": 64, "target_or_other_actor_near": 128,
    "body_or_known_shadow_region": 256, "semantic_or_patch_ambiguous": 512,
    "global_exposure_match_unavailable": 1024,
    "source_sh_rgb_mismatch": 2048,
}
ROW_REASONS = {
    "insufficient_independent_visible_sources": 1,
    "raw_source_rgb_not_stable": 2,
    "source_sh_rgb_mismatch": 4,
    "query_sh_rgb_mismatch": 8,
    "positive_background_geometry_witness": 16,
    "positive_background_protection_candidate": 32,
    "source_shadow_absence_unverified": 64,
    "row_wide_or_inactive": 128,
}


class WitnessFrames:
    def __init__(self, clip, shadow):
        self.clip = clip
        self.shadow = shadow
        self.cache = {}

    def get(self, raw):
        if raw in self.cache:
            return self.cache[raw]
        f = self.clip.get(raw)
        points = np.fromfile(self.clip.source / "velodyne" / (f["meta"]["frame_id"] + ".bin"), np.float32).reshape(-1, 6)[:, :3].astype(np.float64)
        world = points @ f["pose"][:3, :3].T + f["pose"][:3, 3]
        uv, depth = project(world, f)
        valid = (depth > 0.1) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < 480) & (uv[:, 1] >= 0) & (uv[:, 1] < 320)
        ix = np.flatnonzero(valid)
        pp = np.floor(uv[ix]).astype(int)
        nearest = np.full((320, 480), np.inf)
        np.minimum.at(nearest, (pp[:, 1], pp[:, 0]), depth[ix])
        ix = ix[depth[ix] <= nearest[pp[:, 1], pp[:, 0]] + 0.03]
        body_guard = binary_dilation(f["body"], iterations=3) | f["dynamic_guard"]
        known_shadow = np.zeros((320, 480), bool)
        shadow_path = self.shadow / ("%03d.npz" % raw)
        if shadow_path.exists():
            with np.load(shadow_path) as s:
                known_shadow |= s["shadow_candidate"]
                if "photo_supported_unknown" in s.files:
                    known_shadow |= s["photo_supported_unknown"]
        else:
            # 没有本时刻阴影观测时不从邻帧借标签。
            known_shadow[:] = True
        guard = body_guard | binary_dilation(known_shadow, iterations=3)
        mean = uniform_filter(f["rgb"], size=(3, 3, 1))
        variance = np.maximum(uniform_filter(f["rgb"] ** 2, size=(3, 3, 1)) - mean ** 2, 0)
        patch_std = np.sqrt(variance).max(2)
        origin_sensor = -f["camera"][:, :3].T @ f["camera"][:, 3]
        origin = origin_sensor @ f["pose"][:3, :3].T + f["pose"][:3, 3]
        result = {"frame": f, "surface_world": world[ix], "surface_uv": uv[ix],
                  "tree": cKDTree(world[ix]), "guard": guard, "patch_std": patch_std,
                  "camera_origin": origin}
        self.cache[raw] = result
        return result


def exposure_anchors(query, plane):
    anchors, uv = query["road_world"], query["road_uv"]
    ok = footprint_distance(anchors[:, :2], query["target"]) > 7.0
    ok &= actor_distances(anchors[:, :2], query, exclude_target=True) > 3.0
    ok &= np.abs(anchors[:, 2] - plane_height(anchors[:, :2], plane)) < 0.15
    anchors, uv = anchors[ok], uv[ok]
    if len(anchors) > 4000:
        ii = np.linspace(0, len(anchors) - 1, 4000).astype(int)
        anchors, uv = anchors[ii], uv[ii]
    return anchors, sample(query["rgb"], uv)


def independent_pair(valid, raw_ids, origins, timestamps, world):
    """保守取最早/最晚合格见证；不挑选更好看的中间颜色。"""
    n = valid.shape[1]
    count = valid.sum(0)
    first = np.argmax(valid, axis=0)
    last = len(raw_ids) - 1 - np.argmax(valid[::-1], axis=0)
    time_span = timestamps[last] - timestamps[first]
    baseline = np.linalg.norm(origins[last] - origins[first], axis=1)
    ray1 = world - origins[first]
    ray2 = world - origins[last]
    cosine = np.sum(ray1 * ray2, axis=1) / np.maximum(np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1), 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    pair = (count >= 2) & (time_span >= 0.499) & (baseline >= 1.0) & (angle >= 1.0)
    raws = np.column_stack([raw_ids[first], raw_ids[last]]).astype(np.int16)
    raws[count < 2] = -1
    return pair, raws, time_span, baseline, angle


def query_witnesses(raw, ids, row_data, sources, frames, clip, shadow, output):
    started = time.monotonic()
    q = row_data[raw]
    world = q["world"]
    n, ns = len(ids), len(sources)
    query = clip.get(raw)
    plane_json = json.loads((shadow / ("%03d.json" % raw)).read_text())["plane"]
    plane = {"origin": np.array(plane_json["origin_world"]), "coefficients": np.array(plane_json["z_minus_origin_z_equals"])}
    anchors, anchor_rgb = exposure_anchors(query, plane)
    raw_rgb = np.full((ns, n, 3), np.nan, np.float32)
    sh_error = np.full((ns, n), np.nan, np.float32)
    sh_match = np.zeros((ns, n), bool)
    geometry = np.zeros((ns, n), bool)
    reason = np.zeros((ns, n), np.uint16)
    surface_distance = np.full((ns, n), np.nan, np.float32)
    observed_uv = np.full((ns, n, 2), np.nan, np.float32)
    exposure = {}
    origins = []
    timestamps = []
    for si, sr in enumerate(sources):
        data = row_data[sr]
        source = frames.get(sr)
        f = source["frame"]
        origins.append(source["camera_origin"])
        timestamps.append(f["meta"]["timestamp_micros"] / 1e6)
        if sr == raw:
            reason[si] |= SOURCE_REASONS["same_query_not_independent"]
            continue
        gain, exp = fit_exposure(query, f, anchors, anchor_rgb)
        exposure[str(sr)] = exp
        if gain is None:
            reason[si] |= SOURCE_REASONS["global_exposure_match_unavailable"]
            continue
        inactive = data["opacity"] < CONFIG["min_effective_opacity"]
        wide = np.maximum(data["sigma_m"], q["sigma_m"]) > CONFIG["max_sigma_for_protection_m"]
        moved = np.linalg.norm(data["world"] - world, axis=1) > CONFIG["max_row_world_motion_m"]
        uv, z = project(world, f)
        inside = (z > 0.1) & np.isfinite(uv).all(1) & (uv[:, 0] >= 2) & (uv[:, 0] < 478) & (uv[:, 1] >= 2) & (uv[:, 1] < 318)
        distance, nearest = source["tree"].query(world, workers=2)
        measured_uv = source["surface_uv"][nearest]
        tol = np.maximum(0.08, np.minimum(0.20, 2 * data["sigma_m"]))
        surface_ok = (distance <= tol) & (np.linalg.norm(measured_uv - uv, axis=1) <= 2)
        nearest_z = sample(f["all_lidar_depth"], uv, 0)
        foreground = (sample(f["all_lidar_distance"], uv, 0) <= 1.0) & (nearest_z < z - 0.65)
        actor_near = actor_distances(world[:, :2], f, exclude_target=True) < CONFIG["other_actor_footprint_clearance_m"]
        actor_near |= footprint_distance(world[:, :2], f["target"]) < CONFIG["target_footprint_clearance_m"]
        guard = sample(source["guard"], measured_uv, 0) != 0
        ambiguous = (sample(f["city"], measured_uv, 0) >= 11) | (sample(f["confidence"], measured_uv, 0) < 0.70)
        ambiguous |= sample(source["patch_std"], measured_uv, 0) > 0.045
        eligible = inside & surface_ok & ~inactive & ~wide & ~moved & ~foreground & ~actor_near & ~guard & ~ambiguous
        values = sample(f["rgb"], measured_uv) * gain
        native = data["rgb"] * gain
        error = np.abs(native - values)
        match = (error.mean(1) <= 0.060) & (error.max(1) <= 0.120) & np.isfinite(error).all(1)
        geometry[si] = eligible
        raw_rgb[si, eligible] = values[eligible]
        sh_error[si, eligible] = error[eligible].mean(1)
        sh_match[si] = eligible & match
        surface_distance[si] = distance
        observed_uv[si] = measured_uv
        for key, mask in [("row_inactive", inactive), ("row_too_wide_for_center_surface_test", wide),
                          ("row_world_position_changed", moved), ("not_in_front_camera", ~inside),
                          ("no_close_actual_visible_lidar_surface", ~surface_ok), ("foreground_depth_occlusion", foreground),
                          ("target_or_other_actor_near", actor_near), ("body_or_known_shadow_region", guard),
                          ("semantic_or_patch_ambiguous", ambiguous), ("source_sh_rgb_mismatch", eligible & ~match)]:
            reason[si, mask] |= SOURCE_REASONS[key]
    origins = np.array(origins)
    timestamps = np.array(timestamps)
    raws = np.array(sources, np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(raw_rgb, axis=0)
        deviations = np.mean(np.abs(raw_rgb - median[None]), axis=2)
        median_dev = np.nanmedian(deviations, axis=0)
        p90_dev = np.nanpercentile(deviations, 90, axis=0)
        median_sh_error = np.nanmedian(sh_error, axis=0)
    stable = np.isfinite(median_dev) & (median_dev <= 0.035) & (p90_dev <= 0.060)
    pair_geometry, geometry_raws, span, baseline, angle = independent_pair(geometry, raws, origins, timestamps, world)
    pair_match, matched_raws, match_span, match_baseline, match_angle = independent_pair(sh_match, raws, origins, timestamps, world)
    geometry_witness = pair_geometry & stable
    fraction = sh_match.sum(0) / np.maximum(geometry.sum(0), 1)
    query_error = np.abs(q["rgb"] - median)
    query_match = (query_error.mean(1) <= 0.060) & (query_error.max(1) <= 0.120) & np.isfinite(query_error).all(1)
    source_match = pair_match & (fraction >= 0.80)
    protect = geometry_witness & source_match & query_match & (q["opacity"] >= 0.05) & (q["sigma_m"] <= 0.20)
    mixed = geometry_witness & (~source_match | ~query_match)
    state = np.zeros(n, np.uint8)
    state[mixed] = 2
    state[protect] = 1
    row_reason = np.full(n, ROW_REASONS["source_shadow_absence_unverified"], np.uint16)
    for key, mask in [("insufficient_independent_visible_sources", ~pair_geometry), ("raw_source_rgb_not_stable", ~stable),
                      ("source_sh_rgb_mismatch", geometry_witness & ~source_match), ("query_sh_rgb_mismatch", geometry_witness & ~query_match),
                      ("positive_background_geometry_witness", geometry_witness), ("positive_background_protection_candidate", protect),
                      ("row_wide_or_inactive", (q["sigma_m"] > 0.20) | (q["opacity"] < 0.05))]:
        row_reason[mask] |= ROW_REASONS[key]
    strength = np.where(protect, np.clip(1 - query_error.mean(1) / 0.12, 0, 1) * np.clip(geometry.sum(0) / 4, 0, 1) * np.clip(1 - median_dev / 0.07, 0, 1), 0).astype(np.float32)
    arrays = {"ids": ids, "background_protection_candidate": protect, "background_protection_strength": strength,
              "background_geometry_witness": geometry_witness, "mixed_color_mismatch": mixed,
              "unknown": ~(protect | mixed), "row_evidence_state": state,
              "background_certified": np.zeros(n, bool), "source_shadow_state_unknown": np.array(True),
              "source_raws": raws, "source_geometry_valid": geometry.T, "source_sh_match": sh_match.T,
              "source_reason_bits": reason.T, "source_count": geometry.sum(0).astype(np.uint8),
              "source_sh_match_count": sh_match.sum(0).astype(np.uint8), "source_sh_match_fraction": fraction.astype(np.float32),
              "source_surface_distance_m": surface_distance.T, "source_actual_surface_uv": observed_uv.transpose(1, 0, 2),
              "source_original_rgb_exposure_matched": raw_rgb.transpose(1, 0, 2),
              "stable_original_rgb_reference": median.astype(np.float32), "original_rgb_median_deviation": median_dev.astype(np.float32),
              "original_rgb_p90_deviation": p90_dev.astype(np.float32), "query_sh_rgb_mean_abs_error": query_error.mean(1).astype(np.float32),
              "source_sh_rgb_median_abs_error": median_sh_error.astype(np.float32),
              "independent_geometry_source_pair": geometry_raws, "independent_matched_source_pair": matched_raws,
              "independent_time_span_s": match_span.astype(np.float32), "independent_camera_baseline_m": match_baseline.astype(np.float32),
              "independent_view_angle_deg": match_angle.astype(np.float32), "reason_bits": row_reason}
    np.savez_compressed(output / ("%03d.npz" % raw), **arrays)
    examples = []
    for category, mask in [("background_protection_candidate", protect), ("mixed_color_mismatch", mixed)]:
        choices = np.flatnonzero(mask)
        for index in choices[np.linspace(0, len(choices) - 1, min(8, len(choices))).astype(int)] if len(choices) else []:
            examples.append({"category": category, "id": int(ids[index]), "query_world": world[index].tolist(),
                             "query_sh_rgb": q["rgb"][index].tolist(), "raw_reference_rgb": median[index].tolist(),
                             "source_pair": matched_raws[index].tolist() if category == "background_protection_candidate" else geometry_raws[index].tolist(),
                             "query_sh_rgb_error": float(query_error[index].mean()), "source_count": int(geometry[:, index].sum()),
                             "source_sh_match_count": int(sh_match[:, index].sum()), "source_shadow_state": "unknown"})
    report = {"raw": raw, "rows": n, "background_protection_candidate_rows": int(protect.sum()),
              "background_geometry_witness_rows": int(geometry_witness.sum()), "mixed_color_mismatch_rows": int(mixed.sum()),
              "unknown_rows": int((state == 0).sum()), "certified_background_rows": 0,
              "exposure_sources": exposure, "examples": examples,
              "source_missing_reason_counts": {k: int(((reason & bit) != 0).sum()) for k, bit in SOURCE_REASONS.items()},
              "row_reason_counts": {k: int(((row_reason & bit) != 0).sum()) for k, bit in ROW_REASONS.items()},
              "source_raws": sources, "wall_seconds": time.monotonic() - started,
              "source_shadow_state_unknown": True, "human_GT": None}
    (output / ("%03d.json" % raw)).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", nargs="+", type=int, default=[98, 113, 114])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    exported = json.loads((args.rows / "contract.json").read_text())
    if exported["namespace"] != "original_static_rows" or not exported["source_checkpoint_unchanged"]:
        raise ValueError("原static行命名空间/冻结合同不符")
    ids = np.load(args.rows / "ids.npy")
    assert (np.diff(ids) > 0).all()
    sources = sorted(int(r) for r in exported["raws"])
    row_data = {}
    for raw in sources:
        with np.load(args.rows / ("%03d.npz" % raw)) as f:
            row_data[raw] = {k: f[k].copy() for k in ["world", "rgb", "sigma_m", "opacity"]}
        assert row_data[raw]["world"].shape == (len(ids), 3)
        assert row_data[raw]["world"].dtype == np.float64
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(CONFIG)
    contract.update({"root": str(args.root), "row_export": str(args.rows), "shadow_observations": str(args.shadow),
                     "source_raws": sources, "query_raws": args.raw, "row_namespace": "original_static_rows",
                     "source_reason_bits": SOURCE_REASONS, "row_reason_bits": ROW_REASONS,
                     "state_meaning": {"0": "unknown/missing", "1": "positive background protection candidate", "2": "background geometry witnessed but color mixed; not car"}})
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    started = time.monotonic()
    clip = RawClip(args.root)
    frames = WitnessFrames(clip, args.shadow)
    reports = []
    for raw in args.raw:
        report = query_witnesses(raw, ids, row_data, sources, frames, clip, args.shadow, args.output)
        reports.append(report)
        print("ROW_BACKGROUND", json.dumps({k: report[k] for k in ["raw", "rows", "background_geometry_witness_rows", "background_protection_candidate_rows", "mixed_color_mismatch_rows", "unknown_rows", "wall_seconds"]}), flush=True)
    summary = {"schema": CONFIG["schema"], "queries": reports, "wall_seconds": time.monotonic() - started,
               "training_updates": 0, "GPU_calls": 0, "certified_background_rows": 0,
               "source_shadow_state_unknown": True, "human_GT": None}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
