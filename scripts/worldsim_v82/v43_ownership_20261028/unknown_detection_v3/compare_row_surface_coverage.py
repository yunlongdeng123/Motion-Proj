"""固定query世界位置的原始表面覆盖对照；不外推Gaussian运动或源SH。

保持49源严格方案，只诊断真实源时段扩展是否补充可见LiDAR/稳定颜色。
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
from scipy.ndimage import binary_dilation

from build_shadow_evidence import DEFAULT_ROOT, RawClip, actor_distances, fit_exposure, footprint_distance, project, sample
from build_row_background_counterevidence import (CONFIG, WitnessFrames,
                                                   exposure_anchors,
                                                   independent_pair)


class CoverageFrames(WitnessFrames):
    def get(self, raw):
        value = super().get(raw)
        observed = (self.shadow / ("%03d.npz" % raw)).exists()
        if not observed:
            # 额外时刻没有独立阴影图，不能伪造negative；只作未认证表面见证。
            f = value["frame"]
            value["guard"] = binary_dilation(f["body"], iterations=3) | f["dynamic_guard"]
        value["independent_shadow_map_available"] = observed
        return value


def summarize_subset(selection, sources, geometry, rgb, origins, timestamps, world, shadow_available, raw_proximity):
    ix = np.flatnonzero(selection)
    valid = geometry[ix]
    values = rgb[ix]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        deviation = np.mean(np.abs(values - median[None]), axis=2)
        mad = np.nanmedian(deviation, axis=0)
        p90 = np.nanpercentile(deviation, 90, axis=0)
    stable = np.isfinite(mad) & (mad <= CONFIG["max_raw_rgb_median_deviation"]) & (p90 <= CONFIG["max_raw_rgb_p90_deviation"])
    independent, pair, span, baseline, angle = independent_pair(valid, np.array(sources, np.int16)[ix], origins[ix], timestamps[ix], world)
    witness = independent & stable
    return {"raw_surface_witness": witness, "source_count": valid.sum(0).astype(np.uint8),
            "raw_geometric_proximity_count": raw_proximity[ix].sum(0).astype(np.uint8),
            "independent_geometry_pair": independent, "raw_color_stable": stable,
            "reference_rgb": median.astype(np.float32), "rgb_median_deviation": mad.astype(np.float32),
            "rgb_p90_deviation": p90.astype(np.float32), "source_pair": pair,
            "time_span_s": span.astype(np.float32), "camera_baseline_m": baseline.astype(np.float32),
            "view_angle_deg": angle.astype(np.float32),
            "support_from_shadow_unobserved_sources": (valid & ~shadow_available[ix, None]).sum(0).astype(np.uint8)}


def compare_query(raw, query_rows, ids, source_raws, clip, frames, shadow, strict_bg, native, output):
    started = time.monotonic()
    query = clip.get(raw)
    q = query_rows[raw]
    world, sigma = q["world"], q["sigma_m"]
    n, ns = len(ids), len(source_raws)
    plane_json = json.loads((shadow / ("%03d.json" % raw)).read_text())["plane"]
    plane = {"origin": np.array(plane_json["origin_world"]), "coefficients": np.array(plane_json["z_minus_origin_z_equals"])}
    anchors, anchor_rgb = exposure_anchors(query, plane)
    geometry = np.zeros((ns, n), bool)
    proximity = np.zeros((ns, n), bool)
    in_view = np.zeros((ns, n), bool)
    in_lidar_range_view = np.zeros((ns, n), bool)
    rgb = np.full((ns, n, 3), np.nan, np.float32)
    distance_matrix = np.full((ns, n), np.nan, np.float32)
    uv_matrix = np.full((ns, n, 2), np.nan, np.float32)
    reason = np.zeros((ns, n), np.uint16)
    origins, timestamps, shadow_available, lidar_max_depth = [], [], [], []
    exposure = {}
    for si, sr in enumerate(source_raws):
        source = frames.get(sr)
        f = source["frame"]
        origins.append(source["camera_origin"])
        timestamps.append(f["meta"]["timestamp_micros"] / 1e6)
        shadow_available.append(source["independent_shadow_map_available"])
        if "max_actual_camera_depth" not in source:
            source["max_actual_camera_depth"] = float(project(source["surface_world"], f)[1].max())
        lidar_max_depth.append(source["max_actual_camera_depth"])
        if sr == raw:
            reason[si] |= 1
            continue
        uv, z = project(world, f)
        inside = (z > 0.1) & np.isfinite(uv).all(1) & (uv[:, 0] >= 2) & (uv[:, 0] < 478) & (uv[:, 1] >= 2) & (uv[:, 1] < 318)
        in_view[si] = inside
        in_lidar_range_view[si] = inside & (z <= source["max_actual_camera_depth"] + 1.0)
        distance, nearest = source["tree"].query(world, workers=2)
        measured_uv = source["surface_uv"][nearest]
        tolerance = np.maximum(0.08, np.minimum(0.20, 2 * sigma))
        surface_ok = (distance <= tolerance) & (np.linalg.norm(measured_uv - uv, axis=1) <= 2)
        proximity[si] = inside & surface_ok
        distance_matrix[si] = np.where(inside, distance, np.nan)
        uv_matrix[si] = measured_uv
        wide = sigma > CONFIG["max_sigma_for_protection_m"]
        foreground = (sample(f["all_lidar_distance"], uv, 0) <= 1.0) & (sample(f["all_lidar_depth"], uv, 0) < z - 0.65)
        actor_near = actor_distances(world[:, :2], f, exclude_target=True) < CONFIG["other_actor_footprint_clearance_m"]
        actor_near |= footprint_distance(world[:, :2], f["target"]) < CONFIG["target_footprint_clearance_m"]
        guard = sample(source["guard"], measured_uv, 0) != 0
        ambiguous = (sample(f["city"], measured_uv, 0) >= 11) | (sample(f["confidence"], measured_uv, 0) < CONFIG["minimum_semantic_confidence"])
        ambiguous |= sample(source["patch_std"], measured_uv, 0) > CONFIG["max_patch_rgb_std"]
        gain, exp = fit_exposure(query, f, anchors, anchor_rgb)
        exposure[str(sr)] = exp
        visible = inside & surface_ok & ~wide & ~foreground & ~actor_near & ~guard & ~ambiguous
        if gain is None:
            visible[:] = False
            reason[si] |= 512
        else:
            values = sample(f["rgb"], measured_uv) * gain
            geometry[si] = visible
            rgb[si, visible] = values[visible]
        for bit, mask in [(2, wide), (4, ~inside), (8, ~surface_ok), (16, foreground),
                          (32, actor_near), (64, guard), (128, ambiguous)]:
            reason[si, mask] |= bit
        if not source["independent_shadow_map_available"]:
            reason[si] |= 256
    origins, timestamps = np.array(origins), np.array(timestamps)
    shadow_available = np.array(shadow_available, bool)
    raws = np.array(source_raws, np.int16)
    local = (raws >= 77) & (raws <= 125)
    stride = raws % 4 == 0
    selections = {"local49_all": local, "local49_stride4": local & stride,
                  "full199_stride4": stride, "union87": np.ones(ns, bool)}
    stages = {name: summarize_subset(selection, source_raws, geometry, rgb, origins, timestamps, world, shadow_available, proximity)
              for name, selection in selections.items()}
    current_u = np.zeros(n, bool)
    if native is not None:
        with np.load(native / ("%03d.npz" % raw)) as e:
            ix = np.searchsorted(ids, e["ids"])
            assert np.array_equal(ids[ix], e["ids"])
            current_u[ix] = e["static_role"] >= 2
    with np.load(strict_bg / ("%03d.npz" % raw)) as old:
        assert np.array_equal(ids, old["ids"])
        old_geometry = old["background_geometry_witness"]
        old_protection = old["background_protection_candidate"]
    local_w = stages["local49_all"]["raw_surface_witness"]
    full_w = stages["full199_stride4"]["raw_surface_witness"]
    union_w = stages["union87"]["raw_surface_witness"]
    expanded = full_w & ~local_w
    added_union = union_w & ~local_w
    no_raw_match = ~proximity.any(0)
    observed_views = in_view.sum(0)
    no_surface_observed = no_raw_match & (observed_views >= 2)
    no_views = observed_views < 2
    no_surface_in_range = no_raw_match & (in_lidar_range_view.sum(0) >= 2)
    insufficient_range_views = in_lidar_range_view.sum(0) < 2
    excluded_or_exposure = proximity.any(0) & (geometry.sum(0) < 2)
    arrays = {"ids": ids, "query_world_fixed": world, "query_sigma_m": sigma,
              "query_native_opacity": q["opacity"], "source_raws": raws,
              "source_geometry_valid": geometry.T, "source_raw_geometric_proximity": proximity.T,
              "source_in_front_camera": in_view.T, "source_reason_bits": reason.T,
              "source_in_observed_lidar_range_view": in_lidar_range_view.T,
              "source_max_actual_camera_depth_m": np.array(lidar_max_depth, np.float32),
              "source_actual_surface_distance_m": distance_matrix.T,
              "source_actual_surface_uv": uv_matrix.transpose(1, 0, 2),
              "source_original_rgb_exposure_matched": rgb.transpose(1, 0, 2),
              "source_independent_shadow_map_available": shadow_available,
              "source_sh_matching_performed": np.array(False), "native_motion_extrapolated": np.array(False),
              "background_certified": np.zeros(n, bool), "current_native_u": current_u,
              "strict49_background_geometry_witness": old_geometry,
              "strict49_background_protection_candidate": old_protection,
              "extended_surface_witness_vs_local49": expanded,
              "added_union_surface_witness_vs_local49": added_union,
              "no_actual_visible_surface_match_observed": no_surface_observed,
              "insufficient_raw_view_coverage": no_views,
              "no_actual_surface_match_with_two_lidar_range_views": no_surface_in_range,
              "insufficient_view_or_lidar_range_coverage": insufficient_range_views,
              "actual_surface_near_but_excluded_or_exposure_missing": excluded_or_exposure}
    for name, stage in stages.items():
        for key, value in stage.items():
            arrays[name + "_" + key] = value
    assert np.all(stages["full199_stride4"]["source_count"] >= stages["local49_stride4"]["source_count"])
    assert np.all(stages["union87"]["source_count"] >= stages["local49_all"]["source_count"])
    assert not arrays["background_certified"].any()
    for stage in stages.values():
        positive = stage["raw_surface_witness"]
        assert np.all(stage["source_count"][positive] >= 2)
        assert np.all(stage["time_span_s"][positive] >= 0.499)
        assert np.all(stage["camera_baseline_m"][positive] >= 1)
        assert np.all(stage["view_angle_deg"][positive] >= 1)
    np.savez_compressed(output / ("%03d.npz" % raw), **arrays)
    stage_report = {}
    for name, stage in stages.items():
        mask = stage["raw_surface_witness"]
        stage_report[name] = {"source_frames": int(selections[name].sum()),
                              "raw_geometric_proximity_rows": int((stage["raw_geometric_proximity_count"] > 0).sum()),
                              "at_least_two_usable_source_rows": int((stage["source_count"] >= 2).sum()),
                              "independent_visible_geometry_rows": int(stage["independent_geometry_pair"].sum()),
                              "stable_raw_surface_witness_rows": int(mask.sum()),
                              "stable_witness_intersection_current_u": int((mask & current_u).sum()),
                              "witness_using_shadow_unobserved_sources": int((mask & (stage["support_from_shadow_unobserved_sources"] > 0)).sum())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        min_distance = np.nanmin(distance_matrix, axis=0)
    report = {"raw": raw, "rows": n, "current_u_rows": int(current_u.sum()), "stages": stage_report,
              "strict49_reference": {"background_geometry_witness": int(old_geometry.sum()), "background_protection_candidate": int(old_protection.sum())},
              "extended_full_stride4_witness_vs_local49": int(expanded.sum()),
              "extended_full_stride4_witness_intersection_u": int((expanded & current_u).sum()),
              "union_added_witness_vs_local49": int(added_union.sum()),
              "union_added_witness_intersection_u": int((added_union & current_u).sum()),
              "no_near_actual_surface_observed_despite_two_inview_frames": int(no_surface_observed.sum()),
              "no_near_actual_surface_observed_intersection_u": int((no_surface_observed & current_u).sum()),
              "insufficient_raw_view_coverage_rows": int(no_views.sum()),
              "no_actual_surface_match_with_two_lidar_range_views": int(no_surface_in_range.sum()),
              "no_actual_surface_match_with_two_lidar_range_views_intersection_u": int((no_surface_in_range & current_u).sum()),
              "insufficient_view_or_lidar_range_coverage": int(insufficient_range_views.sum()),
              "insufficient_view_or_lidar_range_coverage_intersection_u": int((insufficient_range_views & current_u).sum()),
              "actual_surface_near_but_excluded_or_exposure_missing": int(excluded_or_exposure.sum()),
              "actual_surface_near_but_excluded_or_exposure_missing_intersection_u": int((excluded_or_exposure & current_u).sum()),
              "minimum_actual_surface_distance_u_p10_p50_p90_m": np.nanpercentile(min_distance[current_u], [10, 50, 90]).tolist() if current_u.any() else None,
              "exposure_sources": exposure,
              "source_sh_matching_performed": False, "native_motion_extrapolated": False,
              "certified_background_rows": 0, "human_GT": None, "wall_seconds": time.monotonic() - started,
              "interpretation_limit": "no raw surface match can reflect model-center displacement, LiDAR range/sparsity or occlusion; even in-range absence does not prove Gaussian geometry wrong and is never car/U positive evidence"}
    (output / ("%03d.json" % raw)).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--shadow", type=Path, required=True)
    p.add_argument("--strict-background", type=Path, required=True)
    p.add_argument("--native-evidence", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw", type=int, nargs="+", default=[98, 113, 114])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    export = json.loads((a.rows / "contract.json").read_text())
    assert export["namespace"] == "original_static_rows"
    ids = np.load(a.rows / "ids.npy")
    query_rows = {}
    for raw in a.raw:
        with np.load(a.rows / ("%03d.npz" % raw)) as q:
            query_rows[raw] = {k: q[k].copy() for k in ["world", "sigma_m", "opacity"]}
        assert query_rows[raw]["world"].dtype == np.float64
    full_stride4 = list(range(0, 199, 4))
    local49 = list(range(77, 126))
    sources = sorted(set(local49) | set(full_stride4))
    a.output.mkdir(parents=True, exist_ok=False)
    contract = {"schema": "fixed-query-world-original-surface-coverage-v1", "frozen_thresholds": CONFIG,
                "row_export": str(a.rows), "strict_49_reference_preserved": str(a.strict_background),
                "query_raws": a.raw, "sources_evaluated": sources,
                "comparisons": {"local49_all": local49, "local49_stride4": [r for r in local49 if r % 4 == 0],
                                "full199_stride4": full_stride4, "union87": sources},
                "fixed_query_world_position": True, "source_gaussian_world_or_opacity_used": False,
                "source_sh_matching_performed": False, "native_motion_extrapolated": False,
                "query_opacity_used_for_geometry_gate": False,
                "source_shadow_map_outside49": "missing/unknown; known maps applied where available, same target/other-actor clearance retained",
                "output_role": "original surface witness extension only, never certified background protection",
                "source_reason_bits": {"same_query": 1, "query_sigma_wide": 2, "out_of_view": 4, "no_actual_surface_match": 8,
                                       "foreground_occlusion": 16, "actor_near": 32, "body_or_known_shadow": 64,
                                       "semantic_or_texture_ambiguous": 128, "independent_shadow_map_missing": 256,
                                       "exposure_fit_missing": 512},
                "GPU_calls": 0, "training_updates": 0, "human_GT": None}
    (a.output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    started = time.monotonic()
    clip = RawClip(a.root)
    frames = CoverageFrames(clip, a.shadow)
    results = []
    for raw in a.raw:
        report = compare_query(raw, query_rows, ids, sources, clip, frames, a.shadow, a.strict_background, a.native_evidence, a.output)
        results.append(report)
        print("SURFACE_COVERAGE", json.dumps({"raw": raw, "stages": report["stages"],
                                              "union_added_witness_vs_local49": report["union_added_witness_vs_local49"],
                                              "union_added_witness_intersection_u": report["union_added_witness_intersection_u"]}), flush=True)
    summary = {"schema": contract["schema"], "queries": results, "wall_seconds": time.monotonic() - started,
               "GPU_calls": 0, "training_updates": 0, "source_sh_matching_performed": False,
               "native_motion_extrapolated": False, "certified_background_rows": 0, "human_GT": None}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
