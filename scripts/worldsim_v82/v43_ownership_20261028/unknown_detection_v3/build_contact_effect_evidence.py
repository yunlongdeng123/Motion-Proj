"""原始近地 LiDAR + 同世界原始光度的小规模 contact effect 证据。

不把黑色/近车本身当证据；不膨胀，不补全完整阴影，不排除所有 body。
近地回波仍可能来自轮胎接地点，输出只作低置信 effect 候选。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "2"

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from build_shadow_evidence import (DEFAULT_ROOT, CTX, RawClip, actor_distances,
                                   as_map, collect_references, darkening,
                                   plane_height, project, sample)

CONTRACT = {
    "schema": "sparse-original-contact-effect-v1", "context": CTX,
    "actual_return_height_tolerance_m": 0.05,
    "search_band": "body bbox x +/-4 pixels, bottom y [-8,+5]; only a search domain, never positive evidence",
    "actual_return_visibility": "frontmost original LiDAR return in each camera pixel",
    "reference": ">=3 original target-away road-surface views spanning >=0.5s; actual source LiDAR depth support; original exposure-matched RGB; unstable reference abstains",
    "photo_thresholds": "same as strict r2; no relaxed ratio/drop/attenuation threshold",
    "query_semantic_road_required": False, "body_hard_exclusion": False,
    "score_cap": 0.49, "dilation_or_completion": False,
    "full_shadow_segmentation_claim": False,
    "near_ground_return_may_be_tire_contact": True,
    "source_shadow_state_unknown": True, "human_GT": None,
    "native_model_extrapolated": False, "GPU_calls": 0, "training_updates": 0,
    "inputs": "original raw RGB/LiDAR/pose/calibration/track and independently fitted r2 road plane; never deletion/inpainting/render RGB",
}


def run_query(clip, raw, sources, strict, output):
    frame = clip.get(raw)
    report = json.loads((strict / ("%03d.json" % raw)).read_text())
    plane = {"origin": np.array(report["plane"]["origin_world"]),
             "coefficients": np.array(report["plane"]["z_minus_origin_z_equals"])}
    points = np.fromfile(clip.source / "velodyne" / (frame["meta"]["frame_id"] + ".bin"), np.float32).reshape(-1, 6)[:, :3].astype(np.float64)
    world = points @ frame["pose"][:3, :3].T + frame["pose"][:3, 3]
    uv, z = project(world, frame)
    ok = (z > 0.1) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < 480) & (uv[:, 1] >= 0) & (uv[:, 1] < 320)
    ix = np.flatnonzero(ok)
    pix = np.floor(uv[ix]).astype(int)
    pixel_id = pix[:, 1] * 480 + pix[:, 0]
    order = np.lexsort((z[ix], pixel_id))
    _, first = np.unique(pixel_id[order], return_index=True)
    ix = ix[order[first]]
    pix = np.floor(uv[ix]).astype(int)
    yy, xx = np.where(frame["body"])
    cy, cx = yy.max(), (xx.min() + xx.max()) / 2
    Y, X = np.mgrid[:320, :480]
    domain = (X >= xx.min() - 4) & (X <= xx.max() + 4) & (Y >= cy - 8) & (Y <= cy + 5)
    height = world[ix, 2] - plane_height(world[ix, :2], plane)
    keep = domain[pix[:, 1], pix[:, 0]] & (np.abs(height) <= CONTRACT["actual_return_height_tolerance_m"])
    keep &= actor_distances(world[ix, :2], frame, exclude_target=True) >= 1.0
    if not report["plane"]["valid"]:
        keep[:] = False
    ix, pix, height = ix[keep], pix[keep], height[keep]
    flat_pixel = pix[:, 1] * 480 + pix[:, 0]
    gains = {int(k): np.array(v["gain_rgb_source_to_query"]) for k, v in report["exposure_sources"].items() if v["valid"]}
    reference = collect_references(world[ix], sources, gains, excluded_raw=raw)
    original_rgb = sample(frame["rgb"], uv[ix])
    supported, strength, ratio, drop, residual = darkening(original_rgb, reference)
    score = np.where(supported, np.minimum(0.49, strength * 0.49), 0).astype(np.float32)
    reason = np.full(len(ix), 128 | 256, np.uint16)
    reason[~reference["enough"]] |= 8
    reason[~reference["stable"]] |= 16
    reason[~supported] |= 32
    arrays = {
        "body": frame["body"], "contact_search_domain": domain,
        "contact_ground_photo_effect": as_map(supported, flat_pixel),
        "contact_ground_photo_effect_score": as_map(score, flat_pixel),
        "contact_ground_photo_effect_valid": as_map(reference["valid"], flat_pixel),
        "actual_near_ground_return": as_map(np.ones(len(ix), bool), flat_pixel),
        "actual_return_height_m": as_map(height, flat_pixel, np.float32, np.nan),
        "actual_return_world": as_map(world[ix], flat_pixel, np.float64, np.nan),
        "actual_return_uv": as_map(uv[ix], flat_pixel, np.float32, np.nan),
        "actual_lidar_row_index": as_map(ix.astype(np.int32), flat_pixel, np.int32, -1),
        "source_raw": as_map(reference["source_raw"], flat_pixel, np.int16, -1),
        "source_count": as_map(reference["count"], flat_pixel),
        "reference_rgb": as_map(reference["rgb"], flat_pixel, np.float32, np.nan),
        "reference_rgb_deviation": as_map(reference["deviation"], flat_pixel, np.float32, np.nan),
        "original_rgb_at_return_uv": as_map(original_rgb, flat_pixel, np.float32, np.nan),
        "luminance_ratio": as_map(ratio, flat_pixel, np.float32, np.nan),
        "luminance_drop": as_map(drop, flat_pixel, np.float32, np.nan),
        "attenuation_rgb_residual": as_map(residual, flat_pixel, np.float32, np.nan),
        "reason_bits": as_map(reason, flat_pixel, np.uint16, 1 | 2),
        "source_shadow_state_unknown": np.array(True), "full_shadow_segmentation": np.array(False),
    }
    effect = arrays["contact_ground_photo_effect"]
    arrays["body_overlap_contact_effect"] = effect & frame["body"]
    np.savez_compressed(output / ("%03d.npz" % raw), **arrays)
    image = frame["rgb"] * 255
    image[effect] = image[effect] * 0.25 + np.array([205, 60, 245]) * 0.75
    Image.fromarray(np.round(image).astype(np.uint8)).save(output / ("%03d_overlay.png" % raw))
    result = {"raw": raw, "actual_near_ground_contact_pixels": len(ix),
              "same_world_reference_valid_pixels": int(reference["valid"].sum()),
              "contact_effect_candidate_pixels": int(effect.sum()),
              "candidate_inside_body_pixels": int((effect & frame["body"]).sum()),
              "candidate_outside_body_pixels": int((effect & ~frame["body"]).sum()),
              "candidate_left_pixels": int((effect & (X < cx)).sum()),
              "candidate_right_pixels": int((effect & (X >= cx)).sum()),
              "inside_body_right_candidates": int((effect & frame["body"] & (X >= cx)).sum()),
              "plane": report["plane"], "source_raws": [f["raw"] for f in sources],
              "human_GT": None, "score_cap": 0.49,
              "uncertainty": "near-zero-height return may be tire contact; no complete-shadow or physical-causality claim"}
    (output / ("%03d.json" % raw)).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=int, nargs="+", default=[98, 113, 114])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    clip = RawClip(args.root)
    strict_contract = json.loads((args.strict / "contract.json").read_text())
    sources = [clip.get(raw) for raw in strict_contract["source_raws"]]
    contract = dict(CONTRACT)
    contract.update(root=str(args.root), strict_input=str(args.strict), query_raws=args.raw,
                    source_raws=strict_contract["source_raws"],
                    reason_bits={"outside_search_or_no_direct_near_ground_return": 3, "insufficient_target_away_views": 8,
                                 "unstable_reference": 16, "no_independent_darkening": 32,
                                 "tire_vs_road_unresolved": 128, "source_shadow_unknown": 256})
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    results = []
    for raw in args.raw:
        result = run_query(clip, raw, sources, args.strict, args.output)
        results.append(result)
        print("CONTACT_EFFECT", json.dumps({k: result[k] for k in ["raw", "actual_near_ground_contact_pixels", "contact_effect_candidate_pixels", "candidate_inside_body_pixels", "candidate_right_pixels"]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"schema": CONTRACT["schema"], "queries": results,
                  "wall_seconds": time.monotonic() - started, "GPU_calls": 0, "training_updates": 0, "human_GT": None}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
