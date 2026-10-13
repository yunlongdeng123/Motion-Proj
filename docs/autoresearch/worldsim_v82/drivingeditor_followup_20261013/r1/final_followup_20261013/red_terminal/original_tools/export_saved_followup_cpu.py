"""Export a followup scene's saved FP32 evaluation using CPU arrays only."""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(path, root):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), "saved source path escaped run: " + str(path))
    return path


def validate_arrays(arrays, width, height, static_time=False):
    schema = {"static_rgb": 3, "static_alpha": 1} if static_time else {
        "gt": 3, "composite": 3, "static": 3, "dynamic": 3, "alpha_s": 1,
        "alpha_d": 1, "depth_accum_model": 1, "actor_mask": 1, "sky_mask": 1}
    require(set(arrays) == set(schema), "saved NPZ schema mismatch")
    for key, channels in schema.items():
        value = arrays[key]
        require(value.shape == (channels, height, width), "shape mismatch: " + key)
        require(value.dtype == (np.bool_ if key.endswith("mask") else np.float32),
                "dtype mismatch: " + key)
        require(np.isfinite(value).all(), "nonfinite saved array: " + key)


def png(path, array, rgb=False):
    value = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    value = value.transpose(1, 2, 0) if rgb else value.squeeze(0)
    with path.open("xb") as stream:
        Image.fromarray(value).save(stream, format="PNG", compress_level=4)


def bbox(mask, width, height):
    yy, xx = np.where(mask.squeeze(0))
    if not len(xx):
        return None
    cx, cy = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2
    crop_w = min(width, max(140, int(xx.max() - xx.min() + 1) + 40))
    crop_h = min(height, max(94, int(yy.max() - yy.min() + 1) + 40))
    x = max(0, min(width - crop_w, int(round(cx - crop_w / 2))))
    y = max(0, min(height - crop_h, int(round(cy - crop_h / 2))))
    return [x, y, x + crop_w, y + crop_h]


def region_metrics(arrays):
    actor, sky = arrays["actor_mask"].squeeze(0), arrays["sky_mask"].squeeze(0)
    inside_distance = distance_transform_edt(actor)
    outside_distance = distance_transform_edt(~actor)
    masks = {"full": np.ones(actor.shape, dtype=bool), "actor": actor,
             "non_sky_outside_actor": ~actor & ~sky,
             "actor_core_5px": actor & (inside_distance > 5),
             "actor_inner_boundary_5px": actor & (inside_distance <= 5),
             "outside_boundary_5px": ~actor & ~sky & (outside_distance <= 5),
             "outside_far_5px": ~actor & ~sky & (outside_distance > 5)}
    gt = arrays["gt"].clip(0, 1).astype(np.float64)
    composite = arrays["composite"].clip(0, 1).astype(np.float64)
    static = arrays["static"].clip(0, 1).astype(np.float64)
    dynamic = arrays["dynamic"].clip(0, 1)
    alpha_d, alpha_s = arrays["alpha_d"].squeeze(0), arrays["alpha_s"].squeeze(0)
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        row = {"pixel_count": count}
        if count:
            for key, image in (("composite", composite), ("static", static)):
                mse = float(np.square(image[:, mask] - gt[:, mask]).mean())
                row[key + "_mse"] = mse
                row[key + "_psnr_db"] = -10 * float(np.log10(max(mse, 1e-12)))
            row.update(alpha_d_mean=float(alpha_d[mask].mean()), alpha_s_mean=float(alpha_s[mask].mean()),
                       dynamic_RGB_support_8bit_gt2=float((np.rint(dynamic[:, mask] * 255) > 2).any(axis=0).mean()),
                       composite_static_abs_mean=float(np.abs(composite[:, mask] - static[:, mask]).mean()))
        result[name] = row
    return result


def match_regions(actual, saved):
    require(set(actual) == set(saved), "region schema mismatch")
    maximum = 0.0
    checked = 0
    for name, row in actual.items():
        stored = saved[name]
        require(row["pixel_count"] == stored["pixel_count"], "region pixel count mismatch: " + name)
        if not row["pixel_count"]:
            require(all(value is None for key, value in stored.items() if key != "pixel_count"),
                    "empty region contains a numeric metric: " + name)
            continue
        require(set(row) == set(stored), "nonempty region metric schema mismatch: " + name)
        for key, value in row.items():
            if key == "pixel_count":
                continue
            require(isinstance(stored[key], (int, float)) and math.isfinite(stored[key]),
                    "nonempty stored metric is not finite: " + name + "." + key)
            error = abs(value - stored[key])
            maximum = max(maximum, error)
            require(math.isclose(value, stored[key], rel_tol=1e-9, abs_tol=1e-10),
                    "saved FP32 metric replay mismatch: " + name + "." + key)
            checked += 1
    return checked, maximum


def validate_contract(receipt, source, spec):
    require(receipt["status"] == "FOLLOWUP_COMPLETE_CONTRACT_PASS", "inherited completion contract did not pass")
    require(receipt["output"] == str(source) and receipt["scene_id"] == spec["scene_id"],
            "inherited completion contract source mismatch")
    require(receipt.get("failure_fields") == [] and receipt["restoration_verified"] is False,
            "inherited contract scope mismatch")
    require(set(receipt["nodes"]) == {"10000", "20000", "30000"}, "three node contract missing")
    for key, node in receipt["nodes"].items():
        require(node["status"] == "PASS_CHECKPOINT_FILE_AND_FRAME_CONTRACT" and
                node["unique_chronological_FRONT_frames"] == 49 and node["failure_fields"] == [],
                "inherited node contract failed: " + key)
    # The original completion_contract was invoked by the terminal audit once.
    # Reuse that actual result instead of repeating its full stat/header scan.
    require(receipt["original_completion_contract_actual_result"]["nodes"] == [10000, 20000, 30000],
            "actual original completion contract result missing")


def aggregate_regions(frames):
    result = {}
    for name in frames[0]["metrics"]:
        valid = [f for f in frames if f["metrics"][name]["pixel_count"] > 0]
        empty = [f["id"] for f in frames if f["metrics"][name]["pixel_count"] == 0]
        keys = sorted({key for f in valid for key in f["metrics"][name] if key != "pixel_count"})
        result[name] = {"total_frame_count": len(frames), "valid_frame_count": len(valid),
                        "empty_frame_count": len(empty), "empty_frame_ids": empty,
                        "pixel_count_sum": sum(f["metrics"][name]["pixel_count"] for f in frames),
                        "equal_valid_frame_means": {key: math.fsum(f["metrics"][name][key] for f in valid) / len(valid)
                                                    for key in keys},
                        "aggregation": "equal means over nonempty regions only; all 49 frames retained"}
    return result


def export(source, destination, verified_contract_path, scene_label, architecture_path):
    source = source.resolve()
    config = read(source / "effective_config.json")
    spec = config["followup_scene_spec"]
    arm = config["hybrid_arm"]
    require(arm == "GN11" and config["paper_sgcm_enabled"] is True and config["hybrid_nps_enabled"] is True,
            "effective config does not identify enabled GN11 components")
    require(config["model_path"] == str(source), "effective config source mismatch")
    ids, times = spec["frame_ids_with_camera"], spec["model_timestamps"]
    width, height = spec["render_wh"]
    raw_start, raw_end = spec["raw_frame_range"]
    require(len(ids) == len(times) == 49 and raw_end - raw_start + 1 == 49 and times == sorted(times),
            "followup frame/timestamp specification mismatch")
    inherited = read(verified_contract_path)
    validate_contract(inherited, source, spec)
    event = read(source / "checkpoint_events/iteration_030000.json")
    require(event["status"] == "complete" and event["iteration"] == 30000 and
            event["parameters_evaluated"] == "after_optimizer_and_sgcm_same_as_checkpoint",
            "30k event parameter contract mismatch")
    require([f["frame_id"] for f in event["frames"]] == ids and
            [f["timestamp"] for f in event["frames"]] == times, "30k frame identity mismatch")
    for path in (destination, destination.with_suffix(".tar.gz")):
        require(not path.exists(), "refusing report clobber: " + str(path))
    destination.mkdir(parents=True)
    observed = destination / "assets/observed"
    control = destination / "assets/static_time"
    observed.mkdir(parents=True)
    control.mkdir()
    frames, replay_count, max_replay_error = [], 0, 0.0
    extrema = {}
    for index, row in enumerate(event["frames"]):
        metric_path = within(row["hybrid_metrics"], source)
        stored = read(metric_path)
        require(stored["frame_id"] == row["frame_id"], "metric frame identity mismatch")
        with np.load(within(stored["arrays"], source), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_arrays(arrays, width, height)
        recomputed = region_metrics(arrays)
        count, error = match_regions(recomputed, stored["regions"])
        replay_count += count
        max_replay_error = max(max_replay_error, error)
        images = {}
        for key, raw_key in (("gt", "gt"), ("composite", "composite"), ("static", "static"),
                             ("dynamic", "dynamic"), ("alpha_s", "alpha_s"),
                             ("alpha_d", "alpha_d"), ("body_mask", "actor_mask")):
            filename = "%03d_%s.png" % (index, key)
            png(observed / filename, arrays[raw_key], rgb=key in ("gt", "composite", "static", "dynamic"))
            images[key] = "assets/observed/" + filename
        for key, value in arrays.items():
            if value.dtype == np.bool_:
                continue
            value_min, value_max = float(value.min()), float(value.max())
            prior = extrema.setdefault(key, {"min": value_min, "max": value_max})
            prior["min"], prior["max"] = min(prior["min"], value_min), max(prior["max"], value_max)
        frames.append({"id": row["frame_id"], "raw_index": index, "local_index": index,
                       "source_raw_index": raw_start + index, "timestamp": row["timestamp"],
                       "images": images, "bbox": bbox(arrays["actor_mask"], width, height),
                       "body_pixel_count": recomputed["actor"]["pixel_count"],
                       "metrics": stored["regions"], "metric_source": str(metric_path),
                       "raw_FP32_source": stored["arrays"]})
        if (index + 1) % 10 == 0 or index == 48:
            print("observed PNG + finite + metric replay %d/49" % (index + 1), flush=True)
    time_root = source / "static_time_30000"
    time_summary = read(time_root / "summary.json")
    require(time_summary["anchor_pose_frame"] == spec["anchor_frame"] and
            time_summary["anchor_camera_directory"] == spec["camera_directory"] and
            [r["timestamp_model"] for r in time_summary["rows"]] == times and
            [r["source_timestamp_frame"] for r in time_summary["rows"]] == spec["frame_ids"],
            "fixed-pose saved specification mismatch")
    anchor_index = spec["frame_ids"].index(spec["anchor_frame"])
    require(anchor_index == spec["anchor_index"] and
            time_summary["anchor_timestamp_model"] == times[anchor_index] == spec["anchor_timestamp"],
            "fixed-pose anchor model time mismatch")
    with np.load(time_root / ("%03d.npz" % anchor_index), allow_pickle=False) as archive:
        anchor = {key: archive[key] for key in archive.files}
    validate_arrays(anchor, width, height, True)
    static_frames, max_time_replay_error = [], 0.0
    for index, row in enumerate(time_summary["rows"]):
        with np.load(time_root / ("%03d.npz" % index), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_arrays(arrays, width, height, True)
        diff_rgb = np.abs(arrays["static_rgb"].clip(0, 1) - anchor["static_rgb"].clip(0, 1))
        diff_alpha = np.abs(arrays["static_alpha"] - anchor["static_alpha"])
        replay = {"static_rgb_MAE_from_anchor": float(diff_rgb.mean()),
                  "static_rgb_max_abs_from_anchor": float(diff_rgb.max()),
                  "static_alpha_MAE_from_anchor": float(diff_alpha.mean())}
        for key, value in replay.items():
            require(math.isfinite(row[key]) and math.isclose(value, row[key], rel_tol=2e-6, abs_tol=1e-7),
                    "static time saved FP32 metric mismatch: " + key)
            max_time_replay_error = max(max_time_replay_error, abs(value - row[key]))
        images = {}
        for key, raw_key in (("static", "static_rgb"), ("alpha_s", "static_alpha")):
            filename = "%03d_%s.png" % (index, key)
            png(control / filename, arrays[raw_key], rgb=key == "static")
            images[key] = "assets/static_time/" + filename
        static_frames.append({"id": row["source_timestamp_frame"], "raw_index": index, "local_index": index,
                              "source_raw_index": raw_start + index, "timestamp": row["timestamp_model"],
                              "images": images, "bbox": frames[anchor_index]["bbox"], "metrics": row})
    regions = aggregate_regions(frames)
    memory, status = read(source / "memory_summary.json"), read(source / "status.json")
    require(memory["run_status"] == status["status"] == "complete" and status["iteration"] == 30000,
            "terminal resource/status mismatch")
    eval_metrics = read(within(event["metrics"], source))
    mean = lambda name, key: regions[name]["equal_valid_frame_means"].get(key)
    fp32_psnr = mean("full", "composite_psnr_db")
    require(math.isclose(fp32_psnr, eval_metrics["psnr"], rel_tol=0, abs_tol=1e-4),
            "FP32 full PSNR does not match saved overall evaluation")
    pt, nv, extra = memory["pytorch"], memory["nvidia_smi"], memory["extra"]
    summary = {"scene": spec["scene_id"], "scene_label": scene_label, "arm": arm,
               "training_steps": 30000, "retained_frame_count": 49, "raw_source_frame_range": [raw_start, raw_end],
               "anchor_pose_frame": spec["anchor_frame"], "anchor_timestamp_model": spec["anchor_timestamp"],
               "wall_seconds": status["wall_seconds"], "wall_minutes": status["wall_seconds"] / 60,
               "device_peak_GiB": nv["device_peak_used_bytes"] / 1024 ** 3,
               "process_peak_GiB": nv["process_peak_used_bytes"] / 1024 ** 3,
               "allocated_peak_GiB": pt["peak_allocated_bytes"] / 1024 ** 3,
               "reserved_peak_GiB": pt["peak_reserved_bytes"] / 1024 ** 3,
               "static_points": extra["render_static_points"], "dynamic_points": extra["render_dynamic_points"],
               "iter_forward_backward_ms": status["iter_forward_backward_ms"],
               "full_composite_psnr_db": fp32_psnr, "actor_composite_psnr_db": mean("actor", "composite_psnr_db"),
               "background_static_psnr_db": mean("non_sky_outside_actor", "static_psnr_db"),
               "outside_dynamic_alpha_mean": mean("non_sky_outside_actor", "alpha_d_mean"),
               "saved_overall_photo_metrics": eval_metrics,
               "valid_actor_frame_count": regions["actor"]["valid_frame_count"],
               "empty_actor_frame_count": regions["actor"]["empty_frame_count"],
               "region_aggregation": regions,
               "static_time_difference_means": {key: math.fsum(r[key] for r in time_summary["rows"]) / 49
                                                for key in replay},
               "static_time_difference_maxima": {key: max(r[key] for r in time_summary["rows"]) for key in replay},
               "full_resource_record": {"memory_summary": memory, "status": status},
               "restoration_verified": False, "human_verdict": None}
    verification = {"status": "PASS_SAVED_FP32_CPU_EXPORT", "completion_contract_reused": True,
                    "completion_contract_observed_singapore": inherited["observed_singapore"],
                    "original_completion_contract_actual_result": inherited["original_completion_contract_actual_result"],
                    "observed_all_finite_FP32_frames": 49, "static_time_all_finite_FP32_frames": 49,
                    "all_saved_arrays_finite_including_depth": True,
                    "regional_numeric_metrics_recomputed_and_matched": replay_count,
                    "regional_max_abs_metric_replay_error": max_replay_error,
                    "static_time_metrics_recomputed_and_matched": 49 * 3,
                    "static_time_metric_protocol": "RGB differences from FP32 clip[0,1]; raw unmasked alpha differences",
                    "static_time_max_abs_metric_replay_error": max_time_replay_error,
                    "full_FP32_PSNR_abs_difference_vs_saved_photo_eval": abs(fp32_psnr - eval_metrics["psnr"]),
                    "raw_observed_array_extrema": extrema,
                    "empty_regions_allowed_only_with_pixel_count_zero": True,
                    "PNG_images": 441, "GPU_used": False, "Torch_imported": False,
                    "checkpoint_loaded": False, "rerender_started": False, "training_started": False,
                    "restoration_verified": False, "human_verdict": None}
    manifest = {"schema_version": 1, "title": "DrivingEditor · %s %s · 30k" % (arm, scene_label),
                "arm": arm, "components": {"SGCM": config["paper_sgcm_enabled"], "NPS": config["hybrid_nps_enabled"]},
                "checkpoint": 30000, "scene": spec["scene_id"], "scene_label": scene_label,
                "frame_count": 49, "width": width, "height": height,
                "generated_singapore": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                "summary": summary, "frames": frames, "static_time_frames": static_frames,
                "static_time_summary": time_summary, "effective_config": config,
                "completion_contract": inherited, "verification": verification,
                "source": {"model_path": str(source), "event": str(source / "checkpoint_events/iteration_030000.json"),
                           "effective_config": str(source / "effective_config.json"),
                           "followup_scene_spec": spec,
                           "RGB_display": "stored FP32 clipped [0,1], rounded 8bit; no per-image normalization or body premask",
                           "alpha_display": "raw unmasked alpha clipped [0,1], black0 white1; no per-image normalization",
                           "metrics": "FP32-derived regional metrics; equal nonempty frame means with explicit valid counts",
                           "static_time": "fixed pose " + spec["anchor_frame"] + "; all 49 model timestamps; counterfactual without GT",
                           "scope": "all-observed development reconstruction, body-only mask without shadows",
                           "GPU_used_by_export": False, "checkpoint_loaded": False,
                           "restoration_verified": False, "human_verdict": None},
                "architecture": "architecture-components.svg", "restoration_verified": False, "human_verdict": None}
    require("torch" not in sys.modules, "CPU exporter unexpectedly imported torch")
    dump(destination / "manifest.json", manifest)
    dump(destination / "summary.json", summary)
    dump(destination / "export_verification.json", verification)
    dump(destination / "completion_contract.json", inherited)
    with (destination / "architecture-components.svg").open("x", encoding="utf-8") as stream:
        stream.write(architecture_path.read_text(encoding="utf-8-sig"))
    png_count = len(list((destination / "assets").rglob("*.png")))
    require(png_count == 441, "export did not retain exactly 441 PNGs")
    print(json.dumps({"status": verification["status"], "destination": str(destination), "PNG_images": png_count,
                      "full_composite_psnr_db": fp32_psnr, "valid_actor_frames": summary["valid_actor_frame_count"],
                      "empty_actor_frames": summary["empty_actor_frame_count"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-contract", type=Path, required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output, args.verified_contract, args.scene_label, args.architecture)


if __name__ == "__main__":
    main()
