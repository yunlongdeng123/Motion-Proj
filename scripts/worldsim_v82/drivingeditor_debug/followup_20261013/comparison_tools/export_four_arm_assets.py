"""四臂完整终态后，仅从保存FP32数组与CPU日志生成比较清单和图片。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from contextlib import ExitStack
import datetime
import itertools
import json
import math
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from _legacy_c00_export import rgb, gray, bbox
from _legacy_region_metrics import region_metrics

ARMS = ("C00", "G10", "N01", "GN11")
NODES = (10000, 20000, 30000)
MODULES = {"C00": (False, False), "G10": (True, False), "N01": (False, True), "GN11": (True, True)}
EXPECTED_IDS = ["image_0/0017%03d" % index for index in range(49)]
DEFAULT_RUNS = Path("/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012/runs")
RGB_KEYS = ("gt", "composite", "static", "dynamic")
ARRAY_KEYS = RGB_KEYS + ("alpha_s", "alpha_d", "depth_accum_model", "actor_mask", "sky_mask")
TIME_KEYS = ("static_rgb_MAE_from_anchor", "static_rgb_max_abs_from_anchor", "static_alpha_MAE_from_anchor")
RAW_TIME_KEYS = ("raw_static_velocity_norm_model_mean", "raw_static_velocity_norm_model_max", "temporal_width_model_mean", "static_SHM_peak_to_peak_displacement_bound_model_max")
CONFIG_DIFFERENCES = {"hybrid_arm", "model_path", "paper_sgcm_enabled", "hybrid_nps_enabled"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), "Nonfinite/missing " + label)
    return float(value)


def require_terminal(status, arm):
    require(status.get("status") == "complete" and status.get("arm") == arm and status.get("iteration") == 30000 and status.get("iterations") == 30000,
            arm + " is not a complete30k terminal arm; final comparison is blocked")
    require(finite(status.get("wall_seconds"), arm + " wall_seconds") > 0, arm + " wall time missing")


def headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            require(member.endswith(".npy"), "Unexpected NPZ member")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported saved NPY header version")
            require(not dtype.hasobject, "Object array forbidden")
            result[member[:-4]] = (shape, dtype)
    return result


def check_npz_header(path, static_time=False):
    values = headers(path)
    expected = ("static_rgb", "static_alpha") if static_time else ARRAY_KEYS
    require(all(key in values for key in expected), "Saved array fields missing: " + str(path))
    for key in expected:
        shape, dtype = values[key]
        is_rgb = key in RGB_KEYS or key == "static_rgb"
        require(shape == (3, 320, 480) if is_rgb else shape in ((1, 320, 480), (320, 480)), "Wrong saved shape: " + str(path) + "/" + key)
        require(dtype == (np.dtype(bool) if key.endswith("_mask") else np.dtype(np.float32)), "Wrong saved dtype: " + str(path) + "/" + key)


def canonical_rows(rows, arm):
    require(isinstance(rows, list) and len(rows) == 49, arm + " must have49 frame rows")
    require(len({row["frame_id"] for row in rows}) == 49, arm + " duplicate frame ID")
    ordered = sorted(rows, key=lambda row: finite(row.get("timestamp"), arm + " timestamp"))
    require([row["frame_id"] for row in ordered] == EXPECTED_IDS, arm + " wrong/missing/reordered FRONT frame identities")
    for index, row in enumerate(ordered):
        require(row.get("camera") == "image_0" and row.get("width") == 480 and row.get("height") == 320, arm + " camera/dimensions mismatch")
        require(math.isclose(row["timestamp"], -.48 + .02 * index, abs_tol=1e-12), arm + " model timestamp mismatch")
    return ordered


def time_summary(source, step, ordered):
    summary = read(source / ("static_time_%d/summary.json" % step))
    require(summary.get("iteration") == step and summary.get("anchor_pose_frame") == "0017010" and math.isclose(summary.get("anchor_timestamp_model", float("nan")), -.28, abs_tol=1e-12), "Actual fixed-pose anchor mismatch")
    rows = summary.get("rows", [])
    require(len(rows) == 49, "Fixed pose must retain49 model times")
    for index, row in enumerate(rows):
        require(row["source_timestamp_frame"] == ordered[index]["frame_id"].split("/")[-1] and math.isclose(row["timestamp_model"], ordered[index]["timestamp"], abs_tol=1e-12), "Fixed-pose rows do not match actual observed model timestamps")
        for key in TIME_KEYS:
            finite(row.get(key), "static time " + key)
        check_npz_header(source / ("static_time_%d/%03d.npz" % (step, index)), static_time=True)
    for key in RAW_TIME_KEYS:
        finite(summary.get(key), key)
    return summary


def node_contract(source, arm, step, config):
    event_path = source / ("checkpoint_events/iteration_%06d.json" % step)
    event = read(event_path)
    require(event.get("status") == "complete" and event.get("iteration") == step, arm + " incomplete checkpoint node")
    require(event["config"] == config and event["source_path"] == config["source_path"], arm + " checkpoint config/source drift")
    require(event.get("parameters_evaluated") == "after_optimizer_and_sgcm_same_as_checkpoint", arm + " wrong checkpoint/evaluation phase")
    resume = event["resume_state"]
    require(all(resume.get(key) is True for key in ("optimizer", "camera_sampler", "scale")) and all(key in resume.get("rng", []) for key in ("python", "numpy", "torch", "cuda_if_initialized")), arm + " incomplete state declaration")
    require(resume.get("sgcm_geometry") is MODULES[arm][0] and (event.get("sgcm") is not None) == MODULES[arm][0], arm + " SGCM continuation geometry declaration inconsistent")
    names = {"static": "chkpnt%d.pth", "dynamic": "dynamic_chkpnt%d.pth", "environment": "env_light_chkpnt%d.pth", "training_state": "training_state%d.pth"}
    checkpoint_stats = {}
    for key, name in names.items():
        entry = event["checkpoints"][key]
        path = source / (name % step)
        require(Path(entry["path"]).name == path.name and path.is_file() and path.stat().st_size == entry["bytes"] and entry["bytes"] > 0, arm + " incomplete/mismatched checkpoint bundle")
        checkpoint_stats[key] = {"path": str(path), "bytes": entry["bytes"], "content_loaded": False}
    ordered = canonical_rows(event["frames"], arm)
    canonical_rows(event["camera_manifest"], arm)
    rawroot = source / ("eval/train_%d_render/hybrid_raw" % step)
    for row in ordered:
        name = row["frame_id"].replace("/", "_")
        require(Path(row["hybrid_metrics"]).name == name + ".json", arm + " wrong metric pointer")
        metric = read(rawroot / (name + ".json"))
        require(metric["frame_id"] == row["frame_id"], arm + " wrong frame metric identity")
        check_npz_header(rawroot / (name + ".npz"))
        for channel in RGB_KEYS:
            path = source / ("eval/train_%d_render/%s/%s.png" % (step, channel, name))
            require(Path(row["channels"][channel]).name == path.name, arm + " channel pointer mismatch")
            with Image.open(path) as image:
                require(image.size == (480, 320), arm + " missing/wrong node PNG dimensions")
    times = time_summary(source, step, ordered)
    return {"status": "PASS_READONLY_STAT_AND_ARRAY_HEADERS", "step": step, "event": str(event_path), "checkpoints": checkpoint_stats,
            "frame_count": 49, "static_time_count": 49, "restoration_verified": False}, event, ordered, times


def paired_steps(sources, nodes):
    selected = {arm: {} for arm in ARMS}
    with ExitStack() as stack:
        streams = [stack.enter_context((sources[arm] / "paired_steps.jsonl").open(encoding="utf-8")) for arm in ARMS]
        count = 0
        for iteration, lines in enumerate(itertools.zip_longest(*streams), 1):
            require(all(line is not None for line in lines), "Paired steps have different lengths")
            rows = [json.loads(line) for line in lines]
            expected = {key: rows[0][key] for key in ("iteration", "frame_id", "time_shift", "learning_rates")}
            require(expected["iteration"] == iteration and expected["frame_id"] in EXPECTED_IDS, "Paired step iteration/frame invalid")
            for arm, row in zip(ARMS, rows):
                require({key: row[key] for key in expected} == expected, arm + " frame/time_shift/optimizer LR pairing mismatch at%d" % iteration)
                if iteration in nodes:
                    selected[arm][str(iteration)] = row
            count = iteration
    require(count == 30000, "Final comparison requires exactly30000 paired steps")
    return {"status": "PASS_ALL30000_FRAME_TIME_SHIFT_LR_ROWS", "rows": count, "sources": {arm: str(sources[arm] / "paired_steps.jsonl") for arm in ARMS}}, selected


def diagnostics(source, arm, event, selected, gate, campaign):
    gradient_path = source / "gradient_diagnostics.jsonl"
    gradient = [json.loads(line) for line in gradient_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes = gate["module_diagnostics"]["nps"]["nodes"]
    require(all(any(row.get("iteration") == node for row in gradient) for node in nodes), arm + " incomplete gradient diagnostic nodes")
    sgcm_rows = []
    log = campaign / (arm + "_training.log")
    if MODULES[arm][0]:
        require(event.get("sgcm") is not None and event["sgcm"]["addition_guard"]["enabled"] is True and event["sgcm"]["addition_guard"]["training_view_count"] == 49, arm + " SGCM actualview guard declaration missing")
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "SGCM_STATS " in line:
                    row, _ = json.JSONDecoder().raw_decode(line.split("SGCM_STATS ", 1)[1].lstrip())
                    sgcm_rows.append(row)
        expected = gate["module_diagnostics"]["sgcm"]["expected_due_steps"]
        require([row["iteration"] for row in sgcm_rows] == list(range(expected["first"], expected["last"] + 1, expected["step"])), arm + " SGCM schedule/log completeness failed")
        for row in sgcm_rows:
            require(row["points_after"] == row["points_before"] - row["pruned"] + row["cloned"] + row["split_parents"], arm + " SGCM topology accounting mismatch")
    return {"gradient_source": str(gradient_path), "gradient_records": gradient, "selected_paired_steps": selected,
            "nps_enabled": MODULES[arm][1], "nps_expected_keys": gate["module_diagnostics"]["nps"]["keys"],
            "sgcm_enabled": MODULES[arm][0], "sgcm_metadata": event.get("sgcm"), "sgcm_log_source": str(log), "sgcm_records": sgcm_rows,
            "scope": "Recorded diagnostics; ratio_valid=false remains invalid. Header/metadata checks do not certify full restore or exact contributor visibility."}


def metric_values(metrics, paths):
    values = {}
    for name, path in paths.items():
        value = {"regions": metrics}
        for key in path.split("."):
            value = value[key]
        values[name] = finite(value, name)
    return values


def comparison(arm_data, metric_paths):
    per_frame, summaries = [], {}
    for index in range(49):
        values = {arm: metric_values(arm_data[arm]["frames"][index]["metrics"], metric_paths) for arm in ARMS}
        per_frame.append({"id": EXPECTED_IDS[index], "timestamp": arm_data["C00"]["frames"][index]["timestamp"], "values": values,
                          "delta_vs_C00": {arm: {key: values[arm][key] - values["C00"][key] for key in metric_paths} for arm in ARMS}})
    for arm in ARMS:
        scores, delta_stats = {}, {}
        for key in metric_paths:
            scores[key] = float(np.mean([row["values"][arm][key] for row in per_frame]))
            delta = np.array([row["delta_vs_C00"][arm][key] for row in per_frame])
            delta_stats[key] = {"mean": float(delta.mean()), "median": float(np.median(delta)), "p10": float(np.percentile(delta, 10)),
                                "min": float(delta.min()), "max": float(delta.max()), "negative_count": int((delta < 0).sum()),
                                "below_minus1_count": int((delta < -1).sum()) if key.endswith("psnr_db") else None,
                                "below_minus1_ids": [row["id"] for row, value in zip(per_frame, delta) if value < -1] if key.endswith("psnr_db") else None,
                                "worst_delta_frame": per_frame[int(delta.argmin())]["id"], "selection": "post_endpoint_adverse_selection; all49 retained"}
        summaries[arm] = {"equal_frame_means": scores, "paired_delta_statistics_vs_C00": delta_stats}
    return {"aggregation": "49-frame equal mean of per-frame scores; signed delta=arm-C00; not pooled PSNR", "per_frame": per_frame, "arms": summaries}


def validate(sources, gate, campaign):
    require(gate["primary_gate"]["reference_scene"] == "scene0017_000_048", "This final comparison tool is for the fixed silver scene")
    # 心跳先快速核对四臂终态，未齐全时不扫描已有大日志或导出素材。
    for arm in ARMS:
        require_terminal(read(sources[arm] / "status.json"), arm)
    configs, records = {}, {}
    for arm in ARMS:
        source = sources[arm]
        status, config, memory = read(source / "status.json"), read(source / "effective_config.json"), read(source / "memory_summary.json")
        require_terminal(status, arm)
        require(config["hybrid_arm"] == arm and (config["paper_sgcm_enabled"], config["hybrid_nps_enabled"]) == MODULES[arm], "Arm/module flag mismatch")
        require(config["iterations"] == config["position_lr_max_steps"] == 30000 and config["seed"] == 0 and config["start_checkpoint"] is None and config["resolution_scales"] == [4], "Wrong common30k policy")
        require(memory["run_status"] == "complete" and memory["iteration"] == 30000 and memory["expected_iterations"] == 30000 and memory["error_count"] == 0 and memory["sampling"]["samples_with_errors"] == 0, arm + " incomplete/error memory evidence")
        require(memory["sampling"]["valid_device_samples"] > 0 and memory["sampling"]["valid_pid_present_samples"] > 0 and bool(memory["phases"]), arm + " missing memory/time phase coverage")
        configs[arm] = config
        contracts = []
        for node in NODES:
            contract, event, rows, times = node_contract(source, arm, node, config)
            contracts.append(contract)
        records[arm] = {"source": source, "status": status, "config": config, "memory": memory, "node_contracts": contracts,
                        "event": event, "rows": rows, "static_time_summary": times}
    baseline = {key: value for key, value in configs["C00"].items() if key not in CONFIG_DIFFERENCES}
    require(set(gate["pairing_contract"]["config_allowlist_differences"]) == CONFIG_DIFFERENCES, "Gate/config allowlist drift")
    for arm in ARMS:
        require({key: value for key, value in configs[arm].items() if key not in CONFIG_DIFFERENCES} == baseline, arm + " common resolved config differs")
        require([(row["frame_id"], row["timestamp"]) for row in records[arm]["rows"]] == [(row["frame_id"], row["timestamp"]) for row in records["C00"]["rows"]], "Fourarm49 frame/timestamp pairing failed")
    pairing, selected = paired_steps(sources, gate["module_diagnostics"]["nps"]["nodes"])
    for arm in ARMS:
        records[arm]["diagnostics"] = diagnostics(sources[arm], arm, records[arm]["event"], selected[arm], gate, campaign)
    return records, pairing


def export(sources, gate_path, output, campaign, check_only=False):
    gate = read(gate_path)
    records, pairing = validate(sources, gate, campaign)
    if check_only:
        print(json.dumps({"status": "READY_FOUR_ARM_TERMINAL_CPU_EXPORT", "checkpoint_loaded": False, "GPU_used": False, "output_written": False}))
        return
    require(not output.exists(), "Refuse to overwrite existing comparison directory")
    output.mkdir(parents=True)
    arms = {}
    for arm, record in records.items():
        for directory in ("observed", "static_time"):
            (output / "assets" / arm / directory).mkdir(parents=True)
        arms[arm] = {"source": str(record["source"]), "status": record["status"], "memory": record["memory"], "node_contracts": record["node_contracts"],
                     "diagnostics": record["diagnostics"], "frames": [], "static_time_frames": [], "static_time_summary": record["static_time_summary"]}
    # 每次只保留同一帧四臂数组；跨臂GT、body/sky和ROI严格相同。
    for index in range(49):
        reference = None
        for arm in ARMS:
            record, data = records[arm], arms[arm]
            row = record["rows"][index]
            name = row["frame_id"].replace("/", "_")
            rawroot = record["source"] / "eval/train_30000_render/hybrid_raw"
            metrics = read(rawroot / (name + ".json"))["regions"]
            with np.load(rawroot / (name + ".npz"), allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ARRAY_KEYS}
            require(all(np.isfinite(value).all() for value in arrays.values()), arm + " saved FP32 nonfinite")
            if reference is None:
                reference = {key: arrays[key] for key in ("gt", "actor_mask", "sky_mask")}
                rectangle = bbox(arrays["actor_mask"])
            else:
                require(all(np.array_equal(arrays[key], reference[key]) for key in reference), arm + " actual GT/body/sky array pairing failed")
            recomputed = region_metrics(arrays, arrays["actor_mask"], arrays["sky_mask"])
            values, fresh = metric_values(metrics, gate["metrics"]["key_paths"]), metric_values(recomputed, gate["metrics"]["key_paths"])
            require(all(math.isclose(values[key], fresh[key], rel_tol=1e-7, abs_tol=1e-9) for key in values), arm + " stored metric does not match saved FP32/clamp semantics")
            images = {}
            for key in RGB_KEYS + ("alpha_s", "alpha_d", "body_mask"):
                rawkey = "actor_mask" if key == "body_mask" else key
                relative = "assets/%s/observed/%03d_%s.png" % (arm, index, key)
                Image.fromarray(rgb(arrays[rawkey]) if key in RGB_KEYS else gray(arrays[rawkey])).save(output / relative, compress_level=4)
                images[key] = relative
            data["frames"].append({"id": row["frame_id"], "raw_index": index, "timestamp": row["timestamp"], "images": images, "bbox": rectangle, "metrics": metrics})
    for arm in ARMS:
        record, data = records[arm], arms[arm]
        for index, row in enumerate(record["static_time_summary"]["rows"]):
            images = {}
            with np.load(record["source"] / ("static_time_30000/%03d.npz" % index), allow_pickle=False) as archive:
                for key, rawkey in (("static", "static_rgb"), ("alpha_s", "static_alpha")):
                    relative = "assets/%s/static_time/%03d_%s.png" % (arm, index, key)
                    Image.fromarray(rgb(archive[rawkey]) if key == "static" else gray(archive[rawkey])).save(output / relative, compress_level=4)
                    images[key] = relative
            data["static_time_frames"].append({"id": row["source_timestamp_frame"], "timestamp": row["timestamp_model"], "images": images, "bbox": data["frames"][10]["bbox"], "metrics": row})
    report = {"schema_version": 1, "mode": "real_four_arm_terminal", "status": "PASS_READONLY_FOUR_ARM30K_CONTRACT", "title": "DrivingEditor · 四臂30k比较",
              "scene": "scene0017_000_048", "checkpoint": 30000, "frame_count": 49, "width": 480, "height": 320,
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "arm_order": list(ARMS), "arms": arms,
              "metric_paths": gate["metrics"]["key_paths"], "comparison": comparison(arms, gate["metrics"]["key_paths"]), "pairing": pairing,
              "source_gate_plan": str(gate_path), "GPU_used_by_export": False, "checkpoint_loaded": False,
              "restoration_verified": False, "gate_evaluation_performed": False, "human_verdict": None,
              "display": {"RGB": "SavedFP32 clipped[0,1] rounded8bit, no bodypremask/no per-image normalization", "alpha": "Raw unmasked alpha displayed fixed[0,1]gray", "ROI": "Identical canonicalC00 bodycontext rectangle for all arms and branches", "static_time": "Actualsummary rows fixedpose0017010 at49 modeltimestamps, noGT qualityscore"},
              "scope": "All-observed development reconstruction, body-only masks withoutshadow; comparison does not grant GN11 gate or independenttest/edit quality acceptance"}
    write(output / "manifest.json", report)
    from build_four_arm_viewer import build
    build(output / "manifest.json", output)
    print(json.dumps({"status": "EXPORTED_REAL_FOUR_ARM_TERMINAL_CPU_ONLY", "output": str(output), "images": 4 * 49 * 9, "frames_per_arm": 49, "static_times_per_arm": 49, "checkpoint_loaded": False, "gate_evaluation_performed": False}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--campaign", type=Path, help="SGCM训练日志所在目录；缺省runs-root.parent")
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    sources = {arm: args.runs_root / ("scene0017_000_048_hybrid_30000_seed0_" + arm) for arm in ARMS}
    try:
        export(sources, args.gate_plan, args.output, args.campaign or args.runs_root.parent, args.check_only)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "BLOCKED_NO_FINAL_REPORT", "reason": str(error), "checkpoint_loaded": False, "GPU_used": False}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
