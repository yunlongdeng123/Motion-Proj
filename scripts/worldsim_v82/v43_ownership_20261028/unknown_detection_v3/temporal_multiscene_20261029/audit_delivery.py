"""CPU审计完整浮点交付，并可另建轻量UI包；不修改原full evidence。

full evidence -> audit_sequence / 分量闭合 -> JSON + 完整flip IDs
PNG/receipt/summary + 轻量行ID/布尔范围 -> light_delivery（不是full证据）
"""
import os
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from scope import eligible_from_screen
from temporal_audit import audit_sequence

AUDIT_KEYS = ["starting_removed_ids", "existing_removed_ids", "final_removed_ids", "new_unknown_ids", "eligible_ids",
              "residual_ids", "residual_mass", "allowed", "query", "display_scope", "body", "contact", "alpha",
              "gaussian_rgb", "white_rgb", "baseline_native_rgb", "screen_centers", "row_namespace", "checkpoint_ref", "context", "raw"]
UI_KEYS = ["final_removed_ids", "eligible_ids", "new_unknown_ids", "screen_centers",
           "allowed", "query", "display_scope", "body", "contact"]
MASK_KEYS = ["allowed", "query", "display_scope", "body", "contact"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def scalar(value):
    result = np.asarray(value).item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def relative(path, root):
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def under(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def discover(root, output, light, explicit_summaries=None, contexts=None):
    paths = list(explicit_summaries or root.rglob("summary.json"))
    scenes, seen = [], set()
    for path in sorted(Path(p) for p in paths):
        path = path.resolve()
        if under(path, output) or (light is not None and under(path, light)):
            continue
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "context" not in data or not isinstance(data.get("records"), list):
            continue
        context = str(data["context"])
        if contexts and context not in contexts:
            continue
        if context in seen:
            raise ValueError("同context存在多份summary；请用--summary指定本轮唯一输入: " + context)
        relative(path, root)  # 不复制/审计ROOT外的隐式输入。
        seen.add(context)
        scenes.append({"context": context, "summary_path": path, "directory": path.parent, "summary": data})
    if not scenes:
        raise ValueError("ROOT下未找到context + records的源summary.json")
    return scenes


def collect_entries(scene, root):
    rows = scene["summary"]["records"]
    arms = sorted(set(str(r.get("method_arm", r.get("arm"))) for r in rows if r.get("method_arm", r.get("arm"))))
    entries = []
    for raw_row in rows:
        status = raw_row.get("status", "ok")
        row_arms = [raw_row.get("method_arm", raw_row.get("arm"))]
        if row_arms == [None] and status != "ok":
            row_arms = arms or ["unspecified_arm"]
        for arm in row_arms:
            row = dict(raw_row)
            row.setdefault("context", scene["context"])
            row.setdefault("method_arm", arm)
            row.setdefault("query_mode", arm)
            row["status"] = "ok" if status == "complete" else status
            row.setdefault("skip_reason", row.get("reason"))
            frame_dir = scene["directory"] / str(arm) / ("%03d" % int(row["raw"]))
            receipt_path = frame_dir / "receipt.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                row.update(receipt)
                row["status"] = "ok" if row.get("status") == "complete" else row.get("status", "ok")
                row.setdefault("method_arm", row.get("arm", arm))
                row.setdefault("query_mode", row["method_arm"])
                row.setdefault("skip_reason", row.get("reason"))
            evidence = Path(row.get("evidence", frame_dir / "evidence.npz"))
            if not evidence.is_absolute():
                evidence = scene["directory"] / evidence
            relative(evidence, root)
            if row.get("status", "ok") == "ok" and not evidence.exists():
                row.update(status="skipped", skip_reason="full_float_evidence_missing; no UI or neighbor fallback")
            entries.append({"record": row, "evidence": evidence, "receipt": receipt_path, "directory": frame_dir})
    keys = [(e["record"].get("method_arm"), e["record"]["raw"]) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("同场景arm/raw重复，不能混合多次运行")
    return entries


def contact_metrics(record, arrays):
    contact, alpha, white = arrays.get("contact"), arrays.get("alpha"), arrays.get("white_rgb")
    if contact is None or alpha is None or white is None:
        return {"available": False, "reason": "contact/after alpha/white float missing"}
    contact = np.asarray(contact, bool)
    if not contact.any():
        return {"available": False, "reason": "empty_contact_query"}
    lum = (np.clip(white, 0, 1) * np.array([.2126, .7152, .0722])[:, None, None]).sum(0)
    result = {"available": True, "pixels": int(contact.sum()), "alpha_mean": float(alpha[contact].mean()),
              "alpha_max": float(alpha[contact].max()), "dark_pixels_luma_lt_0p2": int((contact & (lum < .2)).sum()),
              "luminance_mean": float(lum[contact].mean()), "not_physical_shadow_ground_truth": True}
    if "contact_alpha_mean" in record:
        result["receipt_alpha_mean_abs_difference"] = abs(result["alpha_mean"] - record["contact_alpha_mean"])
    if "contact_dark_pixels" in record:
        result["receipt_dark_pixel_difference"] = result["dark_pixels_luma_lt_0p2"] - record["contact_dark_pixels"]
    return result


def load_arrays(entry):
    record = entry["record"]
    if record.get("status", "ok") != "ok":
        return {}, []
    with np.load(entry["evidence"], allow_pickle=False) as f:
        arrays = {k: f[k] for k in AUDIT_KEYS if k in f.files}
    issues = []
    for key in ["context", "raw", "row_namespace", "checkpoint_ref"]:
        if key in arrays and key in record and str(scalar(arrays[key])) != str(record[key]):
            issues.append("evidence_receipt_metadata_mismatch:" + key)
    if "screen_centers" in arrays and "new_unknown_ids" in arrays and "allowed" in arrays:
        screen = arrays["screen_centers"]
        if screen.shape != (len(arrays["new_unknown_ids"]), 2):
            issues.append("screen_centers_not_aligned_with_new_ids")
        elif not eligible_from_screen(screen, np.ones(len(screen)), arrays["allowed"]).all():
            issues.append("new_center_xy_fails_rint_allowed_mask")
    else:
        issues.append("projected_center_membership_payload_missing")
    return arrays, issues


def ui_evidence(entry, arrays, light, root):
    missing = [k for k in UI_KEYS if k not in arrays]
    if missing:
        return None, missing
    destination = light / relative(entry["directory"], root) / "ui_evidence.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = entry["record"]
    values = {k: arrays[k].astype(bool, copy=False) if k in MASK_KEYS else arrays[k] for k in UI_KEYS}
    values.update(schema=np.array("bounded-ui-row-evidence-v1"), full_evidence=np.array(False),
                  context=np.array(record["context"]), raw=np.array(record["raw"]),
                  row_namespace=np.array(record.get("row_namespace", "")),
                  checkpoint_ref=np.array(record.get("checkpoint_ref", "")),
                  static_rows=np.array(record.get("static_rows", -1)),
                  method_arm=np.array(record.get("method_arm", "")))
    np.savez_compressed(destination, **values)
    assert not any(k in values for k in ["alpha", "gaussian_rgb", "environment_rgb", "white_rgb", "baseline_native_rgb"])
    return relative(destination, light), []


def parameter_audit(records):
    arms = {}
    for row in records:
        if row.get("status", "ok") == "ok":
            arms.setdefault(row.get("method_arm", ""), []).append(row)
    result = []
    for arm, values in arms.items():
        templates, parameters = [v.get("template") for v in values], [v.get("method_parameters") for v in values]
        result.append({"method_arm": arm, "scene_contexts": sorted(set(v["context"] for v in values)),
                       "template_same_across_scenes": all(t == templates[0] for t in templates) if all(t is not None for t in templates) else None,
                       "method_parameters_same_across_scenes": all(p == parameters[0] for p in parameters) if all(p is not None for p in parameters) else None,
                       "query_modes": sorted(set(v.get("query_mode", "") for v in values)),
                       "init_modes": sorted(set(v.get("init_mode", "") for v in values)),
                       "outside_baselines": sorted(set(v.get("outside_baseline", "") for v in values))})
    return result


def expected_raws(records, count, sparse):
    if sparse:
        return sorted(set(int(r["raw"]) for r in records))
    offsets = set(int(r["raw"]) - int(r["camera_index"]) for r in records if r.get("camera_index") is not None)
    if len(offsets) == 1:
        return list(range(next(iter(offsets)), next(iter(offsets)) + count))
    start = min(int(r["raw"]) for r in records)
    return list(range(start, start + count))


def lightweight_summary(groups, expected_frames, allow_sparse):
    scenes = []
    for group in groups:
        frames = group["frames"]
        ok = [f for f in frames if not f.get("skipped")]
        worst = sorted([p for p in group["pairs"] if p.get("comparison_available") and p["consecutive_raw"]],
                       key=lambda p: -(p["on_within_shared_eligible"] + p["off_within_shared_eligible"]))[:5]
        contacts = [f["contact_metrics"] for f in ok if f.get("contact_metrics", {}).get("available")]
        def maximum(key):
            values = [f[key] for f in ok if f.get(key) is not None]
            return max(values) if values else None
        scenes.append({"context": group["context"], "method_arm": group["method_arm"],
            "frames_attempted": group["frames_attempted"], "successful_frames": len(ok), "expected_frames": expected_frames,
            "verified_frames": sum(bool(f.get("verified")) for f in ok), "skipped_frames": group["skipped_frames"],
            "missing_expected_raws": group["missing_expected_raws"],
            "coverage_complete": len(ok) == expected_frames and not group["skipped_frames"] and not group["missing_expected_raws"],
            "temporal_coverage_kind": "sparse anchors allowed; not fullclip stability" if allow_sparse else "consecutive source frames required",
            "reset_verified_frames": sum(f.get("reset_verified") is True for f in ok),
            "producer_converged_frames": sum(f.get("producer_converged") is True for f in ok),
            "nonconverged_raws": [f["raw"] for f in ok if f.get("producer_converged") is False],
            "new_rows_outside_eligible_sum": sum(f.get("new_rows_outside_eligible", 0) for f in ok),
            "residual_eligible_rows_sum": sum(f.get("residual_eligible_rows", 0) for f in ok),
            "residual_eligible_mass_sum": sum(f.get("residual_eligible_mass", 0) for f in ok),
            "outside_display_max_difference": maximum("outside_display_max_delta"),
            "white_equation_max_error": maximum("inside_white_equation_max_error"),
            "after_query_alpha_mass_closure_max_abs": maximum("query_alpha_mass_closure_abs"),
            "failed_frames": [{"raw": f["raw"], "issues": f["issues"]} for f in ok if not f.get("verified")],
            "worst_consecutive_transitions_by_shared_eligible_id_flips": worst,
            "contact": {"frames": len(contacts), "dark_pixels_sum": sum(c["dark_pixels_luma_lt_0p2"] for c in contacts),
                        "dark_pixels_max": max((c["dark_pixels_luma_lt_0p2"] for c in contacts), default=None),
                        "alpha_mean_over_frames": float(np.mean([c["alpha_mean"] for c in contacts])) if contacts else None,
                        "alpha_mean_max": max((c["alpha_mean"] for c in contacts), default=None)},
            "not_semantic_recall_or_RGB_flicker_ground_truth": True})
    return scenes


def run(args):
    started = time.monotonic()
    root = args.root.resolve()
    output = (args.output or root / "audit_delivery").resolve()
    light = (args.light_dir or root / "light_delivery").resolve() if (args.light or args.light_dir) else None
    if output.exists() or (light is not None and light.exists()):
        raise FileExistsError("保留旧交付，请指定新output/light目录")
    scenes = discover(root, output, light, args.summary, set(args.contexts or []))
    # 在创建子目录之前收集源资产，避免单scene测试root下递归复制自身。
    source_assets = sorted(set(p for s in scenes for p in s["directory"].rglob("*")
                               if p.is_file() and (p.suffix.lower() == ".png" or p.name in ["receipt.json", "summary.json"])
                               and not under(p, output) and (light is None or not under(p, light))))
    output.mkdir(parents=True, exist_ok=False)
    if light is not None:
        light.mkdir(parents=True, exist_ok=False)
    groups, flips, metadata_records, manifest_frames = [], {}, [], []
    full_evidence_bytes = 0
    for scene_index, scene in enumerate(scenes):
        entries = collect_entries(scene, root)
        records, additional = [], {}
        for entry in entries:
            row = dict(entry["record"])
            arrays, issues = load_arrays(entry)
            if arrays:
                full_evidence_bytes += entry["evidence"].stat().st_size
            row["arrays"] = arrays
            records.append(row)
            metadata_records.append({k: v for k, v in row.items() if k != "arrays"})
            key = (row.get("method_arm"), row["raw"])
            additional[key] = {"issues": issues, "contact": contact_metrics(row, arrays) if arrays else {"available": False},
                               "converged": row.get("converged")}
            ui_path, missing = ui_evidence(entry, arrays, light, root) if light is not None and arrays else (None, [])
            images = {p.stem: relative(p, root) for p in sorted(entry["directory"].glob("*.png"))}
            manifest_frames.append({"context": scene["context"], "raw": row["raw"], "frame_id": row.get("frame_id"),
                "method_arm": row.get("method_arm"), "status": row.get("status", "ok"), "skip_reason": row.get("skip_reason"),
                "receipt": relative(entry["receipt"], root) if entry["receipt"].exists() else None,
                "images": images, "ui_evidence": ui_path, "ui_missing_fields": missing,
                "full_evidence_source": relative(entry["evidence"], root), "full_float_evidence_in_light": False,
                "row_namespace": row.get("row_namespace"), "checkpoint_ref": row.get("checkpoint_ref"),
                "static_rows": row.get("static_rows"), "body_bbox_xyxy": row.get("body_bbox_xyxy"),
                "template": row.get("template"), "float_check_source": "temporal_audit.json"})
        expected = {scene["context"]: expected_raws(records, args.expected_frames, args.allow_sparse_raws)}
        audit, arrays = audit_sequence(records, expected_raws=expected)
        prefix = "scene%d_" % scene_index
        flips.update({prefix + name: value for name, value in arrays.items()})
        for group in audit["groups"]:
            for frame in group["frames"]:
                extra = additional[(group["method_arm"], frame["raw"])]
                frame["issues"].extend(extra["issues"])
                frame["verified"] = bool(frame.get("verified")) and not extra["issues"]
                frame["contact_metrics"] = extra["contact"]
                frame["producer_converged"] = extra["converged"]
                frame["float_audit_input_kind"] = "original full evidence.npz; not ui_evidence.npz"
            for pair in group["pairs"]:
                if pair.get("full_flip_array_prefix"):
                    pair["full_flip_array_prefix"] = prefix + pair["full_flip_array_prefix"]
            for triple in group["triples"]:
                triple["full_flip_array"] = prefix + triple["full_flip_array"]
            groups.append(group)
        print("AUDITED_CONTEXT", scene["context"], "frames", len(records), flush=True)
        del records, arrays, audit, entries
        gc.collect()
    parameters = parameter_audit(metadata_records)
    successful = [f for group in groups for f in group["frames"] if not f.get("skipped")]
    report = {"schema": "frame-independent-bounded-temporal-audit-v1", "groups": groups,
        "global_parameter_audit_by_arm": parameters,
        "producer_frame_reset_verified": bool(successful) and all(f.get("reset_verified") is True for f in successful),
        "possible_carryover_detected": any(f.get("reset_verified") is False for f in successful),
        "selection_modified_by_audit": False, "audit_accumulates_previous_deletions": False,
        "float_audit_input_kind": "original full floating evidence; generated before stripping light UI package",
        "labels_for_training": False, "global_semantic_unknown": False, "human_GT": None}
    summary = {"schema": "bounded-temporal-delivery-summary-v1", "source_root": str(root),
        "contexts": len(scenes), "expected_contexts": args.expected_contexts,
        "groups": lightweight_summary(groups, args.expected_frames, args.allow_sparse_raws),
        "global_parameter_audit_by_arm": parameters,
        "all_successful_frame_checks_passed": bool(successful) and all(f.get("verified") for f in successful),
        "all_contexts_present": len(scenes) == args.expected_contexts,
        "all_parameters_consistent": all(p["template_same_across_scenes"] is True and p["method_parameters_same_across_scenes"] is True for p in parameters),
        "producer_frame_reset_verified": report["producer_frame_reset_verified"],
        "full_evidence_bytes_read": full_evidence_bytes, "wall_seconds": time.monotonic() - started,
        "training_updates": 0, "GPU_calls": 0, "quality_accepted": False, "human_GT": None}
    summary["complete_and_verified"] = bool(summary["all_successful_frame_checks_passed"] and summary["all_contexts_present"]
            and summary["all_parameters_consistent"] and all(g["coverage_complete"] for g in summary["groups"])
            and all(g["residual_eligible_rows_sum"] == 0 for g in summary["groups"]))
    write_json(output / "temporal_audit.json", report)
    np.savez_compressed(output / "flip_ids.npz", **flips)
    write_json(output / "delivery_summary.json", summary)
    manifest = {"schema": "bounded-temporal-audit-delivery-manifest-v1", "source_full_root": str(root),
        "float_audit_performed_on_original_full_evidence": True, "full_float_evidence_preserved_at_source": True,
        "audit_files": {"temporal_audit": "temporal_audit.json", "delivery_summary": "delivery_summary.json", "flip_ids": "flip_ids.npz"},
        "frames": manifest_frames, "frame_image_paths_relative_to": "source_full_root (same structure in light delivery)",
        "human_GT": None, "scope": "local projected-center eligibility, after peeling and local white/external baseline composition; not global semantic unknown"}
    write_json(output / "manifest.json", manifest)
    if light is not None:
        for source in source_assets:
            destination = light / relative(source, root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        audit_copy = light / "audit"
        audit_copy.mkdir()
        for source in output.iterdir():
            if source.is_file():
                shutil.copy2(source, audit_copy / source.name)
        light_manifest = dict(manifest)
        light_manifest.update(schema="bounded-temporal-light-delivery-v1", full_float_evidence_included=False,
            ui_evidence_schema="bounded-ui-row-evidence-v1", ui_evidence_is_full_audit_input=False,
            audit_files={"temporal_audit": "audit/temporal_audit.json", "delivery_summary": "audit/delivery_summary.json", "flip_ids": "audit/flip_ids.npz"},
            float_audit_provenance="computed from original full remote float NPZ; light contains reused reports, IDs and bool scopes only")
        light_manifest["frames"] = [{**f, "float_check_source": "audit/temporal_audit.json"} for f in manifest_frames]
        light_manifest["frame_image_paths_relative_to"] = "this light manifest directory"
        summary["light_ui_evidence_complete"] = all(f["ui_evidence"] and not f["ui_missing_fields"] for f in manifest_frames if f["status"] == "ok")
        summary["complete_and_verified"] &= summary["light_ui_evidence_complete"]
        write_json(light / "manifest.json", light_manifest)
        summary["light_delivery"] = str(light)
        summary["light_delivery_bytes"] = sum(p.stat().st_size for p in light.rglob("*") if p.is_file())
        write_json(output / "delivery_summary.json", summary)
        shutil.copy2(output / "delivery_summary.json", audit_copy / "delivery_summary.json")
    print(json.dumps({"audit_dir": str(output), "light_dir": str(light) if light else None,
                      "contexts": len(scenes), "frames": len(manifest_frames), "complete_and_verified": summary["complete_and_verified"]}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--light-dir", type=Path)
    parser.add_argument("--summary", type=Path, action="append")
    parser.add_argument("--contexts", nargs="+")
    parser.add_argument("--expected-contexts", type=int, default=3)
    parser.add_argument("--expected-frames", type=int, default=49)
    parser.add_argument("--allow-sparse-raws", action="store_true")
    args = parser.parse_args()
    result = run(args)
    raise SystemExit(0 if result["complete_and_verified"] else 2)
