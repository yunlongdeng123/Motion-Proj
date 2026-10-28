"""CPU逐帧select→after与持久static row ID审计；不平滑/不累计删除。

同scene+checkpoint+arm内比较；scope/query与行翻转只是稳定性诊断，非语义GT。
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _ids(values, static_rows=None):
    values = np.asarray(values if values is not None else [], dtype=np.int64).reshape(-1)
    if (values < 0).any() or (static_rows is not None and (values >= static_rows).any()):
        raise ValueError("static row ID out of namespace range")
    if len(np.unique(values)) != len(values):
        raise ValueError("duplicate persistent row IDs")
    return np.sort(values)


def _iou(a, b):
    if a is None or b is None or np.shape(a) != np.shape(b):
        return None
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def _metadata(record):
    return {k: record.get(k) for k in ["context", "raw", "frame_id", "camera_index", "model_time", "checkpoint_ref",
            "method_arm", "query_mode", "init_mode", "outside_baseline", "row_namespace", "edit_case"]}


def audit_frame(record, arrays, *, require_reset_proof=True):
    result = _metadata(record)
    result.update(status=record.get("status", "ok"), skip_reason=record.get("skip_reason"), issues=[])
    if result["status"] != "ok":
        result.update(verified=False, skipped=True)
        return result
    for field in ["context", "checkpoint_ref", "row_namespace", "method_arm", "query_mode", "init_mode", "outside_baseline"]:
        if not record.get(field):
            result["issues"].append("required_provenance_missing:" + field)
    ns = record.get("static_rows")
    new = _ids(arrays.get("new_unknown_ids", arrays.get("selected_ids")), ns)
    existing = _ids(arrays.get("existing_removed_ids"), ns)
    eligible = _ids(arrays.get("eligible_ids"), ns)
    expected_start = np.empty(0, np.int64) if record.get("init_mode", "same_frame_existing") == "empty" else existing
    if record.get("init_mode") == "empty" and len(existing):
        result["issues"].append("empty-init arm contains old existing seed IDs")
    if record.get("query_mode") in ["source_body_contact", "source_local_band"] and record.get("native_RGB_content_used") is True:
        result["issues"].append("source-only query reports native RGB input")
    if "starting_removed_ids" in arrays:
        start = _ids(arrays["starting_removed_ids"], ns)
        reset = bool(np.array_equal(start, expected_start))
        if not reset:
            result["issues"].append("start_state_not_same_frame_expected_seed; possible previous-frame accumulation")
    else:
        reset = None
        if require_reset_proof:
            result["issues"].append("starting_removed_ids_missing; reset not verified")
    final = np.union1d(expected_start, new)
    if "final_removed_ids" in arrays and not np.array_equal(_ids(arrays["final_removed_ids"], ns), final):
        result["issues"].append("final_removed_ids_do_not_equal_this_frame_start_union_new")
    if np.intersect1d(new, expected_start).size:
        result["issues"].append("new_ids_overlap_start_seed")
    outside_new = np.setdiff1d(new, eligible)
    if len(outside_new):
        result["issues"].append("new_selected_centers_not_eligible")
    raw_residual = np.asarray(arrays.get("residual_ids", []), np.int64).reshape(-1)
    residual = _ids(raw_residual, ns)
    masses = np.asarray(arrays.get("residual_mass", np.zeros(len(residual))), float).reshape(-1)
    if len(masses) != len(residual):
        raise ValueError("residual ids/mass length mismatch")
    if not np.isfinite(masses).all() or (masses < 0).any():
        raise ValueError("after attribution residual mass is nonfinite or negative")
    masses = masses[np.argsort(raw_residual)]
    if np.intersect1d(residual, final).size:
        result["issues"].append("removed_rows_still_in_after_residual_ids")
    is_eligible = np.isin(residual, eligible)
    after_payload = all(k in arrays for k in ["residual_ids", "residual_mass", "alpha", "query"])
    after_claim = record.get("after_attribution_recomputed")
    if after_claim is not True:
        result["issues"].append("after_recompute_not_explicitly_recorded; payload alone does not prove recomputation")
    alpha = arrays.get("alpha")
    query = arrays.get("query")
    closure = None
    if after_payload:
        query = np.asarray(query, bool)
        alpha = np.asarray(alpha)
        if alpha.shape != query.shape or not np.isfinite(alpha).all():
            raise ValueError("after alpha/query shape or finite mismatch")
        alpha_mass = float(alpha[query].sum(dtype=np.float64))
        residual_mass = float(masses.sum())
        closure = abs(alpha_mass - residual_mass)
        epsilon = record.get("mass_epsilon", 1e-7)
        allowance = max(3e-4 * max(alpha_mass, 1), (ns or 1000000) * epsilon)
        if closure > allowance:
            result["issues"].append("after residual mass does not close to query alpha within exported threshold allowance")
    else:
        result["issues"].append("after_attribution_payload_incomplete")
    display = arrays.get("display_scope")
    baseline = arrays.get("baseline_native_rgb")
    white = arrays.get("white_rgb")
    gaussian = arrays.get("gaussian_rgb")
    outside_delta = inside_error = None
    if all(v is not None for v in [display, baseline, white, gaussian, alpha]):
        display = np.asarray(display, bool)
        baseline, white, gaussian = map(np.asarray, [baseline, white, gaussian])
        if baseline.shape != (3,) + display.shape or white.shape != baseline.shape or gaussian.shape != baseline.shape:
            raise ValueError("composition payload must be CHW with same-frame display scope")
        if not all(np.isfinite(v).all() for v in [baseline, white, gaussian]):
            raise ValueError("nonfinite local composition payload")
        if (~display).any():
            outside_delta = float(np.max(np.abs(white[:, ~display] - baseline[:, ~display])))
        else:
            outside_delta = 0.0
        inside_error = float(np.max(np.abs(white[:, display] - (gaussian + (1 - alpha)[None])[:, display]))) if display.any() else 0.
        if outside_delta != 0:
            result["issues"].append("outside display scope differs from same-frame baseline")
        if inside_error > 2e-6:
            result["issues"].append("inside display scope not G_after_plus_T_white")
    result.update(skipped=False, reset_verified=reset, new_rows=len(new), eligible_rows=len(eligible),
                  selected_rows_including_same_frame_seed=len(final), new_rows_outside_eligible=len(outside_new),
                  after_payload_present=after_payload, after_recomputed_claim=after_claim,
                  residual_eligible_rows=int(is_eligible.sum()), residual_eligible_mass=float(masses[is_eligible].sum()),
                  residual_outside_center_rows=int((~is_eligible).sum()), residual_outside_center_mass=float(masses[~is_eligible].sum()),
                  query_alpha_mass_closure_abs=closure, outside_display_max_delta=outside_delta,
                  inside_white_equation_max_error=inside_error,
                  verified=not result["issues"], labels_for_training=False, global_semantic_unknown=False)
    return result


def audit_sequence(records, *, expected_raws=None, require_reset_proof=True, max_examples=32):
    """records每项含metadata及arrays；返回JSON报告和完整flip-ID数组供NPZ持久化。"""
    groups = {}
    for record in records:
        key = tuple(str(record.get(k, "")) for k in ["context", "checkpoint_ref", "row_namespace", "method_arm", "edit_case"])
        groups.setdefault(key, []).append(record)
    reports, full_ids = [], {}
    for group_index, (key, values) in enumerate(groups.items()):
        values = sorted(values, key=lambda r: (r.get("camera_index", r["raw"]), r["raw"]))
        raw_list = [r["raw"] for r in values]
        if len(set(raw_list)) != len(raw_list):
            raise ValueError("duplicate raw in same context/checkpoint/arm/edit-case group")
        frame_reports = [audit_frame(r, r.get("arrays", {}), require_reset_proof=require_reset_proof) for r in values]
        pair_reports, triple_reports = [], []
        for index in range(1, len(values)):
            previous, current = values[index - 1:index + 1]
            adjacent = current["raw"] - previous["raw"] == 1
            good = previous.get("status", "ok") == "ok" and current.get("status", "ok") == "ok"
            pair = {"raw_from": previous["raw"], "raw_to": current["raw"], "consecutive_raw": adjacent,
                    "comparison_available": good, "not_temporal_flicker_proof": True}
            if not good:
                pair["reason"] = "missing/skipped frame; do not bridge by carrying previous selection"
                pair_reports.append(pair)
                continue
            a, b = previous["arrays"], current["arrays"]
            na, nb = _ids(a.get("new_unknown_ids", a.get("selected_ids"))), _ids(b.get("new_unknown_ids", b.get("selected_ids")))
            ea, eb = _ids(a.get("eligible_ids")), _ids(b.get("eligible_ids"))
            shared = np.intersect1d(ea, eb)
            entered, exited = np.setdiff1d(nb, na), np.setdiff1d(na, nb)
            on_shared, off_shared = np.intersect1d(entered, shared), np.intersect1d(exited, shared)
            union, intersection = np.union1d(na, nb), np.intersect1d(na, nb)
            prefix = "g%d_%d_%d" % (group_index, previous["raw"], current["raw"])
            for name, ids in [("on", entered), ("off", exited), ("on_shared_eligible", on_shared), ("off_shared_eligible", off_shared)]:
                full_ids[prefix + "_" + name] = ids.astype(np.int32)
            pair.update(new_set_jaccard=float(len(intersection) / len(union)) if len(union) else 1.,
                        new_rows_on=len(entered), new_rows_off=len(exited),
                        on_within_shared_eligible=len(on_shared), off_within_shared_eligible=len(off_shared),
                        eligibility_entered=len(np.setdiff1d(eb, ea)), eligibility_exited=len(np.setdiff1d(ea, eb)),
                        allowed_pixel_iou=_iou(a.get("allowed"), b.get("allowed")), query_pixel_iou=_iou(a.get("query"), b.get("query")),
                        on_examples=entered[:max_examples].tolist(), off_examples=exited[:max_examples].tolist(),
                        full_flip_array_prefix=prefix)
            if not adjacent:
                pair["reason"] = "nonconsecutive anchor frames; report set difference, not framewise flicker"
            pair_reports.append(pair)
        for index in range(2, len(values)):
            triple = values[index - 2:index + 1]
            if any(r.get("status", "ok") != "ok" for r in triple) or any(triple[j + 1]["raw"] - triple[j]["raw"] != 1 for j in [0, 1]):
                continue
            a, b, c = [r["arrays"] for r in triple]
            eligible = np.intersect1d(np.intersect1d(_ids(a.get("eligible_ids")), _ids(b.get("eligible_ids"))), _ids(c.get("eligible_ids")))
            selected = [np.isin(eligible, _ids(v.get("new_unknown_ids", v.get("selected_ids")))) for v in [a, b, c]]
            oscillation = (selected[0] == selected[2]) & (selected[0] != selected[1])
            ids = eligible[oscillation]
            name = "g%d_%d_%d_aba_shared_eligible" % (group_index, triple[0]["raw"], triple[2]["raw"])
            full_ids[name] = ids.astype(np.int32)
            triple_reports.append({"raws": [r["raw"] for r in triple], "shared_eligible_aba_rows": len(ids),
                                   "row_examples": ids[:max_examples].tolist(), "full_flip_array": name})
        templates = [r.get("template") for r in values if r.get("status", "ok") == "ok"]
        template_consistent = all(t == templates[0] for t in templates) if templates and all(t is not None for t in templates) else None
        expected = expected_raws.get(key[0], []) if isinstance(expected_raws, dict) else expected_raws
        reports.append({"context": key[0], "checkpoint_ref": key[1], "row_namespace": key[2], "method_arm": key[3], "edit_case": key[4],
                        "frames_attempted": len(values), "successful_frames": sum(r["status"] == "ok" for r in frame_reports),
                        "skipped_frames": [{"raw": r["raw"], "reason": r["skip_reason"]} for r in frame_reports if r["status"] != "ok"],
                        "missing_expected_raws": sorted(set(expected or []) - set(raw_list)),
                        "template_parameters_consistent": template_consistent,
                        "frames": frame_reports, "pairs": pair_reports, "triples": triple_reports})
    arms = {}
    for record in records:
        if record.get("status", "ok") == "ok":
            arms.setdefault(record.get("method_arm", ""), []).append(record)
    arm_parameters = []
    for arm, values in arms.items():
        templates, params = [r.get("template") for r in values], [r.get("method_parameters") for r in values]
        arm_parameters.append({"method_arm": arm, "scene_contexts": sorted(set(r["context"] for r in values)),
             "template_same_across_scenes": all(v == templates[0] for v in templates) if all(v is not None for v in templates) else None,
             "method_parameters_same_across_scenes": all(v == params[0] for v in params) if all(v is not None for v in params) else None,
             "query_modes": sorted(set(r.get("query_mode", "") for r in values)),
             "init_modes": sorted(set(r.get("init_mode", "") for r in values)),
             "outside_baselines": sorted(set(r.get("outside_baseline", "") for r in values))})
    resets = [r.get("reset_verified") for g in reports for r in g["frames"] if not r.get("skipped")]
    reset_verified = True if resets and all(v is True for v in resets) else False if any(v is False for v in resets) else None
    return {"schema": "frame-independent-bounded-temporal-audit-v1", "groups": reports,
            "global_parameter_audit_by_arm": arm_parameters,
            "selection_modified_by_audit": False, "audit_accumulates_previous_deletions": False,
            "producer_frame_reset_verified": reset_verified,
            "possible_carryover_detected": any(v is False for v in resets),
            "labels_for_training": False, "human_GT": None}, full_ids


def load_records(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    records = []
    for item in manifest["frames"]:
        record = {**manifest.get("defaults", {}), **item}
        if record.get("status", "ok") != "ok":
            record["arrays"] = {}
        else:
            evidence = Path(record["evidence"])
            evidence = evidence if evidence.is_absolute() else path.parent / evidence
            with np.load(evidence, allow_pickle=False) as f:
                record["arrays"] = {k: f[k] for k in f.files}
        records.append(record)
    return records, manifest.get("expected_raws")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-legacy-missing-reset-proof", action="store_true")
    args = parser.parse_args()
    records, expected = load_records(args.manifest)
    report, arrays = audit_sequence(records, expected_raws=expected,
                                   require_reset_proof=not args.allow_legacy_missing_reset_proof)
    if args.output.exists() or args.output.with_suffix(".ids.npz").exists():
        raise FileExistsError("preserve previous audit output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".ids.npz"), **arrays)
    print(json.dumps({"groups": len(report["groups"]), "output": str(args.output),
                      "full_flip_ids": str(args.output.with_suffix(".ids.npz"))}, ensure_ascii=False))
