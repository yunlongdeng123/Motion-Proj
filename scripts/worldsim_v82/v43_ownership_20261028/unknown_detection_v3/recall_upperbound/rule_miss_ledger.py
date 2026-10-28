"""召回上界规则台账；CPU，无背景veto、训练或语义GT生成。

剥离后贡献只用于机械来源/漏选诊断，不能倒灌为训练真值。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "2"

import argparse
import ast
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label

WEIGHT_COLUMNS = ["old_domain", "broad_domain", "body", "effect", "outside", "total"]
DEFAULT_CONFIG = {"body_min_mass": .02, "body_min_fraction": .05, "body_repeat_frames": 2,
                  "single_core_min_mass": .05, "single_core_min_fraction": .20,
                  "shadow_min_mass": .01, "shadow_min_fraction": .05,
                  "boundary_fraction": .2}
ROW_LABELS = {"scope_missed": 1, "inside_but_gated": 2, "newly_visible": 4,
              "not_in_old_probe_export": 8, "entered_old_domain_only_after_removal": 16}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def source_config(path):
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets):
            config.update(ast.literal_eval(node.value))
    return config


def catalog(cfg, classify_source):
    row = {
        "body_mass": "body_mass > %s" % cfg["body_min_mass"],
        "body_fraction": "body_mass / max(total,1e-8) > %s" % cfg["body_min_fraction"],
        "body_repeat": "body_frames >= %s" % cfg["body_repeat_frames"],
        "single_core_mass": "core_mass > %s" % cfg["single_core_min_mass"],
        "single_core_fraction": "core_mass / total > %s" % cfg["single_core_min_fraction"],
        "body_support_requirement": "body_repeat OR (single_core_mass AND single_core_fraction)",
        "boundary_mass": "ring_mass > .02", "boundary_fraction": "ring_mass/total > %s" % cfg["boundary_fraction"],
        "boundary_repeat": "body_frames >= %s" % cfg["body_repeat_frames"], "boundary_body_mass": "body_mass > .001",
        "strict_shadow_mass": "strict_shadow_mass > %s" % cfg["shadow_min_mass"],
        "strict_shadow_fraction": "strict_shadow_mass/total > %s" % cfg["shadow_min_fraction"],
        "photo_shadow_mass": "photo_shadow_mass > %s" % cfg["shadow_min_mass"],
        "photo_shadow_fraction": "photo_shadow_mass/total > %s" % cfg["shadow_min_fraction"],
    }
    return {"row_selection_gates": row,
            "formula": {"body": "body_mass AND body_fraction AND body_support_requirement",
                        "boundary": "boundary_mass AND boundary_fraction AND boundary_repeat AND boundary_body_mass",
                        "shadow_strict": "strict_shadow_mass AND strict_shadow_fraction",
                        "shadow_photo": "photo_shadow_mass AND photo_shadow_fraction",
                        "photo_profile": "body OR boundary OR shadow_strict OR shadow_photo",
                        "shadow_profile": "shadow_strict OR shadow_photo"},
            "input_scope_and_export_gates": ["raw target body mask fixed upstream; core erosion radius clip(round(.04*sqrt(body_pixels)),1,5)",
                "ring=(distance_to_body<=radius) AND NOT core; contains body boundary and exterior band",
                "probe exported ids iff without_actor body>1e-4 OR without_actor ring>1e-4 OR full body>1e-4",
                "render evidence exported ids iff static_role>0 OR weak_bg_mass>.02; this can hide subthreshold contributors",
                "recall broad domain is fixed raw-body-relative bbox expanded once and OR old_domain; it is an observation range, not car GT"],
            "independent_shadow_upstream_gates": ["same-scene/target original observations only; semantic prediction is Cityscapes+1",
                "local actual-road plane >=100 inliers, residual<=.10m, fit radius15m and residual inlier .12m",
                "candidate_domain world footprint distance<=5m; query inside camera; body excluded in r2 strict",
                "query road confidence>=.55; actual-road LiDAR pixel radius<=2.5 and depth residual<=.65m; other actor clearance>=1m",
                "source road confidence>=.75, dynamic semantic dilation3px, all-actor clearance>=2.5m and target>=4m",
                "source luminosity .05..95; exposure >=60 actual shared anchors, medianRGB error<=.045, gain .65..1.5",
                ">=3 different source frames; raw span>=5; reference medianRGB deviation<=.045",
                "query luminance ratio .25..84; luminance drop>=.035; multiplicative RGB residual<=.035",
                "target-carried witnesses offsets[-6,-3,3,6], target displacement>=.8m, >=2 visible darkening supports",
                "strict connected component >=6 pixels",
                "r3 propagation requires strict seed, independent per-pixel photo, color/ref continuity .045, ratio neighbor .12, step<=.8m, geodesic<=1.5m",
                "old r3 propagation blocked weak_bg/body/actor/foreground; this historical rule is audited, NEVER reused as recall veto",
                "contact uses direct frontmost actual LiDAR height within .05m of plane, raw bbox±4 and bottom[-8,+5], same photo gates; no body hard exclusion"],
            "not_candidate_veto_gates": ["weak background depth-compatible mass only marks mixed; mixed is still removed",
                "exterior fraction only marks mixed, not candidate veto", "target-depth moments are computed but unused by current classify",
                "shadow confidence score cap does not affect current bool-to-adjoint row selection", "semantic class is not a direct row selection condition"],
            "new_probe_thresholds": ["producer mass_epsilon from summary, normally1e-4", "later diagnostic peeling may use1e-7 and finite round budget; record exact producer receipt"],
            "source_classify": str(classify_source), "background_veto": False, "human_GT": None}


def tri_and(*values):
    result = np.ones_like(values[0], np.int8)
    any_false = np.zeros_like(result, bool)
    any_unknown = np.zeros_like(result, bool)
    for value in values:
        any_false |= value == 0
        any_unknown |= value < 0
    result[any_unknown] = -1
    result[any_false] = 0
    return result


def tri_or(*values):
    result = np.zeros_like(values[0], np.int8)
    any_true = np.zeros_like(result, bool)
    any_unknown = np.zeros_like(result, bool)
    for value in values:
        any_true |= value == 1
        any_unknown |= value < 0
    result[any_unknown] = -1
    result[any_true] = 1
    return result


def comparison(values, threshold, inclusive=False, upper=None):
    known = np.isfinite(values)
    result = np.full(len(values), -1, np.int8)
    result[known] = (values[known] >= threshold if inclusive else values[known] > threshold).astype(np.int8)
    if upper is not None:
        definitely_false = upper < threshold if inclusive else upper <= threshold
        result[~known & definitely_false] = 0
    return result


def dense_weights(path, n, omitted_zero=True):
    with np.load(path, allow_pickle=False) as f:
        ids, weights = f["ids"].astype(np.int64), f["weights"].astype(np.float32)
        if weights.shape != (len(ids), 6):
            raise ValueError((str(path), weights.shape, WEIGHT_COLUMNS))
        if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= n).any():
            raise ValueError("static行ID命名空间错误: " + str(path))
        dense = np.full((n, 6), 0 if omitted_zero else np.nan, np.float32)
        dense[ids] = weights
        removed = np.zeros(n, bool)
        if "removed_ids" in f.files:
            rid = f["removed_ids"].astype(np.int64)
            if ((rid < 0) | (rid >= n)).any():
                raise ValueError("removed_ids越界")
            removed[rid] = True
        elif "removed" in f.files:
            removed[ids] = f["removed"].astype(bool)
    return dense, removed


def body_history(probe, n, cfg):
    counts = np.zeros(n, np.uint16)
    paths = sorted(probe.glob("[0-9][0-9][0-9].npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            w, ids = f["without_actor"], f["ids"]
            hit = (w[:, 0] > cfg["body_min_mass"]) & (w[:, 0] / np.maximum(w[:, -1], 1e-8) > cfg["body_min_fraction"])
            counts[ids[hit]] += 1
    return counts, len(paths)


def gate_inputs(raw, initial, n, counts, probe, old_profile, old_strict, exact_gates=None):
    total = np.maximum(initial[:, 5], 1e-8)
    data = {"body": initial[:, 2].copy(), "photo": initial[:, 3].copy(), "total": total,
            "history": counts.astype(np.float32), "core": np.full(n, np.nan, np.float32),
            "boundary": np.full(n, np.nan, np.float32), "strict": np.full(n, np.nan, np.float32)}
    present = np.zeros(n, bool)
    probe_path = probe / ("%03d.npz" % raw)
    consistency = {}
    if probe_path.exists():
        with np.load(probe_path, allow_pickle=False) as f:
            ids, w = f["ids"], f["without_actor"]
            present[ids] = True
            data["core"][ids], data["boundary"][ids] = w[:, 1], w[:, 2]
            consistency["body_mass_max_abs_difference_on_common_probe_ids"] = float(np.nanmax(np.abs(w[:, 0] - initial[ids, 2]))) if len(ids) else None
    # 原probe的active>1e-4导出门槛只作coverage事实；不将缺失任意写成0。
    data["core_upper"] = initial[:, 2].copy()
    data["boundary_upper"] = np.full(n, np.inf, np.float32)
    if present.any() and np.nanmax(initial[~present, 2], initial=0) <= 1.01e-4 and consistency.get("body_mass_max_abs_difference_on_common_probe_ids", 1) < 1e-3:
        data["core_upper"][~present] = np.minimum(data["core_upper"][~present], 1e-4)
        data["boundary_upper"][~present] = 1e-4
        consistency["old_probe_export_upper_bound_verified"] = True
    else:
        consistency["old_probe_export_upper_bound_verified"] = False
    # effect图明确是strict|photo，故strict质量不能大于effect质量。
    data["strict_upper"] = initial[:, 3].copy()
    if old_strict is not None:
        path = old_strict / ("%03d.npz" % raw)
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                data["strict"][f["ids"]] = f["shadow_mass"]
    old_selected = np.zeros(n, bool)
    old_bg = np.zeros(n, bool)
    if old_profile is not None:
        path = old_profile / ("%03d.npz" % raw)
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                ids = f["ids"]
                old_selected[ids] = f["static_role"] >= 2
                if "bg_weak" in f.files:
                    old_bg[ids] = f["bg_weak"].astype(bool)
    if exact_gates is not None:
        path = exact_gates / ("%03d_gate_inputs.npz" % raw)
        if path.exists():
            summary = read_json(exact_gates / "summary.json", {})
            channels = summary.get("channels", ["body", "core", "boundary", "shadow_strict", "shadow_photo", "background_weak",
                       "background_depth_sum", "background_depth_squared", "target_depth_valid", "target_depth_sum", "target_depth_squared", "candidate_union", "total"])
            with np.load(path, allow_pickle=False) as f:
                ids, w = f["ids"].astype(int), f["weights"]
                if w.shape[1] != len(channels) or ((ids < 0) | (ids >= n)).any():
                    raise ValueError("exact gate channels/static ids不匹配")
                for field, channel in [("body", "body"), ("core", "core"), ("boundary", "boundary"),
                                       ("strict", "shadow_strict"), ("photo", "shadow_photo"), ("total", "total")]:
                    data[field] = np.zeros(n, np.float32)
                    data[field][ids] = w[:, channels.index(channel)]
                data["total"] = np.maximum(data["total"], 1e-8)
                data["history"] = np.zeros(n, np.float32)
                if "body_counts" in f.files:
                    data["history"][ids] = f["body_counts"]
                else:
                    data["history"] = counts.astype(np.float32)
                old_bg = np.zeros(n, bool)
                bg = w[:, channels.index("background_weak")]
                denom = np.maximum(bg, 1e-8)
                mean = w[:, channels.index("background_depth_sum")] / denom
                variance = np.maximum(w[:, channels.index("background_depth_squared")] / denom - mean ** 2, 0)
                rms = np.sqrt((f["center_z"] - mean) ** 2 + variance)
                old_bg[ids] = (bg > .05) & (rms < .5)
            data["core_upper"] = data["core"].copy()
            data["boundary_upper"] = data["boundary"].copy()
            data["strict_upper"] = data["strict"].copy()
            consistency["exact_gate_input_source"] = str(path)
            consistency["exact_gate_channels"] = channels
            consistency["exact_sparse_omitted_rows_zero"] = True
    return data, present, old_selected, old_bg, consistency


def row_gates(data, cfg):
    total = data["total"]
    return {
        "body_mass": comparison(data["body"], cfg["body_min_mass"]),
        "body_fraction": comparison(data["body"] / total, cfg["body_min_fraction"]),
        "body_repeat": comparison(data["history"], cfg["body_repeat_frames"], inclusive=True),
        "single_core_mass": comparison(data["core"], cfg["single_core_min_mass"], upper=data["core_upper"]),
        "single_core_fraction": comparison(data["core"] / total, cfg["single_core_min_fraction"], upper=data["core_upper"] / total),
        "boundary_mass": comparison(data["boundary"], .02, upper=data["boundary_upper"]),
        "boundary_fraction": comparison(data["boundary"] / total, cfg["boundary_fraction"], upper=data["boundary_upper"] / total),
        "boundary_repeat": comparison(data["history"], cfg["body_repeat_frames"], inclusive=True),
        "boundary_body_mass": comparison(data["body"], .001),
        "strict_shadow_mass": comparison(data["strict"], cfg["shadow_min_mass"], upper=data["strict_upper"]),
        "strict_shadow_fraction": comparison(data["strict"] / total, cfg["shadow_min_fraction"], upper=data["strict_upper"] / total),
        "photo_shadow_mass": comparison(data["photo"], cfg["shadow_min_mass"]),
        "photo_shadow_fraction": comparison(data["photo"] / total, cfg["shadow_min_fraction"]),
    }


def evaluate(gates, profile="photo", drop=()):
    g = {k: np.ones_like(v) if k in drop else v for k, v in gates.items()}
    core = tri_and(g["single_core_mass"], g["single_core_fraction"])
    body_support = tri_or(g["body_repeat"], core)
    if "body_support_requirement" in drop:
        body_support = np.ones_like(body_support)
    body = tri_and(g["body_mass"], g["body_fraction"], body_support)
    boundary = tri_and(g["boundary_mass"], g["boundary_fraction"], g["boundary_repeat"], g["boundary_body_mass"])
    strict = tri_and(g["strict_shadow_mass"], g["strict_shadow_fraction"])
    photo = tri_and(g["photo_shadow_mass"], g["photo_shadow_fraction"])
    shadow = strict if profile == "strict" else tri_or(strict, photo)
    return shadow if profile == "shadow" else tri_or(body, boundary, shadow)


def candidate_variants(gates, data, initial, old_selected, epsilon, profile):
    baseline_state = evaluate(gates, profile)
    result = {"original_selected": old_selected.copy()}
    info = {}
    for gate in list(gates) + ["body_support_requirement"]:
        state = evaluate(gates, profile, drop=[gate])
        mask = old_selected | (state == 1)
        result["drop_" + gate] = mask
        info["drop_" + gate] = {"selected": int(mask.sum()), "definite_additions": int((~old_selected & (state == 1)).sum()),
                                 "possible_additions_missing_gate_fields": int((~old_selected & (state < 0)).sum())}
    result["any_positive_body"] = old_selected | (data["body"] > epsilon)
    result["any_positive_body_or_boundary"] = result["any_positive_body"] | (np.nan_to_num(data["boundary"]) > epsilon)
    result["all_initial_old_domain_contributors"] = old_selected | (initial[:, 0] > epsilon)
    result["all_initial_broad_domain_contributors"] = old_selected | (initial[:, 1] > epsilon)
    for name in result:
        if name not in info:
            info[name] = {"selected": int(result[name].sum()), "definite_additions": int((result[name] & ~old_selected).sum())}
    mismatch = {"known_formula_disagrees_original_membership": int(((baseline_state >= 0) & ((baseline_state == 1) != old_selected)).sum()),
                "formula_unknown_missing_fields": int((baseline_state < 0).sum())}
    return result, info, baseline_state, mismatch


def discover(campaign, raws):
    summary = read_json(campaign / "summary.json", {})
    records = summary.get("records", [])
    queries = []
    for raw in raws:
        directory = campaign / ("%03d" % raw)
        initial = directory / "initial_weights.npz"
        if not initial.exists():
            continue
        order = {r["stage"]: i for i, r in enumerate(records) if r.get("raw") == raw}
        paths = list(directory.glob("*/contributors.npz"))
        def sort_key(path):
            name = path.parent.name
            fixed = {"actor_omitted": 0, "existing": 1, "gates_open_same_domain": 2, "broad_once": 3}
            return order.get(name, fixed.get(name, 4 + int(re.search(r"\d+", name).group()) if re.search(r"\d+", name) else 999))
        stages = [{"name": p.parent.name, "path": p, "receipt": read_json(p.parent / "receipt.json", {})} for p in sorted(paths, key=sort_key)]
        queries.append({"raw": raw, "initial": initial, "stages": stages})
    return queries, summary


def normalize_manifest(path):
    manifest = json.loads(path.read_text())
    base = path.parent
    queries = []
    for query in manifest["queries"]:
        initial = Path(query["initial"])
        initial = initial if initial.is_absolute() else base / initial
        stages = []
        for stage in query["stages"]:
            file = Path(stage.get("path", stage.get("contributors")))
            file = file if file.is_absolute() else base / file
            stages.append({"name": stage["name"], "path": file,
                           "receipt": read_json(file.parent / "receipt.json", stage.get("metadata", {}))})
        queries.append({"raw": int(query["raw"]), "initial": initial, "stages": stages})
    return queries, manifest


def stage_audit(stage, initial, after, removed, old_selected, probe_present, gates, variants, old_bg, epsilon, previous):
    remaining = after[:, 1] > epsilon
    initial_old, initial_broad = initial[:, 0] > epsilon, initial[:, 1] > epsilon
    scope = remaining & (after[:, 0] <= epsilon)
    gated = remaining & initial_old & ~old_selected
    newly = remaining & ~initial_broad
    after_entered_old = remaining & ~initial_old & (after[:, 0] > epsilon)
    bits = np.zeros(len(remaining), np.uint8)
    for bit, mask in [(1, scope), (2, gated), (4, newly), (8, remaining & ~probe_present), (16, after_entered_old)]:
        bits[mask] |= bit
    assert not (removed & (after[:, 5] > 1e-8)).any(), "被删static仍有贡献，检查ID/剥离/输入状态"
    counts = {"remaining_broad_contributors": int(remaining.sum()), "remaining_broad_alpha_mass": float(after[remaining, 1].sum()),
              "scope_missed": int(scope.sum()), "inside_but_gated": int(gated.sum()), "newly_visible_vs_initial": int(newly.sum()),
              "newly_visible_vs_previous_stage": int((remaining & (previous[:, 1] <= epsilon)).sum()) if previous is not None else None,
              "not_in_old_probe_export": int((remaining & ~probe_present).sum()),
              "entered_old_domain_only_after_removal": int(after_entered_old.sum()),
              "original_selected_still_contributing": int((remaining & old_selected).sum()),
              "background_counterevidence_overlap_record_only": int((remaining & old_bg).sum()),
              "gate_fail_counts_multilabel": {name: int((gated & (value == 0)).sum()) for name, value in gates.items()},
              "gate_unknown_counts_multilabel": {name: int((gated & (value < 0)).sum()) for name, value in gates.items()},
              "rule_relaxation_remaining_coverage": {name: int((remaining & mask).sum()) for name, mask in variants.items()},
              "source_artifact": str(stage["path"]), "selection_source": stage["receipt"].get("selection_source"),
              "background_veto": False, "labels_for_training": False, "human_GT": None}
    region = stage["receipt"].get("regions", {}).get("broad_domain", {})
    counts["non_gaussian_environment_diagnostic"] = {key: region.get(key) for key in
          ["alpha_mean", "env_rgb_mean", "negative_environment_pixels", "negative_native_pixels"]}
    counts["non_gaussian_environment_diagnostic"]["source"] = "fixed original environment multiplied by post-removal transmittance; no row ID, not car GT"
    return bits, counts


def pixel_gate_audit(raw, shadow, output):
    if shadow is None or not (shadow / ("%03d.npz" % raw)).exists():
        return {"available": False, "reason": "independent shadow arrays unavailable"}
    with np.load(shadow / ("%03d.npz" % raw), allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    if not {"reason_bits", "source_count", "luminance_ratio", "motion_witness_support_count"} <= a.keys():
        return {"available": False, "reason": "insufficient independent shadow fields"}
    domain = a["candidate_domain"]
    reason = a["reason_bits"]
    gates = {"body_exclusion": ~a["body"], "query_road_semantic": (reason & 4) == 0,
             "query_actual_lidar": (reason & 8) == 0, "query_other_actor_clear": (reason & 16) == 0,
             "source_count_3": a["source_count"] >= 3, "source_span_5": a["source_span_raw"] >= 5,
             "source_rgb_stable": a["reference_rgb_deviation"] <= .045,
             "luminance_ratio_lower": a["luminance_ratio"] >= .25,
             "luminance_ratio_upper": a["luminance_ratio"] <= .84,
             "luminance_drop": a["luminance_drop"] >= .035,
             "multiplicative_rgb_residual": a["attenuation_rgb_residual"] <= .035,
             "motion_support_2": a["motion_witness_support_count"] >= 2}
    strict = a.get("shadow_candidate_strict", a["shadow_candidate"])
    results = {}
    for drop in list(gates) + ["component_minimum_6", "none"]:
        mask = domain.copy()
        for name, value in gates.items():
            if name != drop:
                mask &= value
        components, n = label(mask)
        size = np.bincount(components.ravel())
        retained = mask if drop == "component_minimum_6" else mask & (size[components] >= 6)
        results[drop] = {"before_component_filter_pixels": int(mask.sum()), "after_component_filter_pixels": int(retained.sum()),
                         "added_vs_original_strict_pixels": int((retained & ~strict).sum())}
    return {"available": True, "source": str(shadow / ("%03d.npz" % raw)), "original_strict_pixels": int(strict.sum()),
            "saved_spatial_domain_pixels": int(domain.sum()), "body_pixels": int(a["body"].sum()),
            "single_gate_drop": results, "scope_limit": "outside saved candidate_domain raw comparisons are missing, not zero; domain widening needs the independent broad contribution probe",
            "missing_reference_data_pixels_in_saved_domain": int((domain & ~np.isfinite(a["luminance_ratio"])).sum()),
            "background_counterevidence_pixels_record_only": int(a["road_counterevidence_valid"].sum()),
            "background_veto": False, "human_GT": None}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = source_config(args.classify_source)
    queries, campaign_summary = normalize_manifest(args.manifest) if args.manifest else discover(args.campaign, args.raw)
    if not queries:
        raise ValueError("没有完整initial_weights输入；不把未完成GPU导出当零")
    probe_summary = read_json(args.body_probe / "summary.json", {})
    n = args.static_rows or probe_summary.get("static_rows")
    if n is None:
        with np.load(queries[0]["initial"], allow_pickle=False) as f:
            n = int(f["ids"].max()) + 1
    n = int(n)
    counts, history_frames = body_history(args.body_probe, n, cfg)
    epsilon = args.epsilon if args.epsilon is not None else campaign_summary.get("mass_epsilon", 1e-4)
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {"schema": "recall-upperbound-rule-miss-ledger-v1", "static_rows": n, "row_namespace": "original_static_rows",
                "initial_columns": WEIGHT_COLUMNS, "profile": args.profile, "epsilon": epsilon,
                "configuration": cfg, "body_history_frames": history_frames,
                "category_bits_multilabel": ROW_LABELS,
                "scope_definition": "stage old_domain<=epsilon and broad_domain>epsilon; weighted support, not proof of geometric footprint outside",
                "gate_definition": "initial old_domain>epsilon and absent from original selected set",
                "newly_visible_definition": "stage broad_domain>epsilon and initial broad_domain<=epsilon; fixed actor-omitted reference",
                "background_veto": False, "background_false_positives_record_only": True,
                "post_suppression_contribution_role": "mechanical audit only; never semantic or training GT",
                "edited_RGB_content_used_for_rule_labels": False, "training_updates": 0, "GPU_calls": 0, "human_GT": None,
                "omitted_stage_rows_are_zero": True, "source_classify": str(args.classify_source),
                "body_probe": str(args.body_probe), "old_profile": str(args.old_profile),
                "classify_nonselection_notes": ["bg_rms and exterior fraction only change mixed role; mixed is still removed, so these are not candidate veto gates",
                                                "target_depth_valid/sum/squared are presently unused by classify; no depth veto is invented here",
                                                "semantic class is not a direct row candidate gate; frozen actor has its own namespace"]}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (args.output / "gate_catalog.json").write_text(json.dumps(catalog(cfg, args.classify_source), indent=2), encoding="utf-8")
    all_reports = []
    for query in queries:
        raw = query["raw"]
        destination = args.output / ("%03d" % raw)
        destination.mkdir()
        initial, _ = dense_weights(query["initial"], n)
        data, present, old_selected, old_bg, consistency = gate_inputs(raw, initial, n, counts, args.body_probe, args.old_profile, args.old_strict, args.exact_gates)
        for stage in query["stages"]:
            if stage["name"] == "existing":
                _, old_selected = dense_weights(stage["path"], n)
                break
        gates = row_gates(data, cfg)
        variants, relaxations, baseline_state, mismatch = candidate_variants(gates, data, initial, old_selected, epsilon, args.profile)
        np.savez_compressed(destination / "rule_relaxations.npz", ids=np.arange(n, dtype=np.int32),
                            original_formula_state=baseline_state, **variants)
        gate_rows = {name: {"pass": int((state == 1).sum()), "fail": int((state == 0).sum()), "unknown": int((state < 0).sum())} for name, state in gates.items()}
        stages, previous = [], None
        for stage in query["stages"]:
            after, removed = dense_weights(stage["path"], n)
            bits, report = stage_audit(stage, initial, after, removed, old_selected, present, gates, variants, old_bg, epsilon, previous)
            ids = np.flatnonzero(after[:, 1] > epsilon)
            np.savez_compressed(destination / (stage["name"] + "_misses.npz"), ids=ids.astype(np.int32),
                                category_bits=bits[ids], initial_weights=initial[ids], after_weights=after[ids],
                                original_selected=old_selected[ids], removed=removed[ids],
                                gate_names=np.array(list(gates)), gate_states=np.column_stack([v[ids] for v in gates.values()]))
            report["stage"] = stage["name"]
            stages.append(report)
            previous = after
        pixel = pixel_gate_audit(raw, args.shadow, destination)
        report = {"raw": raw, "static_rows": n, "original_selected": int(old_selected.sum()),
                  "fine_gate_data_consistency": consistency, "formula_membership_audit": mismatch,
                  "row_gate_states": gate_rows, "single_gate_and_scope_relaxations": relaxations,
                  "independent_shadow_pixel_gate_audit": pixel, "stages": stages,
                  "source_initial": str(query["initial"]), "background_veto": False, "labels_for_training": False, "human_GT": None}
        (destination / "ledger.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        all_reports.append(report)
        print("RULE_LEDGER", raw, "old_selected", report["original_selected"], "stages", len(stages),
              "formula_disagreement", mismatch["known_formula_disagrees_original_membership"], flush=True)
    summary = {"schema": contract["schema"], "queries": all_reports, "background_veto": False,
               "labels_for_training": False, "human_GT": None, "GPU_calls": 0}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--body-probe", type=Path, required=True)
    p.add_argument("--old-profile", type=Path)
    p.add_argument("--old-strict", type=Path)
    p.add_argument("--shadow", type=Path)
    p.add_argument("--exact-gates", type=Path)
    p.add_argument("--classify-source", type=Path)
    p.add_argument("--static-rows", type=int)
    p.add_argument("--raw", nargs="+", type=int, default=[98, 113, 114])
    p.add_argument("--profile", choices=["photo", "strict", "shadow"], default="photo")
    p.add_argument("--epsilon", type=float)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (a.campaign is None) == (a.manifest is None):
        p.error("只提供--campaign目录或--manifest JSON之一")
    run(a)
