"""Read-only CPU diagnostics for saved V44 paired floating-point renders.

This script changes no renderer, training, calibration, masks or source receipts.
It prints JSON to stdout. It deliberately preserves the original empty blue-road
domain and reports observed-background diagnostics under different explicit names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HELD = {89, 95, 101, 105, 113, 121, 125, 130, 137}
BLUE = {113: (276, 175, 309, 220), 137: (276, 175, 309, 220)}
SILVER = {130: (202, 192, 298, 244), 137: (210, 213, 290, 233)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def normalized_rgb(value):
    rgb = np.asarray(value, dtype=np.float32)
    if rgb.max() > 1:
        rgb = rgb / 255
    return rgb.transpose(2, 0, 1)


def stats(value, domain):
    selected = np.asarray(value, np.float64)[domain]
    count = int(domain.sum())
    total = float(selected.sum())
    return dict(pixels=count, sum=total, mean=total/count if count else None)


def rectangle(box):
    x0, y0, x1, y1 = box
    domain = np.zeros((320, 480), bool)
    domain[y0:y1, x0:x1] = True
    return domain


def rgb_diagnostics(rgb, target, domain):
    residual = rgb-target
    blue = residual[2]-.5*(residual[0]+residual[1])
    return dict(rgb_mae=stats(abs(residual).mean(0), domain),
                absolute_blue_error=stats(abs(blue), domain),
                signed_blue_error=stats(blue, domain),
                positive_blue_error_gt005=int((blue[domain] > .05).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro/campaign"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    observations = args.campaign/"scene008_instances_20261105_r1/observations/frames"
    fixed_path = args.campaign/"scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz"
    receipts = {name: read(root/name/"receipt.json") for name in ("current", "v44")}
    comparison = read(root/"comparison.json")
    with np.load(fixed_path, allow_pickle=False) as fixed:
        np.testing.assert_array_equal(fixed["raw_ids"], np.arange(89, 138))
        assert fixed["layer_names"].tolist()[3] == "ground_beneath_a37_proxy"
        ground = fixed["depth_m"][:, 3] > 0
    result = dict(schema="v44.cpu_saved_float_independent_review.v1", run_dir=str(root),
                  mode=receipts["current"]["mode"], raws=receipts["current"]["raws"],
                  source_sha256={name: sha(root/name/"receipt.json") for name in receipts},
                  comparison_sha256=sha(root/"comparison.json"),
                  no_GPU_import_or_render=True, no_source_or_metric_mutation=True,
                  new_training_updates=0, new_calibration_solves=0,
                  checks=[], per_frame=[], pooled=[], blue_support_diagnostics=[], silver_fixed_ROI=[],
                  observed_road_blue_diagnostics=[], observed_road_blue_pooled=[],
                  posthoc_hood_diagnostics=[],
                  limitations=["HELD9 is internal development, not independent native generalization.",
                               "Original empty blue road domain remains empty; supplemental metrics do not replace it.",
                               "Supplemental observed-background targets are original RGB, not hidden deleted-object truth.",
                               "Common asset exclusion uses the same intersection on both endpoints; its denominator is prediction-conditioned.",
                               "Silver brightness and current V44 layer composition do not measure hidden-background accuracy or prove a net-change cause."])

    def check(name, ok, **detail):
        result["checks"].append(dict(name=name, passed=bool(ok), **detail))

    check("raw_ids_pair_equal", receipts["current"]["raws"] == receipts["v44"]["raws"])
    check("persistent_SF_pair_equal", receipts["current"]["persistent_SF_hash"] == receipts["v44"]["persistent_SF_hash"])
    calculated = {}
    for before, after in zip(receipts["current"]["frames"], receipts["v44"]["frames"]):
        raw = before["raw"]
        check("paired_raw_and_actual_camera_pose", raw == after["raw"]
              and before["actual_camera_pose"] == after["actual_camera_pose"]
              and before["camera_pose_identity"] == after["camera_pose_identity"], raw=raw)
        obs = load(observations/f"{raw:03}.npz")
        gt = normalized_rgb(obs["rgb"])
        proxy = normalized_rgb(obs["road_edit_rgb"])
        a6 = obs["visible"][0].astype(bool).squeeze()
        a37 = obs["visible"][1].astype(bool).squeeze()
        original_background = ~((obs["actor_labels"] > 0) | obs["visible_reviewed"].astype(bool).any(0))
        for state in ("keep", "delete"):
            pair = {name: load(root/name/f"{raw:03}"/(state+"_float.npz")) for name in receipts}
            road = (obs["road_edit"].astype(bool).squeeze() & ground[raw-89]
                    if state == "delete" else obs["road"].astype(bool).squeeze())
            domains = dict(a6_RGB=a6, a6_depth_m=obs["depth_valid"][0].astype(bool).squeeze(),
                           road_RGB=road, protected_RGB=obs["protect"].astype(bool).squeeze() & ~a6 & ~a37)
            if state == "keep":
                domains["a37_RGB"] = a37
            frame_stats = {}
            for name, receipt in (("current", before), ("v44", after)):
                saved = pair[name]
                rgb = saved["rgb"]
                error = abs(rgb-gt).mean(0)
                errors = dict(a6_RGB=error, a6_depth_m=abs(saved["depth_m"]-obs["depth_m"][0].squeeze()),
                              road_RGB=abs(rgb-(proxy if state == "delete" else gt)).mean(0),
                              protected_RGB=error, a37_RGB=error)
                frame_stats[name] = {key: stats(errors[key], mask) for key, mask in domains.items()}
                for key, calculated_value in frame_stats[name].items():
                    declared = receipt["states"][state][key]
                    check("recomputed_float_error_sum_and_count", declared["pixels"] == calculated_value["pixels"]
                          and abs(declared["sum"]-calculated_value["sum"]) < 1e-8,
                          raw=raw, state=state, variant=name, metric=key,
                          sum_error=declared["sum"]-calculated_value["sum"])
                check("finite_saved_float", all(np.isfinite(saved[key]).all() for key in saved), raw=raw, state=state, variant=name)
                conservation = float(abs(saved["group_alpha_T"].sum(0)-saved["alpha"]).max())
                check("independent_alpha_conservation", conservation < 2e-5, raw=raw, state=state,
                      variant=name, maxabs=conservation)
                png = root/receipt["states"][state]["PNG"]["path"]
                check("saved_PNG_hash", sha(png) == receipt["states"][state]["PNG"]["sha256"], raw=raw, state=state, variant=name)
            calculated[raw, state] = frame_stats
            fixed_observed_road = obs["road"].astype(bool).squeeze() & original_background
            common_observed_road = (fixed_observed_road & (pair["current"]["group_alpha_T"][0] <= .01)
                                    & (pair["v44"]["group_alpha_T"][0] <= .01))
            result["observed_road_blue_diagnostics"].append(dict(raw=raw, state=state,
                split="HELD9_internal_development" if raw in HELD else "BUILD40",
                target="original RGB on originally visible observed road, original actors/reviewed excluded; no hidden deletion GT",
                fixed_observed_road=dict(pixels=int(fixed_observed_road.sum()),
                    mask_sha256=hashlib.sha256(fixed_observed_road.tobytes()).hexdigest(),
                    **{name: rgb_diagnostics(saved["rgb"], gt, fixed_observed_road) for name, saved in pair.items()}),
                common_C2_excluded_observed_road=dict(pixels=int(common_observed_road.sum()),
                    mask="fixed observed road AND both current/V44 C2 alphaT<=.01; prediction-conditioned common intersection",
                    mask_sha256=hashlib.sha256(common_observed_road.tobytes()).hexdigest(),
                    **{name: rgb_diagnostics(saved["rgb"], gt, common_observed_road) for name, saved in pair.items()})))
            changes = {key: dict(pixels=frame_stats["current"][key]["pixels"],
                                before=frame_stats["current"][key]["mean"], after=frame_stats["v44"][key]["mean"],
                                delta=frame_stats["v44"][key]["mean"]-frame_stats["current"][key]["mean"])
                       for key in domains}
            result["per_frame"].append(dict(raw=raw, state=state, changes=changes,
                alpha_a6={name: dict(mean=receipt["states"][state]["a6_alpha"]["mean"],
                    below09=receipt["states"][state]["a6_alpha_below09"],
                    below05=receipt["states"][state]["a6_alpha_below05"])
                    for name, receipt in (("current", before), ("v44", after))},
                group_alpha_means_on_a6={name: saved["group_alpha_T"][:, a6].mean(1).tolist()
                                         for name, saved in pair.items()}))
            if raw in BLUE:
                roi = rectangle(BLUE[raw])
                original_domain = roi & obs["road_edit"].astype(bool).squeeze() & ground[raw-89] & ~a6 & ~a37
                fixed_background = roi & original_background
                common_background = fixed_background & (pair["current"]["group_alpha_T"][0] <= .01) & (pair["v44"]["group_alpha_T"][0] <= .01)
                diagnostic = dict(raw=raw, state=state, xyxy=BLUE[raw], rectangle_pixels=int(roi.sum()),
                    original_blue_road_domain_preserved=dict(pixels=int(original_domain.sum()),
                        declared_current=before["states"][state]["blue_fixed_ROI"],
                        declared_v44=after["states"][state]["blue_fixed_ROI"]),
                    support_cascade=dict(road_edit=int((roi & obs["road_edit"]).sum()),
                        ground=int((roi & ground[raw-89]).sum()),
                        road_edit_and_ground=int((roi & obs["road_edit"] & ground[raw-89]).sum()),
                        a6=int((roi & a6).sum()), a37=int((roi & a37).sum())),
                    target="original observed RGB; actor/observed-reviewed pixels excluded",
                    fixed_observed_background=dict(pixels=int(fixed_background.sum()),
                        mask_sha256=hashlib.sha256(fixed_background.tobytes()).hexdigest(),
                        **{name: rgb_diagnostics(saved["rgb"], gt, fixed_background) for name, saved in pair.items()}),
                    legacy_compare_principle_common_C2_exclusion=dict(pixels=int(common_background.sum()),
                        mask_sha256=hashlib.sha256(common_background.tobytes()).hexdigest(),
                        mask="fixed original observed background AND current C2 alphaT<=.01 AND V44 C2 alphaT<=.01",
                        old_source="finalize/evaluate_outputs.py float_roi/compare_roi; same principle with current C2 endpoints, new explicit denominator",
                        **{name: rgb_diagnostics(saved["rgb"], gt, common_background) for name, saved in pair.items()}))
                result["blue_support_diagnostics"].append(diagnostic)
            if raw in SILVER:
                item = dict(raw=raw, state=state, current=before["states"][state]["silver_fixed_ROI"],
                            v44=after["states"][state]["silver_fixed_ROI"])
                if state == "delete":
                    components = load(root/"v44"/f"{raw:03}"/"delete_components.npz")
                    reconstruction = components["group_rgb_T"].sum(0)+components["env_rgb_T"]+components["constant_background_rgb_T"]
                    maxabs = float(abs(reconstruction-pair["v44"]["rgb"]).max())
                    check("independent_same_forward_RGB_reconstruction", maxabs < 5e-5, raw=raw, maxabs=maxabs)
                    roi = rectangle(SILVER[raw])
                    item["independent_RGB_reconstruction_maxabs"] = maxabs
                    item["v44_group_brightness_means"] = dict(zip(("asset", "a37", "road", "native_static"),
                        components["group_rgb_T"][:, :, roi].mean((1, 2)).tolist()))
                    item["v44_environment_brightness_mean"] = float(components["env_rgb_T"][:, roi].mean())
                    item["v44_constant_background_brightness_mean"] = float(components["constant_background_rgb_T"][:, roi].mean())
                result["silver_fixed_ROI"].append(item)
            if raw == 105:
                hood_box = (18, 237, 91, 262)
                hood = rectangle(hood_box)
                hood_item = dict(raw=raw, state=state, xyxy=hood_box, pixels=int(hood.sum()),
                    selection="posthoc internal-development rectangle requested after visual inspection; not a formal acceptance gate",
                    target="original observed RGB; rectangle can include car and non-car pixels",
                    original_RGB_mean=gt[:, hood].mean(1).tolist(),
                    **{name: dict(RGB_mean=saved["rgb"][:, hood].mean(1).tolist(),
                                  group_alpha_mean=saved["group_alpha_T"][:, hood].mean(1).tolist(),
                                  **rgb_diagnostics(saved["rgb"], gt, hood)) for name, saved in pair.items()})
                if state == "delete":
                    components = load(root/"v44"/"105"/"delete_components.npz")
                    hood_item["v44_same_forward_RGB_means"] = dict(zip(("asset", "a37", "road", "native_static"),
                        components["group_rgb_T"][:, :, hood].mean(2).tolist()))
                    hood_item["v44_environment_RGB_mean"] = components["env_rgb_T"][:, hood].mean(1).tolist()
                    hood_item["v44_constant_background_RGB_mean"] = components["constant_background_rgb_T"][:, hood].mean(1).tolist()
                    reconstruction = components["group_rgb_T"].sum(0)+components["env_rgb_T"]+components["constant_background_rgb_T"]
                    hood_item["RGB_reconstruction_maxabs"] = float(abs(reconstruction-pair["v44"]["rgb"]).max())
                result["posthoc_hood_diagnostics"].append(hood_item)
    for row in comparison["pixel_weighted_pooled"]:
        state, split, metric = row["state"], row["split"], row["metric"]
        selected = [raw for raw in result["raws"] if split == "ALL" or
                    (split == "HELD9_internal_development" and raw in HELD) or
                    (split == "BUILD40" and raw not in HELD)]
        count = sum(calculated[raw, state]["current"][metric]["pixels"] for raw in selected)
        sums = {name: sum(calculated[raw, state][name][metric]["sum"] for raw in selected)
                for name in receipts}
        independent = dict(state=state, split=split, metric=metric, frames=len(selected), pixels=count,
                           before=sums["current"]/count if count else None,
                           after=sums["v44"]/count if count else None,
                           delta=(sums["v44"]-sums["current"])/count if count else None)
        check("independent_pixel_weighted_pooled", row["pixels"] == count and row["frames"] == len(selected)
              and all(row[key] == independent[key] for key in ("before", "after", "delta")),
              state=state, split=split, metric=metric)
        result["pooled"].append(independent)
    runtime = receipts["v44"]["runtime_receipt"]
    for state in ("keep", "delete"):
        for split in ("ALL", "BUILD40", "HELD9_internal_development"):
            selected = [row for row in result["observed_road_blue_diagnostics"]
                        if row["state"] == state and (split == "ALL" or row["split"] == split)]
            if not selected:
                continue
            for domain in ("fixed_observed_road", "common_C2_excluded_observed_road"):
                count = sum(row[domain]["pixels"] for row in selected)
                pooled_row = dict(state=state, split=split, domain=domain, frames=len(selected), pixels=count)
                for variant in receipts:
                    pooled_row[variant] = {}
                    for metric in ("rgb_mae", "absolute_blue_error", "signed_blue_error"):
                        total = sum(row[domain][variant][metric]["sum"] for row in selected)
                        pooled_row[variant][metric] = dict(sum=total, mean=total/count if count else None)
                    pooled_row[variant]["positive_blue_error_gt005"] = sum(row[domain][variant]["positive_blue_error_gt005"] for row in selected)
                result["observed_road_blue_pooled"].append(pooled_row)
    result["regression_extrema"] = {}
    result["coverage_summary"] = {}
    result["observed_road_blue_regressions"] = []
    for state in ("keep", "delete"):
        state_rows = [row for row in result["per_frame"] if row["state"] == state]
        result["regression_extrema"][state] = {}
        for metric in ("a6_RGB", "a6_depth_m", "road_RGB", "protected_RGB") + (("a37_RGB",) if state == "keep" else ()):
            changes = [{"raw": row["raw"], **row["changes"][metric]} for row in state_rows]
            result["regression_extrema"][state][metric] = dict(
                worsened_frames=sum(row["delta"] > 0 for row in changes),
                worst_regression=max(changes, key=lambda row: row["delta"]),
                best_improvement=min(changes, key=lambda row: row["delta"]))
        result["regression_extrema"][state]["largest_asset_alphaT_drop"] = max(
            [dict(raw=row["raw"], before=row["alpha_a6"]["current"]["mean"], after=row["alpha_a6"]["v44"]["mean"],
                  drop=row["alpha_a6"]["current"]["mean"]-row["alpha_a6"]["v44"]["mean"],
                  current_group_alpha=row["group_alpha_means_on_a6"]["current"],
                  v44_group_alpha=row["group_alpha_means_on_a6"]["v44"],
                  current_below09=row["alpha_a6"]["current"]["below09"], v44_below09=row["alpha_a6"]["v44"]["below09"],
                  current_below05=row["alpha_a6"]["current"]["below05"], v44_below05=row["alpha_a6"]["v44"]["below05"])
             for row in state_rows], key=lambda row: row["drop"])
        count = sum(row["changes"]["a6_RGB"]["pixels"] for row in state_rows)
        result["coverage_summary"][state] = dict(pixels=count,
            **{name: dict(mean_asset_alphaT=sum(row["alpha_a6"][name]["mean"]*row["changes"]["a6_RGB"]["pixels"] for row in state_rows)/count,
                          below09=sum(row["alpha_a6"][name]["below09"] for row in state_rows),
                          below05=sum(row["alpha_a6"][name]["below05"] for row in state_rows)) for name in receipts},
            alphaT_below09_increased_frames=[row["raw"] for row in state_rows if row["alpha_a6"]["v44"]["below09"] > row["alpha_a6"]["current"]["below09"]],
            alphaT_below05_increased_frames=[row["raw"] for row in state_rows if row["alpha_a6"]["v44"]["below05"] > row["alpha_a6"]["current"]["below05"]],
            largest_static_alphaT_increase=max([dict(raw=row["raw"],
                current=row["group_alpha_means_on_a6"]["current"][3], v44=row["group_alpha_means_on_a6"]["v44"][3],
                delta=row["group_alpha_means_on_a6"]["v44"][3]-row["group_alpha_means_on_a6"]["current"][3])
                for row in state_rows], key=lambda row: row["delta"]))
        for domain in ("fixed_observed_road", "common_C2_excluded_observed_road"):
            road_rows = [row for row in result["observed_road_blue_diagnostics"] if row["state"] == state and row[domain]["pixels"]]
            changes = [dict(raw=row["raw"], pixels=row[domain]["pixels"],
                current=row[domain]["current"]["absolute_blue_error"]["mean"], v44=row[domain]["v44"]["absolute_blue_error"]["mean"],
                delta=row[domain]["v44"]["absolute_blue_error"]["mean"]-row[domain]["current"]["absolute_blue_error"]["mean"],
                current_count_gt005=row[domain]["current"]["positive_blue_error_gt005"],
                v44_count_gt005=row[domain]["v44"]["positive_blue_error_gt005"],
                delta_count_gt005=row[domain]["v44"]["positive_blue_error_gt005"]-row[domain]["current"]["positive_blue_error_gt005"])
                for row in road_rows]
            result["observed_road_blue_regressions"].append(dict(state=state, domain=domain,
                absolute_blue_mean_worsened_frames=[row["raw"] for row in changes if row["delta"] > 0],
                positive_blue_count_increased_frames=[row["raw"] for row in changes if row["delta_count_gt005"] > 0],
                worst_mean_delta=max(changes,key=lambda row: row["delta"]),
                worst_count_delta=max(changes,key=lambda row: row["delta_count_gt005"])))
    result["runtime_receipt_summary"] = {key: runtime[key] for key in
        ("status", "gradient_updates", "colour_fit_steps", "calibration_solves",
         "asset_all_named_tensors_bitwise_hash_unchanged", "frozen_native_road_gates_environment_unchanged",
         "persistent_SF_unchanged", "frozen_transfer_is_not_C2_or_LOD_recalibration")}
    result["protected_outside_changes"] = [dict(raw=row["raw"], state=row["state"], **row["protected_outside_change"])
                                           for row in comparison["comparisons"]]
    result["check_count"] = len(result["checks"])
    result["all_checks_passed"] = all(row["passed"] for row in result["checks"])
    result["failures"] = [row for row in result["checks"] if not row["passed"]]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
