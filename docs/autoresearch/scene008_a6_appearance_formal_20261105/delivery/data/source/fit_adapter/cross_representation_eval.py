"""Fixed49 native paired preservation of the formal projective representation.

The 50322-row chart differs from 18432-row baselines in geometry and kernels.
We validate the persistent chart retirement/source-id contract, then measure
actual RGB/depth/occlusion changes without geometry-equality claims or tuning.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from adapter import APPEARANCE, bank, same_bytes, sha, write_json

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
OLD = CAM/"scene008_a6_asset_controls_20261105"
RUN = CAM/"scene008_a6_appearance_formal_20261105"
HELD = (89, 95, 101, 105, 113, 121, 125, 130, 137)


def validate_representation(args, comparators):
    root = RUN/"output_adapter/projective_surface"
    prepare_path = root/"prepare_receipt.json"
    chart_path = root/"reference_chart.npz"
    prototype_path = root/"projective_surface_asset.npz"
    prepare = json.loads(prepare_path.read_text())
    chart, prototype, target = bank(chart_path), bank(prototype_path), bank(args.asset)
    source_path = Path(prepare["source_asset"])
    source = bank(source_path)
    assert sha(source_path) == prepare["source_asset_sha256"]
    assert sha(prototype_path) == prepare["export"]["file_sha256"]
    assert str(source["geometry_hash"]) == prepare["source_geometry_hash"]
    assert str(target["geometry_hash"]) == str(prototype["geometry_hash"]) == prepare["geometry_hash"]
    assert set(target) == set(prototype)
    for key in set(target)-APPEARANCE:
        assert same_bytes(target[key], prototype[key]), "Formal/prototype frozen field changed: "+key
    retired = chart["source_retired"]
    retained = chart["source_retained"]
    assert retired.dtype == bool and retired.shape == (len(source["xyz"]),)
    assert np.array_equal(retired, chart["source_native_reference_contribution"] > 0)
    assert np.array_equal(retained, np.flatnonzero(~retired))
    assert len(source["xyz"]) == prepare["source_rows"] == 18432
    assert int(retired.sum()) == prepare["retired_source_rows"]
    assert len(retained) == prepare["retained_source_rows"]
    keep = target["representation_kind"] == 0
    added = target["representation_kind"] == 1
    assert np.all(keep | added)
    assert int(keep.sum()) == len(retained) and int(added.sum()) == prepare["chart_rows"]
    assert len(target["xyz"]) == len(retained)+len(chart["pixels"]) == 50322
    assert np.array_equal(target["resample_source_row"][keep], retained)
    assert np.array_equal(target["resample_source_row"][added], chart["nearest_source_row"])
    assert np.all((target["resample_source_row"] >= 0) & (target["resample_source_row"] < len(source["xyz"])))
    assert np.array_equal(target["resample_source_parent"], source["sampling_parent_index"][target["resample_source_row"]])
    for key in ("xyz", "normal", "sigma", "opacity", "thickness", "source_index", "sampling_parent_index", "sampling_child_index"):
        assert same_bytes(target[key][keep], source[key][retained]), "Retained source geometry changed: "+key
    assert np.array_equal(target["xyz"][added], chart["local_xyz"])
    assert np.array_equal(target["sigma"][added], chart["sigma_local_m"])
    assert np.all(target["opacity"][added] == np.float32(prepare["opacity"]))
    assert prepare["reference_raw"] == 106
    assert np.all(target["resample_reference_raw"][added] == prepare["reference_raw"])
    assert np.array_equal(target["resample_reference_pixel_xy"][added], chart["pixel_xy"])
    for out_key, chart_key in (("resample_rgb_real","color_real"),("resample_rgb_generated","color_generated"),("resample_rgb_prior_fallback","color_prior_fallback")):
        assert np.array_equal(target[out_key][added], chart[chart_key])
    common = dict(target_rows=len(target["xyz"]), source_rows=len(source["xyz"]),
        retained_source_rows=len(retained), retired_source_rows=int(retired.sum()), chart_rows=int(added.sum()),
        target_geometry_hash=str(target["geometry_hash"]), formal_vs_prototype_geometry_opacity_provenance_bitwise_exact=True,
        source_ids_and_native_positive_contribution_retirement_contract_verified=True,
        source_asset_sha256=sha(source_path), prototype_sha256=sha(prototype_path),
        prepare_receipt_sha256=sha(prepare_path), reference_chart_sha256=sha(chart_path),
        different_means_kernels_and_primitive_count=True, original_PVD_weights_updated=False,
        generated_chart_depth_is_prediction_not_measurement=True)
    return {label:dict(common, comparator_rows=len(bank(path)["xyz"]), comparator_geometry_hash=str(bank(path)["geometry_hash"]),
                       geometry_equality_claimed=False) for label,path in comparators.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--compare", action="append", help="label=/absolute/asset.npz; repeat for multiple thin baselines")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", default=str(OLD/"source"))
    args = parser.parse_args()
    comparisons = args.compare or [f"thin_m2={OLD/'m2/complete_a6.npz'}",
                                  f"sphere_strong_v2={RUN/'output_adapter/sphere_strong_v2/complete_a6.npz'}"]
    comparators = {}
    for item in comparisons:
        label, path = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_]+", label) or label in comparators:
            raise ValueError("Use distinct plain comparator labels")
        comparators[label] = Path(path).resolve()
    dest = Path(args.output).resolve()
    if dest.exists():
        if any(dest.iterdir()):
            raise FileExistsError("Refusing to overwrite an evaluation containing results")
    else:
        dest.mkdir(parents=True, exist_ok=False)
    geometry_checks = validate_representation(args, comparators)

    sys.path.insert(0, str(Path(args.source_root).resolve()))
    import torch
    from sh_asset import SHAsset
    from scene_runtime import Runtime, raster, tagged
    from metrics import (evaluate_frame, evaluate_pair, expected_depth, fixed_masks,
                         metric, observation)
    from evaluate_control import aggregate

    rt = Runtime(); asset = SHAsset.from_geometry(args.asset).cuda()
    baselines = {label:SHAsset.from_geometry(path).cuda() for label, path in comparators.items()}
    records = {label:[] for label in comparators}; start = time.time()

    def solo(data, material):
        rows = tagged(material.rows(data["pose"], data["cam"], rt.pipe, 16), "asset")
        out = raster(data["cam"], rt.pipe, 3, rows, rt.n.bg.new_zeros(3))
        out["component_alpha_T"] = out["extra"][:4]
        return out

    def scalar(value):
        value = value.detach().cpu().numpy()
        return value[0] if value.ndim == 3 else value

    def extras(data, current, before, state):
        source, masks = observation(data), fixed_masks(data)
        z, z0 = expected_depth(current, rt.scale), expected_depth(before, rt.scale)
        a, a0 = scalar(current["alpha"]), scalar(before["alpha"])
        groups, groups0 = current["component_alpha_T"], before["component_alpha_T"]
        fixed = dict(a37=masks["a37_visible"], road=masks["road_visible"], outside=masks["outside"],
                     observed_non_a6=masks["observed_non_a6"])
        result = {}
        for label, mask in fixed.items():
            real_depth = source["scene_depth_m"]
            measured_mask = mask & np.isfinite(real_depth) & (real_depth > 0)
            row = dict(alpha_parent_change=metric(abs(a-a0), mask),
                       expected_depth_parent_change_m=metric(abs(z-z0), mask),
                       measured_expected_depth_error_m=metric(abs(z-real_depth), measured_mask),
                       baseline_measured_expected_depth_error_m=metric(abs(z0-real_depth), measured_mask),
                       fixed_measured_depth_pixels=int(measured_mask.sum()))
            for index, name in enumerate(("asset", "a37", "road", "legacy_static")):
                row[name+"_alphaT_parent_change"] = metric(abs(groups[index]-groups0[index]), mask)
            result[label] = row
        if state == "keep":
            mask = masks["a37_depth"]; real_depth = source["depth_m"][1]
            result["a37_associated_LiDAR"] = dict(
                measured_depth_error_m=metric(abs(z-real_depth), mask),
                baseline_measured_depth_error_m=metric(abs(z0-real_depth), mask),
                parent_depth_change_m=metric(abs(z-z0), mask))
        else:
            real_depth = source["road_edit_depth_m"]
            mask = masks["road_reobserved"] & np.isfinite(real_depth) & (real_depth > 0)
            result["road_reobserved_LiDAR"] = dict(
                measured_depth_error_m=metric(abs(z-real_depth), mask),
                baseline_measured_depth_error_m=metric(abs(z0-real_depth), mask),
                parent_depth_change_m=metric(abs(z-z0), mask))
        return result

    with torch.no_grad():
        for raw in range(89, 138):
            data = rt.raw(raw)
            new_solo = solo(data, asset)
            keep, deleted = rt.render_scene(raw, asset), rt.render_scene(raw, asset, delete=True)
            assert keep["meta"]["native_a6_rows_input"] == deleted["meta"]["native_a6_rows_input"] == 0
            for label, baseline in baselines.items():
                old_solo = solo(data, baseline)
                old_keep = rt.render_scene(raw, baseline)
                old_deleted = rt.render_scene(raw, baseline, delete=True)
                row = dict(raw=raw, split="HELD" if raw in HELD else "BUILD",
                    solo=evaluate_frame(data, new_solo, old_solo, rt.scale, "solo"),
                    scene=evaluate_pair(data, keep, deleted, old_keep, old_deleted, rt.scale),
                    keep_fixed_depth_preservation=extras(data, keep, old_keep, "keep"),
                    delete_fixed_depth_preservation=extras(data, deleted, old_deleted, "delete"),
                    full_image_alpha_maxabs=float((keep["alpha"]-old_keep["alpha"]).abs().max()),
                    full_image_depth_sum_maxabs=float((keep["depth_sum"]-old_keep["depth_sum"]).abs().max()))
                records[label].append(row)
                del old_solo, old_keep, old_deleted
            print(json.dumps(dict(stage="fixed_cross_representation_evaluation", raw=raw)), flush=True)
            del data, new_solo, keep, deleted
            rt._data.clear(); torch.cuda.empty_cache()
    assert rt.modelunchanged() and asset.assert_geometry_unchanged()
    assert all(value.assert_geometry_unchanged() for value in baselines.values())
    write_json(dest/"metrics.json", records)
    summaries = {label:dict(all49=aggregate(rows), BUILD=aggregate([r for r in rows if r["split"] == "BUILD"]),
        HELD=aggregate([r for r in rows if r["split"] == "HELD"])) for label, rows in records.items()}
    write_json(dest/"summary.json", summaries)
    write_json(dest/"receipt.json", dict(status="PASS_FIXED_CROSS_REPRESENTATION_PRESERVATION_AUDIT",
        source_asset=str(Path(args.asset).resolve()), source_asset_sha256=sha(args.asset),
        comparators={label:dict(path=str(path), sha256=sha(path)) for label, path in comparators.items()},
        source_sha256=sha(__file__), frames=49, geometry_checks=geometry_checks,
        alpha_depth_equality_required=False, one_prespecified_representation_only=True,
        primitive_count_and_geometry_differ_from_baselines=True,
        original_formal_a6_evaluation_had_no_non_a6_preservation_metrics=True,
        masks_and_denominators="Original observation bank only; no predicted-alpha mask shrinking",
        scene_depth_reference="Original observation-bank scene_depth_m, finite positive fixed pixels",
        a37_depth_reference="Original associated actor37 depth_m/depth_valid in keep state",
        deleted_road_depth_reference="Original reobserved road_edit_depth_m on fixed road_edit pixels",
        frozen_background_a37_road_environment_parameters_unchanged=True,
        occlusion_effects_measured_not_assumed_unchanged=True,
        generated_teacher_is_not_hidden_RGB_or_depth_truth=True,
        HELD_only_final_evaluation=True, model_updates=0, seconds=time.time()-start))
    print(json.dumps(dict(status="PASS", output=str(dest), frames=49)), flush=True)


if __name__ == "__main__":
    main()
