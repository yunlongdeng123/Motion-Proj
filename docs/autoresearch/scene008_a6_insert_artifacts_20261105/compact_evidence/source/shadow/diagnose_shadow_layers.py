"""Same-camera native RGB/transmittance attribution of inherited dark road patches.

Prepared locally; run on the existing host only after parent coordination.
No model is trained and no parent tensors, source files or checkpoints change.
Dark pixels and shadow_proxy are diagnostic domains, not physical-shadow labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

GROUPS = ("asset", "a37", "road", "legacy_static")
CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
SOURCE = CAM/"scene008_a6_asset_controls_20261105/source"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def luma_index(rgb):
    # A reproducible index in renderer RGB units, not calibrated luminance.
    return np.einsum("c,chw->hw", [.2126, .7152, .0722], np.asarray(rgb, np.float64))


def pixel_stats(rgb, mask):
    values = np.asarray(rgb, np.float64)[:, mask]
    return dict(pixels=int(mask.sum()), mean_rgb=values.mean(1).tolist() if values.size else None,
                mean_luma_index=float(luma_index(rgb)[mask].mean()) if mask.any() else None)


def delta_stats(after, before, mask):
    delta = np.asarray(after, np.float64)-np.asarray(before, np.float64)
    values = np.abs(delta[:, mask]).mean(0)
    return dict(pixels=int(mask.sum()), RGB_MAE=float(values.mean()) if len(values) else None,
                RGB_max=float(values.max()) if len(values) else None,
                above_002=int((values > .02).sum()), above_005=int((values > .05).sum()),
                signed_luma_change=float(luma_index(delta)[mask].mean()) if mask.any() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE))
    parser.add_argument("--asset", help="Optional current complete asset; omit for the older native-a6 baseline")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--raws", default="105,113,130")
    parser.add_argument("--roi-json", help="Optional {raw: [x0,y0,x1,y1]} in full480x320 coordinates")
    parser.add_argument("--dark-threshold", type=float, default=.18)
    parser.add_argument("--static-dark-threshold", type=float, default=.05)
    parser.add_argument("--neutral-rgb", type=float, default=.4)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    import torch
    from scene_runtime import Runtime, raster, prepare_rasterization, select_rows
    from sh_asset import SHAsset
    from utils.sh_utils import eval_sh
    import scene_runtime

    torch.set_num_threads(1)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    roi = json.loads(Path(args.roi_json).read_text()) if args.roi_json else {}
    rt = Runtime(fine_road=True)
    asset = SHAsset.from_geometry(args.asset).cuda() if args.asset else None
    asset_sha = sha(args.asset) if args.asset else None
    start = time.time(); records = []

    def cpu(value):
        return value.detach().cpu().numpy()

    def save_image(path, rgb):
        Image.fromarray(np.uint8(np.clip(rgb.transpose(1, 2, 0), 0, 1)*255+.5)).save(path)

    with torch.no_grad():
        for raw in [int(value) for value in args.raws.split(",")]:
            d = rt.raw(raw); camera = d["cam"]
            scene = rt.scene(raw, asset, delete=True, baseline=asset is None)
            original_delete = rt.render_rows(raw, scene)
            a37_index = GROUPS.index("a37")
            packed_before = int((scene["rows"]["features"][:, 2+a37_index] > .5).sum())
            without_a37 = select_rows(scene["rows"], scene["rows"]["features"][:, 2+a37_index] < .5)
            scene = dict(scene, rows=without_a37, meta=dict(scene["meta"], native_a37_packed_rows=0))
            rows = scene["rows"]
            owners = {name:rows["features"][:, 2+index] > .5 for index, name in enumerate(GROUPS)}
            active_counts = {name:int((mask & rows["mask"]).sum()) for name, mask in owners.items()}
            assert active_counts["a37"] == 0, "Deletion has active actor37 rows"
            full = rt.render_rows(raw, scene)
            physical_delete_error = float((full["rgb"]-original_delete["rgb"]).abs().max())
            assert physical_delete_error < 5e-6, physical_delete_error
            assert not bool(owners["a37"].any()) and scene["meta"]["native_a37_packed_rows"] == 0
            original = cpu(full["rgb"])
            alpha = cpu(full["alpha"])
            transmittance = 1-alpha
            env = transmittance*cpu(d["env"])
            background = transmittance*cpu(scene["bg"])[:, None, None]
            if rows["colors_precomp"] is not None:
                point_rgb = rows["colors_precomp"].clone()
            else:
                directions = torch.nn.functional.normalize(rows["means3D"]-camera.camera_center, dim=1)
                point_rgb = (eval_sh(rt.n.static.active_sh_degree,
                                    rows["shs"].transpose(1, 2), directions)+.5).clamp_min(0)

            def draw_with_rgb(colors, mask=None, opacities=None):
                variant = dict(rows, shs=None, colors_precomp=colors,
                               mask=rows["mask"] if mask is None else mask,
                               opacities=rows["opacities"] if opacities is None else opacities)
                out = raster(camera, rt.pipe, 0, variant, scene["bg"])
                return out, cpu(out["rgb"]+(1-out["alpha"])*d["env"])

            converted, converted_rgb = draw_with_rgb(point_rgb)
            conversion_error = float(abs(converted_rgb-original).max())
            assert conversion_error < 3e-5, conversion_error
            components = {}
            for name, owner in owners.items():
                isolated = dict(rows, shs=None, colors_precomp=point_rgb*owner[:, None])
                out = raster(camera, rt.pipe, 0, isolated, scene["bg"].new_zeros(3))
                assert torch.equal(out["alpha"], full["alpha"])
                assert torch.equal(out["depth_sum"], full["depth_sum"])
                components[name] = cpu(out["rgb"])
            reconstructed = sum(components.values())+env+background
            additivity_error = float(abs(reconstructed-original).max())
            assert additivity_error < 5e-5, additivity_error

            images = dict(full_fine=original, no_environment=original-env,
                          neutral_environment=original-env+transmittance*args.neutral_rgb)
            # Colour-only controls keep the entire joint opacity/depth order.
            neutral = point_rgb.new_full(point_rgb.shape, args.neutral_rgb)
            for group in ("road", "legacy_static", "asset"):
                colors = point_rgb.clone(); colors[owners[group]] = neutral[owners[group]]
                out, image = draw_with_rgb(colors)
                assert torch.equal(out["alpha"], full["alpha"])
                assert torch.equal(out["depth_sum"], full["depth_sum"])
                images[group+"_neutral_RGB_same_opacity"] = image
            road_base = rt.exp.repair.road.colors().detach()
            assert len(road_base) == int(owners["road"].sum())
            colors = point_rgb.clone(); colors[owners["road"]] = road_base
            out, images["road_fine_field_off"] = draw_with_rgb(colors)
            assert torch.equal(out["alpha"], full["alpha"])
            assert torch.equal(out["depth_sum"], full["depth_sum"])

            # Deletion controls are causal interventions; T/depth may change.
            for group in ("road", "legacy_static", "asset"):
                mask = rows["mask"] & ~owners[group]
                _, images[group+"_disabled"] = draw_with_rgb(point_rgb, mask=mask)
            dark_static = owners["legacy_static"] & (point_rgb.max(dim=1).values < args.static_dark_threshold)
            _, images["dark_static_disabled"] = draw_with_rgb(point_rgb, mask=rows["mask"] & ~dark_static)
            native_static = prepare_rasterization(camera, rt.n.static, rt.pipe)
            assert torch.equal(rows["means3D"][owners["legacy_static"]], native_static["means3D"])
            opacity = rows["opacities"].clone()
            opacity[owners["legacy_static"]] = native_static["opacities"]
            _, images["static_gates_off"] = draw_with_rgb(point_rgb, opacities=opacity)
            images["native10k_delete_no_road_no_gates"] = cpu(rt.n.render(camera, dynamic=False)["rgb"])

            observed = d["obs"]
            road_domain = observed["road"].astype(bool) | observed["road_edit"].astype(bool)
            fixed_masks = dict(observed_road=observed["road"].astype(bool),
                               reobserved_road_after_delete=observed["road_edit"].astype(bool),
                               baseline_dark_road=road_domain & (luma_index(original) < args.dark_threshold))
            if str(raw) in roi:
                x0, y0, x1, y1 = map(int, roi[str(raw)])
                custom = np.zeros(road_domain.shape, bool); custom[y0:y1, x0:x1] = True
                fixed_masks["fixed_requested_patch"] = custom
            group_alpha = cpu(full["component_alpha_T"])
            descriptions = {}
            for mask_name, mask in fixed_masks.items():
                descriptions[mask_name] = dict(
                    controls={name:dict(pixel_stats(image, mask), delta_from_full=delta_stats(image, original, mask))
                              for name, image in images.items()},
                    exact_joint_T_RGB={name:pixel_stats(image, mask) for name, image in components.items()},
                    environment_additive=pixel_stats(env, mask),
                    constant_background_additive=pixel_stats(background, mask),
                    mean_residual_transmittance=float(transmittance[0, mask].mean()) if mask.any() else None,
                    mean_joint_alphaT={name:float(group_alpha[index, mask].mean()) if mask.any() else None
                                       for index, name in enumerate(GROUPS)})
            folder = output/f"{raw:03d}"; folder.mkdir()
            for name, image in images.items():
                save_image(folder/(name+".png"), image)
            for name, image in components.items():
                save_image(folder/("joint_T_RGB_"+name+".png"), image)
            np.savez_compressed(folder/"exact_attribution.npz", **images,
                **{"joint_T_RGB_"+k:v for k,v in components.items()}, environment_additive=env,
                constant_background_additive=background, alpha=alpha, group_alpha_T=group_alpha,
                residual_transmittance=transmittance, **fixed_masks)
            record = dict(raw=raw, active_group_rows=active_counts, masked_or_absent_a37_produces_no_RGB=True,
                native_a37_packed_rows_before_physical_diagnostic_removal=packed_before,
                native_a37_packed_rows_after=0, source_runtime_meta=scene["meta"],
                physical_vs_original_masked_delete_RGB_maxabs=physical_delete_error,
                native_SH_to_precomputed_RGB_maxabs=conversion_error,
                joint_RGB_additivity_maxabs=additivity_error, static_dark_rows=int((dark_static & rows["mask"]).sum()),
                fixed_domains=descriptions)
            records.append(record)
            write(folder/"receipt.json", record)
            rt._data.clear(); torch.cuda.empty_cache()
            print(json.dumps(dict(stage="dark_road_attribution", raw=raw)), flush=True)
    assert rt.modelunchanged()
    if asset is not None:
        assert asset.assert_geometry_unchanged() and sha(args.asset) == asset_sha
    write(output/"receipt.json", dict(status="COMPLETE_SOURCE_ATTRIBUTION_DIAGNOSTIC",
        physical_shadow_classification=False, no_shadow_layer_assumed=True, rows=records,
        model_updates=0, parent_parameters_unchanged=True, asset_sha256=asset_sha,
        runtime_source_sha256=sha(scene_runtime.__file__), script_sha256=sha(__file__),
        seconds=time.time()-start, decomposition="Native joint alpha*T RGB per group + T*env + T*constant background",
        causal_control_warning="Layer removal changes transmittance/depth order; isolated-on-black renders alone are not attribution",
        counterfactual_warning="Deleted actor pixels have no factual same-camera RGB; road_edit is a reobserved proxy"))


if __name__ == "__main__":
    main()
