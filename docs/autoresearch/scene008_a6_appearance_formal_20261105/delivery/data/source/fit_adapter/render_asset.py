"""Independent old-runtime preview/fixed all49 evaluation for any same-M2 asset.

No colour/geometry/model optimization occurs here. HELD9 is evaluated only;
these outputs must not select or tune generated appearance candidates.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from PIL import Image

from adapter import APPEARANCE, bank, same_bytes, sha, write_json

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
OLD = CAM/"scene008_a6_asset_controls_20261105"
HELD = (89, 95, 101, 105, 113, 121, 125, 130, 137)


def cpu(value):
    return value.detach().cpu().numpy()


def save_rgb(path, value):
    rgb = cpu(value).transpose(1, 2, 0)
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255+.5)).save(path)


def camera_hr(camera):
    result = copy.copy(camera)
    result.image_width = 1920; result.image_height = 1280
    for key in ("fx", "fy", "cx", "cy"):
        setattr(result, key, getattr(camera, key)*4)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--compare", default=str(OLD/"m2/complete_a6.npz"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("preview", "all49"), default="preview")
    parser.add_argument("--preview-raws", default="105,113,130")
    parser.add_argument("--black-only", action="store_true")
    parser.add_argument("--source-root", default=str(OLD/"source"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    import torch
    from sh_asset import SHAsset
    from scene_runtime import Runtime, raster, tagged
    from metrics import evaluate_frame, evaluate_pair
    from evaluate_control import aggregate

    target_arrays, baseline_arrays = bank(args.asset), bank(args.compare)
    if set(target_arrays) != set(baseline_arrays):
        raise ValueError("Asset schema differs from the frozen comparator")
    for key in set(target_arrays)-APPEARANCE:
        if not same_bytes(target_arrays[key], baseline_arrays[key]):
            raise ValueError("Frozen geometry/provenance differs: " + key)
    dest = Path(args.output).resolve(); dest.mkdir(parents=True, exist_ok=False)
    rt = Runtime(); asset = SHAsset.from_geometry(args.asset).cuda()
    base = SHAsset.from_geometry(args.compare).cuda()
    start = time.time(); records = []; preview_records = []
    raw_ids = tuple(range(89, 138)) if args.mode == "all49" else ()

    def solo(data, material, camera=None):
        camera = data["cam"] if camera is None else camera
        rows = tagged(material.rows(data["pose"], camera, rt.pipe, 16), "asset")
        out = raster(camera, rt.pipe, 3, rows, rt.n.bg.new_zeros(3))
        out["component_alpha_T"] = out["extra"][:4]
        return out

    with torch.no_grad():
        for raw in raw_ids:
            data = rt.raw(raw); old, new = solo(data, base), solo(data, asset)
            assert torch.equal(new["alpha"], old["alpha"])
            assert torch.equal(new["depth_sum"], old["depth_sum"])
            assert torch.equal(new["component_alpha_T"], old["component_alpha_T"])
            row = dict(raw=raw, split="HELD" if raw in HELD else "BUILD",
                       solo=evaluate_frame(data, new, old, rt.scale, "solo"))
            if not args.black_only:
                oldkeep = rt.render_scene(raw, base); olddelete = rt.render_scene(raw, base, delete=True)
                keep = rt.render_scene(raw, asset); delete = rt.render_scene(raw, asset, delete=True)
                row["scene"] = evaluate_pair(data, keep, delete, oldkeep, olddelete, rt.scale)
                for current, previous in ((keep, oldkeep), (delete, olddelete)):
                    for key in ("alpha", "depth_sum", "component_alpha_T"):
                        assert torch.equal(current[key], previous[key])
                    assert current["meta"]["native_a6_rows_input"] == 0
                del oldkeep, olddelete, keep, delete
            records.append(row)
            print(json.dumps(dict(stage="fixed_all49_colour_evaluation", raw=raw)), flush=True)
            del data, old, new
            rt._data.clear(); torch.cuda.empty_cache()

        for raw in [int(value) for value in args.preview_raws.split(",") if value]:
            data = rt.raw(raw); camera = camera_hr(data["cam"])
            old, new = solo(data, base, camera), solo(data, asset, camera)
            for key in ("alpha", "depth_sum", "component_alpha_T"):
                assert torch.equal(new[key], old[key])
            save_rgb(dest/f"{raw:03d}_black_1920.png", new["rgb"])
            save_rgb(dest/f"{raw:03d}_compare_black_1920.png", old["rgb"])
            support = (new["alpha"][0] > .01).cpu().numpy()
            yy, xx = np.where(support)
            crop = (max(0, int(xx.min())-16), max(0, int(yy.min())-16),
                    min(1920, int(xx.max())+17), min(1280, int(yy.max())+17))
            Image.open(dest/f"{raw:03d}_black_1920.png").crop(crop).save(dest/f"{raw:03d}_black_crop.png")
            Image.open(dest/f"{raw:03d}_compare_black_1920.png").crop(crop).save(dest/f"{raw:03d}_compare_black_crop.png")
            np.savez_compressed(dest/f"{raw:03d}_black_replay.npz", rgb_srgb=cpu(new["rgb"]).astype(np.float32),
                alpha=cpu(new["alpha"]), camera_z_depth_m=cpu(new["depth_sum"]/new["alpha"].clamp_min(1e-5)/rt.scale),
                crop_xyxy=np.asarray(crop), asset_sha256=np.asarray(sha(args.asset)))
            if not args.black_only:
                # Same frozen scene rows and camera centre/time; only full-image raster sampling changes.
                yy_t, xx_t = torch.meshgrid(torch.arange(1280, device="cuda", dtype=torch.float32),
                                            torch.arange(1920, device="cuda", dtype=torch.float32), indexing="ij")
                direction = torch.stack(((xx_t+.5-camera.cx)/camera.fx,
                                         (yy_t+.5-camera.cy)/camera.fy, torch.ones_like(xx_t)), dim=-1)
                direction = torch.nn.functional.normalize(direction, dim=-1)@camera.c2w[:3, :3].T
                environment = rt.n.env(direction).permute(2, 0, 1).detach()
                for delete in (False, True):
                    scene = rt.scene(raw, asset, delete=delete)
                    out = raster(camera, rt.pipe, rt.n.static.active_sh_degree, scene["rows"], scene["bg"])
                    rgb = out["rgb"]+(1-out["alpha"])*environment
                    save_rgb(dest/f"{raw:03d}_{'delete_a37' if delete else 'keep'}_street_1920.png", rgb)
                    assert scene["meta"]["native_a6_rows_input"] == 0
                    del out, scene, rgb
                del direction, environment, yy_t, xx_t
            preview_records.append(dict(raw=raw, role="HELD_final_evaluation" if raw in HELD else "BUILD_diagnostic",
                full_image_hw=[1280, 1920], crop_xyxy=crop, black_background=True,
                alpha_depth_group_contributions_bitwise_equal=True))
            print(json.dumps(dict(stage="actual_asset_preview", raw=raw, output=str(dest))), flush=True)
            del data, camera, old, new
            rt._data.clear(); torch.cuda.empty_cache()
    assert rt.modelunchanged() and asset.assert_geometry_unchanged() and base.assert_geometry_unchanged()
    write_json(dest/"receipt.json", dict(status="PASS_FROZEN_RUNTIME_ASSET_EVALUATION",
        source_asset=str(Path(args.asset).resolve()), source_asset_sha256=sha(args.asset),
        comparator_asset=str(Path(args.compare).resolve()), comparator_sha256=sha(args.compare),
        mode=args.mode, metric_frames=len(records), previews=preview_records,
        alpha_depth_and_group_contributions_bitwise_unchanged=True,
        same_asset_for_keep_and_delete=True, native_a6_rows_input=0,
        frozen_native_road_environment_unchanged=True, geometry_updates=0, model_updates=0,
        generated_colours_not_measurements=True, HELD_used_only_for_final_evaluation=True,
        seconds=time.time()-start, runtime=rt.meta))
    if records:
        write_json(dest/"metrics.json", records)
        write_json(dest/"summary.json", dict(all49=aggregate(records),
            BUILD=aggregate([r for r in records if r["split"] == "BUILD"]),
            HELD=aggregate([r for r in records if r["split"] == "HELD"])))
    print(json.dumps(dict(status="PASS", output=str(dest), metric_frames=len(records))), flush=True)


if __name__ == "__main__":
    main()
