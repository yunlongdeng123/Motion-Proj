"""Extract exact native alpha*T rays for explicit cropped generated targets.

Run only when a GPU extraction is authorised. This file does not train a
model, alter source assets, or load a native a6/road checkpoint. It keeps a
FULL-image camera; cropping only selects target pixels. A cropped FoV would
change native EWA covariance clamping and is intentionally not substituted.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.sparse import save_npz

from adapter import (NATIVE_SCHEMA, array_sha, bank, load_manifest, scalar, sha,
                     validate_asset, view_inputs, write_json)

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
OLD = CAM/"scene008_a6_asset_controls_20261105"
ENGINE = CAM/"r15_baseline_owner5_20261018/training_source/scene0008_089_137"
CONFIG = CAM/"scene008_a37_only_B10k_20261103_r1/formal10k/effective_config.json"


def native_projection_audit(asset, pose, camera, view):
    """Compare actual native projection to independently composed metre K/T."""
    import torch
    from adapter import project_local
    xyz = asset.xyz
    means = xyz@pose[:3, :3].T+pose[:3, 3]
    homogeneous = torch.cat((means, torch.ones_like(means[:, :1])), dim=1)
    clip = homogeneous@camera.full_proj_transform
    ndc = clip[:, :2]/clip[:, 3:4]
    native = (ndc+1)*ndc.new_tensor([camera.image_width, camera.image_height])/2-.5
    independent = project_local(xyz.cpu().numpy(), view, crop=False)
    discrepancy = abs(native.cpu().numpy()-independent)
    error = float(discrepancy.max())
    if error > .002:
        raise AssertionError(f"Native/full K/T projection differs by {error} pixels")
    return dict(full_image_native_vs_independent_maxabs_pixels=error,
                primitive_centres_tested=len(xyz), full_image_camera_used=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--source-root", default=str(OLD/"source"))
    parser.add_argument("--engine-root", default=str(ENGINE))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--foreground-erosion", type=int, default=1)
    args = parser.parse_args()
    source = Path(args.source_root).resolve()
    sys.path[:0] = [str(source), str(Path(args.engine_root).resolve())]
    import json
    import torch
    import sh_asset
    import linear_weights
    from scene.cameras import Camera
    if Path(sh_asset.__file__).resolve().parent != source or Path(linear_weights.__file__).resolve().parent != source:
        raise RuntimeError("Unexpected old native source binding")
    from sh_asset import SHAsset
    from linear_weights import design

    base = bank(args.asset); validate_asset(base)
    asset = SHAsset.from_geometry(args.asset).cuda()
    config = json.loads(Path(args.config).read_text())
    pipe = SimpleNamespace(compute_cov3D_python=bool(config.get("compute_cov3D_python", False)),
                           neg_fov=bool(config.get("neg_fov", False)))
    manifest, root = load_manifest(args.manifest)
    records = []
    with torch.no_grad():
        for entry in manifest["views"]:
            inputs = view_inputs(entry, root, args.foreground_erosion)
            view, eligible = inputs["view"], inputs["eligible"]
            if not eligible.any():
                raise ValueError(entry["id"] + ": empty explicit unknown foreground")
            h, w = view["full_image_hw"].astype(int)
            K = view["camera_K"].astype(np.float64)
            W = view["model_to_camera"].astype(np.float64)
            pose = torch.as_tensor(view["model_from_local"], device="cuda", dtype=torch.float32)
            raw = int(np.asarray(view.get("raw", -1)))
            camera = Camera(raw, W[:3, :3].T.copy(), W[:3, 3].copy(),
                cx=float(K[0, 2]), cy=float(K[1, 2]), fx=float(K[0, 0]), fy=float(K[1, 1]),
                image=torch.zeros((3, int(h), int(w))), image_name=entry["id"],
                resolution=(int(w), int(h)), timestamp=float(view.get("model_time", 0.)))
            projection = native_projection_audit(asset, pose, camera, view)
            full_mask = np.zeros((h, w), bool)
            x0, y0, x1, y1 = view["crop_xyxy"].astype(int)
            full_mask[y0:y1, x0:x1] = eligible
            rows = asset.rows(pose, camera, pipe, 1, precomputed=True)
            lin = design(camera, pipe, rows, torch.as_tensor(full_mask, device="cuda"))
            if not np.array_equal(lin["pixels"], np.flatnonzero(full_mask.ravel())):
                raise AssertionError("Native selected-pixel order changed")
            pixels = lin["pixels"].astype(np.int64)
            crop_pixels = (pixels//w-y0)*(x1-x0)+(pixels % w-x0)
            dest = (root/entry["operator_dir"]).resolve()
            dest.mkdir(parents=True, exist_ok=False)
            save_npz(dest/"weights.npz", lin["matrix"])
            np.savez_compressed(dest/"rays.npz", full_pixels=pixels, crop_pixels=crop_pixels,
                                alpha=lin["alpha"], full_image_hw=view["full_image_hw"],
                                crop_xyxy=view["crop_xyxy"])
            metadata = dict(schema=NATIVE_SCHEMA, id=entry["id"], primitive_count=len(asset.xyz),
                geometry_hash=scalar(base["geometry_hash"]), provenance_hash=scalar(base["provenance_hash"]),
                source_asset_sha256=sha(args.asset), camera_package_sha256=sha(inputs["camera_path"]),
                eligible_mask_sha256=array_sha(eligible), foreground_erosion_pixels=args.foreground_erosion,
                weights_sha256=sha(dest/"weights.npz"), rays_sha256=sha(dest/"rays.npz"),
                selected_pixels=len(pixels), native_alpha_maxabs=lin["alpha_maxabs"],
                native_colour_maxabs=lin["colour_maxabs"], projection=projection,
                appearance_targets_read=False, full_image_EWA_camera_preserved=True,
                fixed_geometry=True, extracted_by=sha(__file__),
                source_code_SHA256={str(Path(sh_asset.__file__)):sha(sh_asset.__file__),
                                    str(Path(linear_weights.__file__)):sha(linear_weights.__file__)})
            write_json(dest/"operator.json", metadata)
            records.append(dict(id=entry["id"], operator_dir=str(dest), selected_pixels=len(pixels)))
            print(json.dumps(records[-1]), flush=True)
            if not asset.assert_geometry_unchanged():
                raise AssertionError("Geometry changed during extraction")
            del lin, rows, camera, pose
            torch.cuda.empty_cache()
    print(json.dumps(dict(status="PASS_NATIVE_FIXED_GEOMETRY_EXTRACTION", views=records,
                          PVD_updates=0, geometry_updates=0, colour_fit_updates=0)), flush=True)


if __name__ == "__main__":
    main()
