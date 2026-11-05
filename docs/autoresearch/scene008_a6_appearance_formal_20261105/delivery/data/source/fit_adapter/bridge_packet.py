"""Adapt the reviewed raw106/etc native operator package without a GPU rerun.

Generated RGB is explicitly sampled at native pixel centres through the
recorded affine. Native operator rows, actual camera, crop and masks remain
authoritative. This is explicit resampling, never an implicit shape resize.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.sparse import load_npz, save_npz

from adapter import (NATIVE_SCHEMA, SCHEMA, array_sha, bank, fit_cli, scalar, sha,
                     original_zero_columns, validate_view, write_json)

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
OLD = CAM/"scene008_a6_asset_controls_20261105"


def binary_png(path):
    value = np.asarray(Image.open(path))
    if value.ndim != 2 or not np.isin(value, [0, 1, 255]).all():
        raise ValueError("Expected a separate binary candidate mask: " + str(path))
    return value != 0


def prepare(args):
    packet = Path(args.packet).resolve()
    masks = Path(args.mask_directory).resolve() if args.mask_directory else packet
    output = Path(args.output).resolve()
    prep = output.with_name(output.name+"_inputs")
    prep.mkdir(parents=True, exist_ok=False)
    transform = bank(packet/"transform.npz")
    samples = bank(packet/"operator_samples.npz")
    receipt = json.loads((packet/"receipt.json").read_text())
    base = bank(args.asset)
    if receipt.get("geometry_hash") and receipt["geometry_hash"] != scalar(base["geometry_hash"]):
        raise ValueError("The packet kernel/geometry does not match the source asset")
    exact_zero = original_zero_columns(load_npz(args.build_design), len(base["xyz"]))
    if "column_unknown" in samples and not np.array_equal(samples["column_unknown"], exact_zero):
        raise ValueError("Packet unknown columns do not match the supplied real BUILD design")
    W = load_npz(packet/"design_W_premultiplied.npz").astype(np.float64).tocsr()
    A = load_npz(packet/"design_A_normalized.npz").astype(np.float64).tocsr()
    alpha = np.asarray(samples["alpha"], np.float64)
    norm_error = (W.multiply(1/np.maximum(alpha, 1e-12)[:, None])-A).tocsr()
    norm_error = float(abs(norm_error.data).max()) if norm_error.nnz else 0.
    alpha_error = float(abs(np.asarray(W.sum(1)).ravel()-alpha).max())
    if norm_error > 1e-6 or alpha_error > 5e-6:
        raise ValueError("Precomputed A/W/alpha conservation mismatch")
    if W.shape != (len(alpha), len(base["xyz"])):
        raise ValueError("Native packet primitive count mismatch")
    if receipt.get("design_sha256") and receipt["design_sha256"] != sha(packet/"design_A_normalized.npz"):
        raise ValueError("Source design no longer matches its native receipt")
    raw = int(transform["raw"])
    if raw in (89, 95, 101, 105, 113, 121, 125, 130, 137):
        raise ValueError("HELD views cannot be used for generation fitting/selection")
    observation = Path(args.observation) if args.observation else CAM/"scene008_instances_20261105_r1/observations/frames"/f"{raw:03d}.npz"
    obs = bank(observation)
    h, w = np.asarray(samples["original_hw"], dtype=int)
    transform_hw = transform.get("original_hw", transform.get("highres_hw"))
    if not np.array_equal(transform_hw, [h, w]):
        raise ValueError("Transform and native-operator full image sizes differ")
    xy = np.asarray(samples["pixel_xy"], np.float64)
    pixels = np.asarray(samples["pixels"], np.int64)
    expected_xy = np.column_stack((pixels % w, pixels//w))
    if not np.array_equal(xy, expected_xy) or not np.array_equal(pixels, np.sort(pixels)):
        raise ValueError("Native rows do not match their full-image pixel identities")
    matrix = np.asarray(transform.get("original_to_generated_pixel_center",
                                    transform.get("highres_to_generated_pixel_center")), np.float64)
    inverse = np.asarray(transform.get("generated_to_original_pixel_center",
                                     transform.get("generated_to_highres_pixel_center")), np.float64)
    if not np.allclose(inverse@matrix, np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("Generated/native pixel affine is not invertible")
    generated_xy = (np.column_stack((xy, np.ones(len(xy))))@matrix.T)[:, :2]
    if "generated_xy" in samples and not np.allclose(generated_xy, samples["generated_xy"], atol=1e-10, rtol=0):
        raise ValueError("Provided per-row generated XY differs from the recorded affine")
    rgb = np.asarray(Image.open(args.candidate))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Candidate must be an explicit three-channel RGB PNG")
    gh, gw = rgb.shape[:2]
    if not np.array_equal(transform["generated_hw"], [gh, gw]):
        raise ValueError("Candidate does not match the recorded generated frame; no implicit resize")
    foreground = binary_png(masks/"object_mask_512.png")
    inpaint = binary_png(masks/"inpaint_mask_512.png")
    if foreground.shape != (gh, gw) or inpaint.shape != (gh, gw):
        raise ValueError("Explicit generated masks and candidate frame differ")
    # Nearest semantic masks, bilinear straight-sRGB targets at the same exact centres.
    nearest = np.floor(generated_xy+.5).astype(int)
    in_bounds = ((nearest >= 0).all(1) & (nearest[:, 0] < gw) & (nearest[:, 1] < gh))
    clipped = np.clip(nearest, [0, 0], [gw-1, gh-1])
    fg = foreground[clipped[:, 1], clipped[:, 0]]
    unknown = inpaint[clipped[:, 1], clipped[:, 0]]
    use = in_bounds & fg & unknown & ~samples["known"].astype(bool) & samples["inpaint"].astype(bool)
    if not use.any():
        raise ValueError("No explicitly unknown object rows map to the candidate")
    if "crop_xyxy" in transform:
        crop_xyxy = transform["crop_xyxy"].astype(int)
    else:
        crop_xyxy = transform["original_low_crop_xyxy"].astype(int)*np.tile(
            np.asarray([w/transform["lowres_hw"][1], h/transform["lowres_hw"][0]]), 2)
        if not np.array_equal(crop_xyxy, crop_xyxy.astype(int)):
            raise ValueError("High-resolution crop boundaries must be exact integers")
        crop_xyxy = crop_xyxy.astype(int)
    x0, y0, x1, y1 = crop_xyxy
    if not ((xy[use, 0] >= x0).all() and (xy[use, 0] < x1).all() and
            (xy[use, 1] >= y0).all() and (xy[use, 1] < y1).all()):
        raise ValueError("Candidate crop fails to contain its native target rays")
    shape = (int(y1-y0), int(x1-x0))
    crop_pixels = (pixels//w-y0)*(x1-x0)+(pixels % w-x0)
    target = np.column_stack([map_coordinates(rgb[..., c].astype(np.float64)/255.,
        [generated_xy[:, 1], generated_xy[:, 0]], order=1, mode="constant", cval=0., prefilter=False)
        for c in range(3)])
    rgb_crop = np.zeros((*shape, 3), np.float32)
    rgb_crop.reshape(-1, 3)[crop_pixels[use]] = target[use].astype(np.float32)
    fit_mask = np.zeros(shape, bool); fit_mask.ravel()[crop_pixels[use]] = True
    known_mask = np.zeros(shape, bool)
    inside = ((xy[:, 0] >= x0)&(xy[:, 0] < x1)&(xy[:, 1] >= y0)&(xy[:, 1] < y1))
    known_mask.ravel()[crop_pixels[inside & samples["known"].astype(bool)]] = True
    K = np.asarray(obs["K"], np.float64).copy()
    K[0] *= w/obs["rgb"].shape[1]; K[1] *= h/obs["rgb"].shape[0]
    Kc = K.copy(); Kc[:2, 2] -= [x0, y0]
    pose = np.asarray(obs["model_from_local"][0], np.float64)
    camera = np.asarray(obs["model_to_camera"], np.float64)
    scale = float(np.linalg.norm(pose[:3, :3], axis=0).mean())
    metric = camera@pose; metric[:3, :4] /= scale
    view = dict(raw=np.asarray(raw), camera_K=K, camera_K_crop=Kc,
        camera_from_local_m=metric, model_from_local=pose, model_to_camera=camera,
        meter_to_model=np.asarray(scale), model_time=obs["model_time"],
        full_image_hw=np.asarray([h, w]), crop_xyxy=crop_xyxy,
        RGB_known_mask=known_mask, RGB_unknown_mask=~known_mask)
    audit = validate_view(view)
    np.savez_compressed(prep/"view.npz", **view)
    np.savez_compressed(prep/"candidate_on_native_crop.npz", RGB_srgb=rgb_crop)
    np.savez_compressed(prep/"masks.npz", foreground_mask=fit_mask, RGB_unknown_mask=~known_mask)
    op = prep/"operator"; op.mkdir()
    save_npz(op/"weights.npz", W[use])
    np.savez_compressed(op/"rays.npz", full_pixels=pixels[use], crop_pixels=crop_pixels[use], alpha=alpha[use])
    source_hashes = {str(p):sha(p) for p in (packet/"receipt.json", packet/"transform.npz",
        packet/"operator_samples.npz", packet/"design_A_normalized.npz", packet/"design_W_premultiplied.npz",
        Path(args.candidate), masks/"object_mask_512.png", masks/"inpaint_mask_512.png", observation)}
    write_json(op/"operator.json", dict(schema=NATIVE_SCHEMA, primitive_count=len(base["xyz"]),
        geometry_hash=scalar(base["geometry_hash"]), provenance_hash=scalar(base["provenance_hash"]),
        camera_package_sha256=sha(prep/"view.npz"), eligible_mask_sha256=array_sha(fit_mask),
        foreground_erosion_pixels=0, weights_sha256=sha(op/"weights.npz"), rays_sha256=sha(op/"rays.npz"),
        imported_exact_native_packet=True, source_receipt=receipt, source_SHA256=source_hashes,
        W_vs_A_alpha_maxabs=norm_error, W_alpha_conservation_maxabs=alpha_error))
    manifest = dict(schema=SCHEMA, views=[dict(id=f"raw{raw}_external_candidate", camera_npz="view.npz",
        foreground_mask="masks.npz", unknown_mask="masks.npz", rgb="candidate_on_native_crop.npz",
        rgb_mode="straight_srgb", operator_dir="operator")])
    write_json(prep/"manifest.json", manifest)
    write_json(prep/"bridge_receipt.json", dict(status="PASS_EXPLICIT_NATIVE_PIXEL_MAPPING", raw=raw,
        native_full_image_hw=[int(h), int(w)], candidate_hw=[gh, gw], target_rows=int(use.sum()),
        source_rows=len(pixels), target_RGB_interpolation="bilinear at recorded pixel centres",
        target_mask_interpolation="nearest at the same recorded pixel centres",
        original_to_generated_pixel_center=matrix.tolist(), inverse_affine_maxabs=float(abs(inverse@matrix-np.eye(3)).max()),
        actual_camera_projection_audit=audit, source_SHA256=source_hashes,
        generated_2D_alignment_is_unverified=True, generated_colours_are_inferred_not_measurements=True,
        known_RGB_not_fitted=True, black_background_RGB_not_fitted=True))
    return prep/"manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mask-directory")
    parser.add_argument("--observation")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--build-design", default=str(OLD/"m2/design.npz"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ridge", type=float, default=.001)
    parser.add_argument("--min-prior-alpha", type=float, default=.9)
    parser.add_argument("--min-unknown-fraction", type=float, default=.25)
    parser.add_argument("--additional-zero-mask")
    parser.add_argument("--additional-protected-design")
    args = parser.parse_args()
    manifest = prepare(args)
    fit_cli(SimpleNamespace(manifest=str(manifest), asset=args.asset, build_design=args.build_design,
        output=args.output, steps=args.steps, ridge=args.ridge, min_prior_alpha=args.min_prior_alpha,
        min_unknown_fraction=args.min_unknown_fraction, resume=None,
        additional_zero_mask=args.additional_zero_mask,
        additional_protected_design=args.additional_protected_design))


if __name__ == "__main__":
    main()
