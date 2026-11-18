"""构建 raw86/110 的两种分辨率 2D 候选；保留原实验/渲染路径。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from PIL import Image

from teacher_supervision_cpu import (build_supervision, map_pixel_indices, save_package,
                                     scale_intrinsics)


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in (saved.files if keys is None else keys)}


def rgb(path):
    with Image.open(path) as im:
        if im.mode != "RGB" or im.format != "PNG":
            raise ValueError("实际输入必须是无损 RGB PNG: " + str(path))
        return np.asarray(im).copy()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def native_domains(case, extra, raw):
    roles = load(case / "B_roles_native.npz")
    guard = load(case / "guard_roles_native.npz")
    field = load(extra / "shadow_field.npz", ("probability", "uncertainty", "valid", "metadata_json"))
    evidence = load(extra / f"{raw:03d}_native_routing_evidence.npz", ("other_shadow_prediction",))
    route = read(extra / "routing_receipt.json")
    metadata = json.loads(str(field["metadata_json"].item()))
    policy = route["public_routing_receipt"]["policy"]
    high = (field["valid"] & (field["probability"] >= policy["high"]) & roles["road_observed"]
            & ~roles["foreground_possible"] & ~roles["K"] & ~roles["physical_unknown"])
    if not np.array_equal(high, roles["shadow_high"]):
        raise ValueError("实际冻结 shadow_high 与 policy/field 不一致")
    interior = (high & roles["W_output"] & ~roles["shadow_uncertain"] & ~roles["shadow_boundary_guard"]
                & (field["uncertainty"] <= policy["max_write_uncertainty"]))
    association = metadata["diagnostics"]["association_uncertainty"]
    if association > policy["max_write_uncertainty"]:
        interior[:] = False
    other = evidence["other_shadow_prediction"]
    teacher_write, k = guard["W_guard"], roles["K"]
    a = (roles["road_observed"] & roles["C_mat"] & roles["visible_support"] & ~guard["H_B"] & ~k
         & ~roles["shadow_possible"] & ~roles["shadow_uncertain"] & ~roles["shadow_boundary_guard"])
    c = roles["physical_unknown"] & teacher_write & ~k & ~other
    s = interior & ~other & teacher_write & ~k
    return dict(A=a, C=c, S=s, teacher_write=teacher_write, K=k, protected=np.zeros_like(k)), dict(
        policy=policy, association_uncertainty=association,
        teacher_write="frozen original output_guard_probe W_guard; H_B is not permission",
        source_W_output_equals_guard_W_old=bool(np.array_equal(roles["W_output"], guard["W_old"])),
        source_W_output_subset_W_guard=bool(np.all(~roles["W_output"] | teacher_write)),
        raw_S_rule="frozen high & W_output & ~uncertain & ~boundary & uncertainty_gate & association_gate & ~other_shadow & teacher_W_guard & ~K",
        geometric_world_and_retained_projection_qualification_available=False)


def camera(extra, raw):
    observation = read(extra / f"{raw:03d}.json")
    frame_id = f"0065{raw:03d}"
    lines = (extra / f"{frame_id}.txt").read_text().splitlines()
    calibration = {line.split(":")[0]: np.fromstring(line.split(":")[1], sep=" ") for line in lines}
    k = calibration["P0"].reshape(3, 4)[:, :3].copy()
    return k, observation["camera_size"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-assets", type=Path, required=True)
    parser.add_argument("--extra-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for raw in (86, 110):
        start = time.monotonic()
        case, extra = args.report_assets / f"scene065_raw{raw:03d}", args.extra_inputs / f"{raw:03d}"
        masks, routing = native_domains(case, extra, raw)
        np.savez_compressed(args.output / f"{raw:03d}_native_domains.npz", **masks)
        raw_rgb, observed = rgb(case / "B_raw_model.png"), rgb(case / "observation.png")
        k, camera_wh = camera(extra, raw)
        for wh in ((480, 320), (768, 512)):
            arrays, manifest = build_supervision(raw_rgb, observed, masks, output_wh=wh, intrinsics=k,
                                                 camera_wh=camera_wh, camera_convention="half_pixel",
                                                 source_kind="lossless_raw_model")
            row = dict(raw=raw, output_wh=list(wh), counts=manifest["counts"], routing=routing,
                       teacher_raw_identity=(bool(np.array_equal(arrays["target_C_srgb"], raw_rgb.transpose(2, 0, 1).astype(np.float32) / 255))
                                             if wh == (768, 512) else None))
            manifest.update(task_id="ROAD-2D-TEACHER-HF-CPU-20261119", raw=raw, routing=routing,
                            camera_source_convention="ASSUMED half_pixel from inherited CPU road_asset_supervision.camera_rays x+.5; not independently qualified against actual raster",
                            camera_convention_assumed=True, actual_raster_camera_convention_verified=False)
            folder = args.output / f"{raw:03d}_{wh[0]}x{wh[1]}"
            config = dict(raw_model=str(case / "B_raw_model.png"), observation=str(case / "observation.png"),
                          native_domains=str(args.output / f"{raw:03d}_native_domains.npz"), output_wh=list(wh),
                          source_kind="lossless_raw_model", camera=dict(intrinsics=k.tolist(), camera_wh=camera_wh,
                                                                        convention="half_pixel"))
            save_package(folder, arrays, manifest, sources=config)
            (folder / "rebuild_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # 同一相机射线在两种显式 convention 下均应代数一致；不是原生raster验收。
            points = np.array([[0, 0], [1919, 1279], [959.5, 639.5]])
            scaled_points = map_pixel_indices(points, camera_wh, wh)
            row["camera_ray_algebra_max"] = {}
            for convention, offset in (("half_pixel", .5), ("integer_center", 0.)):
                scaled_k = scale_intrinsics(k, camera_wh, wh, convention)
                first = np.c_[points + offset, np.ones(len(points))] @ np.linalg.inv(k).T
                second = np.c_[scaled_points + offset, np.ones(len(points))] @ np.linalg.inv(scaled_k).T
                row["camera_ray_algebra_max"][convention] = float(np.abs(first - second).max())
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        print(json.dumps(dict(raw=raw, elapsed_seconds=time.monotonic() - start)), flush=True)
    receipt = dict(schema="road_2d_teacher_cpu_build.v1", task_id="ROAD-2D-TEACHER-HF-CPU-20261119", rows=results,
                   GPU_executed=False, no_3d_code_modified=True, no_training=True,
                   masks_are_2d_candidates_not_old_training_denominators=True,
                   no_fixed_alpha_or_native_contribution_certificate=True)
    (args.output / "cpu_build_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
