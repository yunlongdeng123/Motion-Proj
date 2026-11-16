"""由训练前固定世界图集生成五帧同相机监督；不读取联合渲染作道路真值。"""
from pathlib import Path
import argparse
import json
import os
import hashlib
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "2")
import numpy as np
from PIL import Image
from .road_asset import WorldRoadMaterialAtlas, _bilinear_grid
from .road_surface_projection import project_surface_mesh


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def camera_rays(bank, observation, frame_id, wh):
    width, height = wh
    lines = (bank / "calib" / (frame_id + ".txt")).read_text().splitlines()
    calibration = {line.split(":")[0]: np.fromstring(line.split(":")[1], sep=" ") for line in lines}
    intrinsic = calibration["P0"].reshape(3, 4)[:, :3].copy()
    intrinsic[:2] *= width / observation["camera_size"][0]
    camera = np.eye(4)
    camera[:3] = calibration["Tr_velo_to_cam_0"].reshape(3, 4)
    camera = camera @ np.linalg.inv(np.asarray(observation["ego_world"]).reshape(4, 4))
    inverse = np.linalg.inv(camera)
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack([xx + .5, yy + .5, np.ones_like(xx)], axis=2) @ np.linalg.inv(intrinsic).T @ inverse[:3, :3].T
    return inverse[:3, 3], rays


def camera_matrices(bank, observation, frame_id, wh):
    width, _ = wh
    lines = (bank / "calib" / (frame_id + ".txt")).read_text().splitlines()
    calibration = {line.split(":")[0]: np.fromstring(line.split(":")[1], sep=" ") for line in lines}
    intrinsic = calibration["P0"].reshape(3, 4)[:, :3].copy()
    intrinsic[:2] *= width / observation["camera_size"][0]
    camera = np.eye(4)
    camera[:3] = calibration["Tr_velo_to_cam_0"].reshape(3, 4)
    camera = camera @ np.linalg.inv(np.asarray(observation["ego_world"]).reshape(4, 4))
    return intrinsic, camera


def intersect_atlas(atlas, origin, rays, initial_depth):
    depth = np.asarray(initial_depth, np.float64).copy()
    for _ in range(10):
        hit = origin + depth[:, None] * rays
        sample = atlas.query(hit, surface_tolerance_m=np.inf)
        point, du, dv = _bilinear_grid(atlas.surface_world_xyz, sample["uv"], derivatives=True)
        determinant = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        gx = (du[:, 2] * dv[:, 1] - dv[:, 2] * du[:, 1]) / determinant
        gy = (-du[:, 2] * dv[:, 0] + dv[:, 2] * du[:, 0]) / determinant
        denominator = rays[:, 2] - gx * rays[:, 0] - gy * rays[:, 1]
        correction = np.divide(hit[:, 2] - point[:, 2], denominator,
            out=np.zeros(len(hit)), where=np.abs(denominator) > 1e-7)
        depth -= np.clip(correction, -2, 2)
        if not len(correction) or np.max(np.abs(correction)) < 1e-6:
            break
    hit = origin + depth[:, None] * rays
    sample = atlas.query(hit, surface_tolerance_m=np.inf)
    surface, _, _, _ = _bilinear_grid(atlas.surface_world_xyz, sample["uv"])
    converged = np.abs(hit[:, 2] - surface[:, 2]) <= .03
    valid = sample["inside"] & converged & (depth > .2) & (np.linalg.norm(hit - origin, axis=1) <= 60)
    return hit, sample, valid


def build(asset, output, frames=(86, 110, 134, 179, 198), projection="triangle"):
    asset, output = Path(asset).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    campaign = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
    cfg = json.loads((campaign / "scene065_material_loop_20261114_r1/config.json").read_text())
    source = json.loads((campaign / "scene065_closedloop_20261116_r1/handoff_coverage_r2/persistent/scene.json").read_text())
    atlas = WorldRoadMaterialAtlas.load(asset / "atlas.npz")
    provenance = load(asset / "provenance.npz")
    road = load(asset / "road.npz")
    bank = Path(cfg["bank"])
    receipt = json.loads((bank / "receipt.json").read_text())
    ids = {row["raw"]: row["id"] for row in receipt["frames"]}
    frame_data = {row["key"]: row for row in source["frames"]}
    width, height = source["render_wh"]
    grid = atlas.grid_world_row
    guard = provenance.get("guarded_world_rows", np.zeros(len(provenance["known"]), bool))
    rows = []
    for raw in frames:
        observation = json.loads((bank / "observations" / f"{raw:03d}.json").read_text())
        view = load(Path(cfg["road_views"]) / f"{raw:03d}.npz")
        labels = np.asarray(Image.open(frame_data[raw]["instance_mask"]).resize((width, height), Image.Resampling.NEAREST))
        K = (labels > 0) & (labels != 10)
        mesh = None
        if projection == "triangle":
            intrinsic, camera = camera_matrices(bank, observation, ids[raw], (width, height))
            mesh = project_surface_mesh(atlas, provenance["surface_supported"], intrinsic, camera, [height, width],
                road.get("minimum_model_depth", np.zeros(len(road["active"]))) / float(road["meter_to_model"]))
            finite = mesh["valid"]
            pixel_ids = np.flatnonzero(finite.ravel())
            hit = mesh["world_xyz"].reshape(-1, 3)[pixel_ids]
            query = atlas.query_xy(hit[:, :2])
            inside = query["inside"] & (mesh["camera_depth_m"].ravel()[pixel_ids] <= 60)
        else:
            finite = view["inside"] & np.isfinite(view["ground_depth"]) & (view["ground_depth"] > 0)
            origin, rays = camera_rays(bank, observation, ids[raw], (width, height))
            pixel_ids = np.flatnonzero(finite.ravel())
            hit, query, inside = intersect_atlas(atlas, origin, rays.reshape(-1, 3)[pixel_ids], view["ground_depth"].ravel()[pixel_ids])
        uv = query["uv"]
        low = np.minimum(np.floor(np.clip(uv, [0, 0], [grid.shape[1] - 1, grid.shape[0] - 1])).astype(int), [grid.shape[1] - 2, grid.shape[0] - 2])
        x, y = low.T
        corner = np.stack([grid[y, x], grid[y, x + 1], grid[y + 1, x], grid[y + 1, x + 1]], axis=1)
        frac = np.clip(uv - low, 0, 1)
        fx, fy = frac.T
        weight = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
        contribute = weight > 1e-7
        def all_corners(value):
            return np.all(~contribute | value[corner], axis=1)
        coverage_request = provenance.get("coverage_requested", provenance["requested"])
        legal_world = all_corners(coverage_request & provenance["legal_write"])
        geometry_support = all_corners(provenance["surface_supported"])
        if mesh is not None:
            triangle_ids = mesh["triangle_world_rows"].reshape(-1, 3)[pixel_ids]
            legal_world = np.all(coverage_request[triangle_ids], axis=1)
            geometry_support = np.all(provenance["surface_supported"][triangle_ids], axis=1)
        persistent_guard = np.any(contribute & guard[corner], axis=1)
        known = all_corners(provenance["known"])
        legal = inside & legal_world & geometry_support & ~persistent_guard
        def image(value, fill=0):
            value = np.asarray(value)
            result = np.full((height * width, *value.shape[1:]), fill, dtype=value.dtype)
            result[pixel_ids] = value
            return result.reshape(height, width, *value.shape[1:])
        source_map = image(query["source"], fill=0)
        unknown = image(query["unknown"], fill=True)
        alpha_valid = image(legal) & ~K & ~view["retained_reveal_guard"] & ~view["retained_box_guard"]
        color_valid = alpha_valid & ~unknown
        frozen = view["target"] & view["inside"] & ~K & ~view["retained_reveal_guard"] & ~view["retained_box_guard"]
        target_rgb = image(query["rgb_srgb"]).transpose(2, 0, 1)
        target_rgb[:, ~color_valid] = 0
        mesh_data = {} if mesh is None else {"triangle_world_rows": mesh["triangle_world_rows"],
            "triangle_weights": mesh["triangle_weights"], "triangle_face_id": mesh["triangle_face_id"],
            "surface_reprojection_error_px": mesh["reprojection_error_px"]}
        np.savez_compressed(output / f"{raw:03d}.npz", target_rgb_srgb=target_rgb,
            target_alpha=alpha_valid.astype(np.float32), legal=alpha_valid, legal_color=color_valid,
            known=image(known), confidence=image(query["confidence"]) * color_valid,
            K=K, retained_world_guard=image(persistent_guard), geometry_support=image(geometry_support),
            surface_intersection_valid=image(inside),
            ground_world_xyz=image(hit, fill=np.nan), atlas_uv=image(uv, fill=np.nan), source=source_map,
            source_raw=image(query["source_raw"], fill=-1), source_world_row=image(query["source_world_row"], fill=-1),
            source_corner_world_rows=image(corner, fill=-1), source_corner_weights=image(weight),
            world_row_index=image(grid[np.clip(np.rint(uv[:, 1]).astype(int), 0, grid.shape[0] - 1),
                                       np.clip(np.rint(uv[:, 0]).astype(int), 0, grid.shape[1] - 1)], fill=-1),
            frozen_evaluation_domain=frozen, full_target=(labels == 10), unknown=unknown, **mesh_data)
        Image.fromarray(np.uint8(np.clip(target_rgb.transpose(1, 2, 0), 0, 1) * 255 + .5)).save(output / f"{raw:03d}_target.png")
        rows.append(dict(raw=raw, legal_surface_pixels=int(alpha_valid.sum()), legal_color_pixels=int(color_valid.sum()),
            frozen_evaluation_pixels=int(frozen.sum()), full_target_pixels=int((labels == 10).sum()),
            unsupported_frozen_pixels=int((frozen & ~image(geometry_support)).sum()),
            surface_intersection_failed_frozen_pixels=int((frozen & ~image(inside)).sum()),
            unknown_material_frozen_pixels=int((frozen & unknown).sum()),
            persistent_guard_pixels=int(image(persistent_guard).sum()), protected_pixels=int(K.sum())))
    report = dict(schema="fixed_world_road_supervision.v1", frames=list(frames), asset=str(asset),
        fixed_before_training=True, target_role="world-atlas self-consistency projection; not hidden road truth",
        coordinate="native renderer pinhole pixel centres at x+.5,y+.5; fixed final atlas surface intersection",
        color="linear-light bilinear atlas query returned to sRGB; native road raster uses sRGB appearance",
        surface_projection=projection,
        road_target_from_old_joint_RGB=False, geometry_and_material_support_separate=True,
        source_lookup="four source_corner_world_rows/weights reference immutable asset provenance.npz",
        frozen_denominators_preserved=True, rows=rows)
    (output / "receipt.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    def pin(path):
        digest=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
                digest.update(block)
        return dict(path=str(path),sha256=digest.hexdigest())
    freeze=dict(schema="immutable_road_training_supervision.v1", source_asset=str(asset),
        source_files=[pin(asset/name) for name in ["road.npz","atlas.npz","provenance.npz"]],
        target_files=[pin(output/f"{raw:03d}.npz") for raw in frames],
        target_role="fixed pretraining world-atlas self-consistency projection; no hidden GT", actual_training_updates=0)
    (output/"freeze.json").write_text(json.dumps(freeze,indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection", choices=["triangle", "newton"], default="triangle")
    args = parser.parse_args()
    build(args.asset, args.output, projection=args.projection)


if __name__ == "__main__":
    main()
