"""复用已登记的测距道路筛选，保存独立几何锚点；不读取RGB颜色。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def collect_road_measurements(config, geometry):
    from drivingeditor_dynamic_edit.v44_road import project
    bank = Path(config["bank"])
    masks = Path(config["masks"])
    semantics = Path(config["semantics"])
    anchor = geometry["coordinate_anchor"]
    inverse = np.linalg.inv(anchor)
    plane = geometry["plane"]
    width, height = config["render_wh"]
    saved = json.loads((bank / "receipt.json").read_text())
    low = geometry["world_xyz"].min(axis=0)
    high = geometry["world_xyz"].max(axis=0)
    pool, frames = [], []
    for raw in saved["build_raws"]:
        obs = json.loads((bank / "observations" / f"{raw:03d}.json").read_text())
        cloud = np.fromfile(bank / obs["lidar_path"], dtype="<f4").reshape(-1, 6)[:, :3]
        ego = np.asarray(obs["ego_world"]).reshape(4, 4)
        world = cloud @ ego[:3, :3].T + ego[:3, 3]
        uv, depth = project(world, obs)
        pix = np.floor(uv * [width / obs["camera_size"][0], height / obs["camera_size"][1]]).astype(np.int64)
        inside = (depth > .1) & (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
        ids = np.flatnonzero(inside)
        x, y = pix[ids].T
        nearest = np.full((height, width), np.inf)
        np.minimum.at(nearest, (y, x), depth[ids])
        labels = np.asarray(Image.open(masks / f"{raw:03d}/labels.png"))
        foreground = np.asarray(Image.open(masks / f"{raw:03d}/dynamic_union.png")) > 0
        guard = binary_dilation(foreground | (labels > 0), iterations=config["source_guard_pixels"])
        semantic = np.asarray(Image.open(semantics / f"{raw:03d}.png").resize((width, height), Image.Resampling.NEAREST))
        good = (semantic[y, x] == config["road_semantic_id"]) & ~guard[y, x] & (depth[ids] <= nearest[y, x] + .3)
        ids = ids[good]
        local = world[ids] @ inverse[:3, :3].T + inverse[:3, 3]
        residual = np.abs(local[:, 2] - local[:, :2] @ plane[:2] - plane[2])
        okay = (residual <= .08) & (world[ids, :2] >= low[:2] - 1).all(axis=1) & (world[ids, :2] <= high[:2] + 1).all(axis=1)
        ids = ids[okay]
        pool.append(world[ids])
        frames.append(np.full(len(ids), raw, np.int16))
    pool, frames = np.concatenate(pool), np.concatenate(frames)
    before = len(pool)
    _, keep = np.unique(np.floor(pool[:, :2] / .10).astype(np.int64), axis=0, return_index=True)
    return pool[keep], frames[keep], dict(raw_measured_points=before, deduplicated_points=len(keep),
        spatial_dedup_m=.10, ground_height_residual_limit_m=.08, road_semantic_id=config["road_semantic_id"],
        foreground_guard_pixels=config["source_guard_pixels"], nearest_lidar_depth_tolerance_m=.3,
        source_frames=saved["build_raws"], RGB_used=False, hidden_RGB_used=False)
