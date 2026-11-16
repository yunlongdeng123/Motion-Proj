"""独立的实测道路漆线片段；未知虚线相位不被强制绘制成实线。"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.ndimage import label
from .road_asset import WorldRoadMaterialAtlas


def observed_segments(atlas, observed_white, observed_yellow):
    grid = atlas.grid_world_row
    world = atlas.surface_world_xyz.reshape(-1, 3)
    rows = []
    for paint, mask in [("white", observed_white), ("yellow", observed_yellow)]:
        qualified = np.asarray(mask, bool)[grid] & (atlas.source == 1) & atlas.legal
        components, count = label(qualified, structure=np.ones((3, 3)))
        sizes = np.bincount(components.ravel())
        for component in range(1, count + 1):
            if sizes[component] < 12:
                continue
            flat = np.flatnonzero(components.ravel() == component)
            if len(flat) < 12:
                continue
            points = world[flat]
            center = points[:, :2].mean(axis=0)
            _, _, vectors = np.linalg.svd(points[:, :2] - center, full_matrices=False)
            direction = vectors[0]
            if direction[np.argmax(np.abs(direction))] < 0:
                direction = -direction
            perpendicular = np.array([-direction[1], direction[0]])
            along = (points[:, :2] - center) @ direction
            across = (points[:, :2] - center) @ perpendicular
            length = float(np.ptp(along))
            width = float(np.quantile(across, .95) - np.quantile(across, .05) + .02)
            if length < .15 or width > .40 or length < 3 * width:
                continue
            lo, hi = int(np.argmin(along)), int(np.argmax(along))
            endpoints = points[[lo, hi]].tolist()
            source_ids = grid.ravel()[flat]
            rows.append(dict(id=len(rows), paint=paint, endpoint_world_xyz=endpoints,
                endpoint_observed_world_row_ids=source_ids[[lo, hi]].tolist(), direction_world_xy=direction.tolist(),
                width_m=width, length_m=length, observed_node_count=len(flat),
                observed_world_row_ids=source_ids.tolist(), dash_phase_known=False, dash_phase_m=None,
                style="observed paint fragment; solid/dashed continuation unverified", source="legal observed RGB only"))
    return dict(schema="observed_world_road_marking_structure.v1", segments=rows,
        inferred_segments=[], forced_paint_texels=0, unknown_dash_phase_preserved=True,
        scope="metric endpoints, tangent and width of legal observed color fragments; no unsupported line extension")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    atlas = WorldRoadMaterialAtlas.load(args.asset / "atlas.npz")
    with np.load(args.asset / "markings.npz") as saved:
        report = observed_segments(atlas, saved["observed_white"], saved["observed_yellow"])
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(segments=len(report["segments"]), forced_paint_texels=0)), flush=True)


if __name__ == "__main__":
    main()
