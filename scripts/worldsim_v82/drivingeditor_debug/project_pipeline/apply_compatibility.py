#!/usr/bin/env python3
"""Apply narrowly scoped, asserted compatibility fixes to a WORKING copy.

Use --dry-run to inspect. Refuses unexpected source or already changed files.
It never modifies the upstream baseline unless explicitly pointed at it.
"""
import argparse
import difflib
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, f"Expected one source fragment: {old[:90]!r}"
    return text.replace(old, new, 1)


def patches(root):
    changes = {}
    path = root / "scene/scene_utils.py"
    old = path.read_text(encoding="utf-8")
    new = replace_once(old, "    bbx_info_1: list = None\n", "    bbx_info_1: list = None\n    object_mask: np.array = None\n")
    changes[path] = (old, new)

    path = root / "scene/waymo_loader2.py"
    old = path.read_text(encoding="utf-8")
    start = old.index('    # if args.require_boundingbox:\n', old.index('def readWaymoInfo_add'))
    end = old.index('    pose_id = 0\n', start)
    # Preserve the original dynamic branch and its threshold exactly. Static
    # runs no longer attempt to read four unrelated dynamic point-cloud files.
    replacement = '''    pointcloud1 = np.asarray(pcd.points)
    pointcloud_time = np.asarray(pcd.normals)[:, 0].reshape(-1, 1)
    if args.including_dynamic:
        dynamic_paths = sorted(glob(os.path.join(args.source_path, "points3d_dynamic_*.ply")))
        dynamic_clouds = [o3d.io.read_point_cloud(path) for path in dynamic_paths]
        if not dynamic_clouds or any(not len(p.points) for p in dynamic_clouds):
            raise ValueError("including_dynamic=True requires nonempty real dynamic PLY files")
        dynamic_pointcloud_1 = np.asarray(dynamic_clouds[0].points)
        if args.require_boundingbox:
            if len(dynamic_clouds) < 2:
                raise ValueError("The optional legacy bounding-box branch requires two actors")
            dynamic_pointcloud_2 = np.asarray(dynamic_clouds[1].points)
        dynamic_pointcloud = np.concatenate([np.asarray(p.points) for p in dynamic_clouds])
        dynamic_time = np.concatenate([np.asarray(p.normals)[:, :1] for p in dynamic_clouds])
        if not len(dynamic_pointcloud):
            raise ValueError("including_dynamic=True requires real dynamic PLY points")
        # Exact nearest-neighbour query: retain precisely those points without
        # any dynamic point at Euclidean distance < 0.1 (the original rule).
        nearest, _ = cKDTree(dynamic_pointcloud).query(
            pointcloud1, k=1, eps=0, distance_upper_bound=0.1, workers=4)
        mask = nearest >= 0.1
        static_pointcloud = pointcloud1[mask]
        static_time = pointcloud_time[mask]
    else:
        static_pointcloud = pointcloud1
        static_time = pointcloud_time
        dynamic_pointcloud = np.empty((0, 3), dtype=np.float32)
        dynamic_time = np.empty((0, 1), dtype=np.float32)
'''
    new = old[:start] + replacement + old[end:]
    new = replace_once(new, "import os\n", "import os\nfrom glob import glob\n")
    new = replace_once(new, "from scipy.spatial import ConvexHull\n", "from scipy.spatial import ConvexHull, cKDTree\n")
    new = replace_once(new, 'np.pad(point_xyz, (0, 1), constant_values=1) @ ego_pose.T', 'np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ ego_pose.T')
    changes[path] = (old, new)

    path = root / "train.py"
    old = path.read_text(encoding="utf-8")
    new = replace_once(old, "\n        v_map = feature[1:]\n", "\n        v_map = feature[1:]\n        if dynamic_gaussians is None:\n            # The only branch is the static branch; retain its regularizers.\n            static_t_map = t_map\n            static_v_map = v_map\n")
    new = replace_once(new, "        dynamic_gaussians.restore(model_params, args)\n", '''        if dynamic_gaussians is not None:
            dynamic_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                             os.path.basename(args.start_checkpoint).replace("chkpnt", "dynamic_chkpnt", 1))
            (dynamic_params, dynamic_iter) = torch.load(dynamic_checkpoint)
            if dynamic_iter != first_iter:
                raise ValueError("Static and dynamic checkpoint iterations differ")
            dynamic_gaussians.restore(dynamic_params, args)
''')
    new = replace_once(new, "os.path.dirname(args.checkpoint)", "os.path.dirname(args.start_checkpoint)")
    new = replace_once(new, "os.path.basename(args.checkpoint)", "os.path.basename(args.start_checkpoint)")
    new = replace_once(new, "            (light_params, _) = torch.load(env_checkpoint)\n            env_map.restore(light_params)\n", '''            (light_params, light_iter) = torch.load(env_checkpoint)
            if light_iter != first_iter:
                raise ValueError("Static and environment checkpoint iterations differ")
            env_map.restore(light_params, args)
''')
    changes[path] = (old, new)

    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("working_copy", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--patch-out", type=Path)
    args = parser.parse_args()
    changes = patches(args.working_copy.resolve())
    diff = "".join("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/" + str(path.relative_to(args.working_copy.resolve())).replace("\\", "/"),
        tofile="b/" + str(path.relative_to(args.working_copy.resolve())).replace("\\", "/")))
        for path, (before, after) in changes.items())
    if args.patch_out:
        args.patch_out.write_text(diff, encoding="utf-8")
    print(diff)
    if not args.dry_run:
        for path, (_, after) in changes.items():
            newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
            path.write_bytes(after.replace("\n", newline).encode("utf-8"))


if __name__ == "__main__":
    main()
