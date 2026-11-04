"""Conditional candidate-seed provenance; CPU/mmap only, no training or rasterization."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial import cKDTree
from candidate_seed_source import read_ply, digest

sys.path.append(r"C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages")
import torch

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
GEOMETRY = BASE / "work/c2_a37_20261103/observation/a6_geometry_minimal.npz"
TOP = [528407, 132314, 520617, 192988, 42783]


def stats(a):
    return dict(zip(["min", "p25", "median", "p75", "p95", "max"],
                    np.quantile(a, [0, .25, .5, .75, .95, 1]).tolist()))


def nearest(tree, point, xyz, times=None):
    distance, index = tree.query(point, k=1)
    row = {"distance_m": float(distance), "index": int(index), "xyz_world": xyz[index].tolist()}
    if times is not None:
        row["model_time"] = float(times[index])
    return row


def main():
    seed = np.load(OUT / "candidate_filtered_seed.npz")
    xyz, times = seed["xyz_world"], seed["time"]
    geo = np.load(GEOMETRY)
    a6xyz, a6raw = geo["world_xyz"], geo["raw_frame"]
    a6times = geo["model_time"][a6raw]
    wm = geo["model_from_world"]
    scale = float(np.mean(np.linalg.svd(wm[:3, :3], compute_uv=False)))
    target, target_xyz, _ = read_ply(OUT / "b10k_snapshot/points3d_dynamic_0.ply")
    target_times = np.asarray(target["nx"], float)
    a37tree, a6tree = cKDTree(target_xyz), cKDTree(a6xyz)
    top = []
    for root in TOP:
        point, timestamp = xyz[root], times[root]
        raw = int(np.rint(113 + timestamp / .02))
        same37 = np.flatnonzero(np.abs(target_times - timestamp) < 1e-7)
        same6 = np.flatnonzero(a6raw == raw)
        row = {"candidate_filtered_root_id": root,
               "candidate_original_row": int(seed["original_row_indices"][root]),
               "candidate_time": float(timestamp), "candidate_raw_frame": raw,
               "candidate_world_xyz": point.tolist(),
               "candidate_model_xyz": (point @ wm[:3, :3].T + wm[:3, 3]).tolist(),
               "nearest_a37_any_time": nearest(a37tree, point, target_xyz, target_times),
               "nearest_a6_any_time": nearest(a6tree, point, a6xyz, a6times)}
        for label, ids, points, stamps in [("nearest_a37_same_time", same37, target_xyz, target_times),
                                            ("nearest_a6_same_time", same6, a6xyz, a6times)]:
            if len(ids):
                value = nearest(cKDTree(points[ids]), point, points[ids], stamps[ids])
                value["index"] = int(ids[value["index"]])
                row[label] = value
            else:
                row[label] = None
        top.append(row)

    model, step = torch.load(OUT / "b10k_snapshot/chkpnt10000.pth", map_location="cpu", weights_only=False, mmap=True)
    assert step == 10000
    lineage = model[14]["sgcm_lineage_v1"]["lineage_ids"].numpy()
    final_xyz, final_t = model[1].detach().numpy(), model[7].detach().numpy().reshape(-1)
    counts = np.bincount(lineage, minlength=len(xyz))
    singleton_rows = np.flatnonzero(counts[lineage] == 1)
    singleton_roots = lineage[singleton_rows]
    candidate_model = (xyz @ wm[:3, :3].T + wm[:3, 3]).astype(np.float32)
    delta = np.linalg.norm(final_xyz[singleton_rows].astype(float) - candidate_model[singleton_roots], axis=1)
    wrong_roots = (singleton_roots + 123457) % len(xyz)
    wrong_delta = np.linalg.norm(final_xyz[singleton_rows].astype(float) - candidate_model[wrong_roots], axis=1)
    time_delta = np.abs(final_t[singleton_rows].astype(float) - times[singleton_roots])
    sampled = singleton_rows[np.linspace(0, len(singleton_rows) - 1, min(50000, len(singleton_rows)), dtype=int)]
    near_dist, near_ids = cKDTree(candidate_model).query(final_xyz[sampled], k=1, workers=2)
    record = {
        "status": "CONDITIONAL_CANDIDATE_SEED_PROVENANCE",
        "claim_limit": "Candidate filtering matches original initial count, but original B10k points3d.ply hash/order are unavailable. All candidate-root world locations and semantics are conditional. Final parameter drift is expected after 10000 updates; comparisons use saved raw _xyz, not time-evaluated SHM positions.",
        "candidate_npz_sha256": digest(OUT / "candidate_filtered_seed.npz"),
        "a6_geometry": str(GEOMETRY), "a6_geometry_sha256": digest(GEOMETRY),
        "a6_geometry_limit": "Real raw-frame LiDAR returns selected by annotated a6 OBB; may include ground/other returns and are not an exact semantic mesh.",
        "world_to_model_source": "a6_geometry_minimal.model_from_world; preparation derives it from prior native B-model road bundle",
        "world_to_model": wm.tolist(), "world_to_model_scale": scale,
        "top_candidate_roots": top,
        "singleton_sequence_consistency": {
            "final_checkpoint_step": step, "final_rows": len(lineage),
            "singleton_families_compared": len(singleton_rows),
            "comparison": "saved final raw _xyz versus world-to-model(candidate_xyz[lineage_id]); singleton final families only",
            "assigned_root_distance_model": stats(delta),
            "assigned_root_distance_world_m_equivalent": stats(delta / scale),
            "fraction_saved_xyz_bitwise_equal_candidate_float32": float(np.mean(np.all(final_xyz[singleton_rows] == candidate_model[singleton_roots], axis=1))),
            "fraction_assigned_distance_below_world_m": {str(e): float(np.mean(delta / scale < e)) for e in [1e-4, .001, .01, .1, 1.]},
            "time_abs_delta": stats(time_delta),
            "fraction_time_abs_delta_below_1e_6": float(np.mean(time_delta < 1e-6)),
            "wrong_root_control": "same singleton rows versus candidate root IDs cyclically shifted by123457",
            "wrong_root_distance_world_m_equivalent": stats(wrong_delta / scale),
            "sampled_nearest_candidate_check_count": len(sampled),
            "sampled_nearest_candidate_equals_assigned_lineage_fraction": float(np.mean(near_ids == lineage[sampled])),
            "sampled_nearest_candidate_distance_world_m_equivalent": stats(near_dist / scale),
        },
    }
    (OUT / "candidate_provenance_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"top_candidate_roots": [{"root": row["candidate_filtered_root_id"], "time": row["candidate_time"],
         "a6_same_time_m": row["nearest_a6_same_time"]["distance_m"] if row["nearest_a6_same_time"] else None,
         "a37_same_time_m": row["nearest_a37_same_time"]["distance_m"] if row["nearest_a37_same_time"] else None} for row in top],
         "singleton_sequence_consistency": record["singleton_sequence_consistency"]}))


if __name__ == "__main__":
    main()
