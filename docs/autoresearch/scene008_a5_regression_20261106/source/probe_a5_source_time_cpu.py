"""Read-only verification of real a5 source indices, coordinates and timestamps."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD = B / 'nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137'
TRACK = '3S_GJHyCGGK4ypFPVKOOqA'


def main():
    prep = json.loads((OLD / 'preparation_manifest.json').read_text())
    vertices = PlyData.read(OLD / 'points3d_dynamic_0.ply')['vertex']
    actual = np.column_stack([vertices[key] for key in ['x', 'y', 'z']]).astype(np.float32)
    time = np.asarray(vertices['nx'])
    indices = np.load(OLD / 'dynamic_actor_assets/actor_0/source_lidar_indices.npz', allow_pickle=False)
    rows = []
    centers, seconds, model_times = [], [], []
    offset = 0
    for frame in prep['frames']:
        fid = frame['id']
        source = np.asarray(indices[fid], dtype=np.int64)
        lidar = np.fromfile(OLD / f'velodyne/{fid}.bin', np.float32).reshape(-1, 6)
        pose = np.loadtxt(OLD / f'pose/{fid}.txt')
        expected = (np.column_stack((lidar[source, :3].astype(np.float64), np.ones(len(source)))) @ pose.T)[:, :3].astype(np.float32)
        saved = actual[offset:offset + len(source)]
        saved_time = time[offset:offset + len(source)]
        assert np.array_equal(saved, expected), fid
        assert np.array_equal(saved_time, np.full(len(source), frame['model_time'], dtype=np.float32)), fid
        ann = json.loads((OLD / f'annotations/{fid}.json').read_text())
        box = next(label for label in ann['laser_labels'] if label['id'] == TRACK)['box']
        center = (pose @ [box['center_x'], box['center_y'], box['center_z'], 1])[:3]
        centers.append(center)
        seconds.append(frame['timestamp_micros'] / 1e6)
        model_times.append(frame['model_time'])
        rows.append(dict(frame_id=fid, source_rows=len(source), coordinates_exact=True, time_exact=True,
                         model_time=frame['model_time']))
        offset += len(source)
    assert offset == len(actual) == 21358
    speed = np.linalg.norm(np.diff(centers, axis=0)[:, :2], axis=1) / np.diff(seconds)
    ratio = np.diff(model_times) / np.diff(seconds)
    original = PlyData.read(OLD / 'points3d.ply')['vertex']
    static_xyz = np.column_stack([original[key] for key in ['x', 'y', 'z']]).astype(np.float64)
    static_time = np.asarray(original['nx'])
    nearest, index = cKDTree(actual.astype(np.float64)).query(static_xyz, distance_upper_bound=2e-5, workers=4)
    matching = nearest < 2e-5
    mismatches = np.abs(static_time[matching] - time[index[matching]]) > 1e-7
    print(json.dumps(dict(status='PASS_REAL_SOURCE_COORDINATE_AND_TIME_IDENTITY', GPU_used=False, training_updates=0,
                         source_points_verified=len(actual), frame_rows=rows,
                         sampled_static_rows_matching_a5=int(matching.sum()),
                         matching_static_a5_time_mismatches=int(mismatches.sum()),
                         a5_world_track_displacement_m=float(np.linalg.norm(centers[-1] - centers[0])),
                         a5_world_xy_speed_mps=dict(min=float(speed.min()), median=float(np.median(speed)), max=float(speed.max())),
                         model_time_per_physical_second=dict(min=float(ratio.min()), median=float(np.median(ratio)), max=float(ratio.max())),
                         coordinate_pipeline='raw LiDAR xyz -> per-frame ego pose -> world float32 PLY -> shared world-to-model transform; temporal nx is the same frame model_time',
                         note='GT boxes/centers are read only as diagnostics; no filtering or new seeds are created.'), indent=2))


if __name__ == '__main__':
    main()
