"""Official VDBFusion TSDF from the same static build returns as PCA background.

Read only construction scans; heldout rays/ownership/poses are linked unchanged.
No hole filling: meshing needs observed weight at every cell corner. Optional
later explicit build-ray carving uses the existing common carving script.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import vdbfusion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_proj.worldsim_v73.actor_rays import ActorRayDataset
from motion_proj.worldsim_v73.native_data import interpolate_pose
from motion_proj.worldsim_v72.data.nuscenes_camera import _transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-data', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path,
                        default=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-size-m', type=float, default=.1)
    parser.add_argument('--sdf-trunc-m', type=float, default=.3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    save('manifest.json', {
        'task_id': 'WS-V73-M4-SCENE-DATA-01',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'parent_scene_data': str(args.scene_data), 'dataset_root': str(args.dataset_root),
        'implementation': 'official vdbfusion pip wheel',
        'vdbfusion_version': importlib.metadata.version('vdbfusion'),
        'voxel_size_m': args.voxel_size_m, 'sdf_trunc_m': args.sdf_trunc_m,
        'space_carving': True, 'fill_holes': False, 'min_weight': 0.,
        'background_inputs': 'same build scans; outside every known annotation box+0.1m; exclude acquisition-sensor abs(x),abs(y)<1m',
        'integration': 'all retained world endpoints and original scan sensor origin; uniform weight; no input point cap',
        'unknown_boundary': 'all 8 meshing-cell corners must have observed weight; default positive values in unobserved voxels do not create closing surfaces',
        'dynamic_boundary': 'annotated-object returns do not create static surfaces; their free prefixes are not integrated by this TSDF API; common post-mesh build carving can use all original beams',
        'measurement_boundary': 'same nuScenes scan-time poses as parent; no new deskew or semantic ground truth',
        'heldout_values_read': False, 'optimizer_updates': 0,
        'references': ['https://github.com/PRBonn/vdbfusion',
                       'https://raw.githubusercontent.com/PRBonn/vdbfusion/main/src/vdbfusion/vdbfusion/VDBVolume.cpp',
                       'https://raw.githubusercontent.com/PRBonn/vdbfusion/main/src/vdbfusion/vdbfusion/MarchingCubes.cpp'],
    })
    save('status.json', {'status': 'running', 'phase': 'load_metadata'})
    try:
        source = json.loads((args.scene_data / 'index.json').read_text())
        sensor = ActorRayDataset(str(args.dataset_root), {s['scene'] for s in source['scenes']})
        index = sensor.index
        output = []
        for scene in source['scenes']:
            tick = time.monotonic()
            name = scene['scene']
            folder = args.output / name
            folder.mkdir()
            tracks = sensor.tracks[name]
            token = next(s['token'] for s in index.scenes if s['name'] == name)
            samples = sorted([s for s in index.samples if s['scene_token'] == token], key=lambda s: s['timestamp'])
            volume = vdbfusion.VDBVolume(args.voxel_size_m, args.sdf_trunc_m, space_carving=True)
            records = []
            for build in scene['build']:
                sample = samples[build['sample_index']]
                lidar = index.sample_data[index.data_by_sample_channel[(sample['token'], 'LIDAR_TOP')]]
                stamp = int(lidar['timestamp'])
                world = index.sensor_points_world(sample['token']).astype(np.float64)
                calibration = index.calibrated[lidar['calibrated_sensor_token']]
                ego = index.ego_poses[lidar['ego_pose_token']]
                sensor_pose = _transform(ego['translation'], ego['rotation']) @ _transform(calibration['translation'], calibration['rotation'])
                origin = sensor_pose[:3, 3]
                ranges = np.linalg.norm(world - origin, axis=-1)
                valid = np.isfinite(ranges) & (ranges > 0)
                world = world[valid]
                raw = np.fromfile(index.dataset_root / lidar['filename'], np.float32).reshape(-1, 5)[:, :3][valid]
                near = (np.abs(raw[:, 0]) < 1) & (np.abs(raw[:, 1]) < 1)
                annotated = np.zeros(len(world), bool)
                for trajectory in tracks.values():
                    pose = interpolate_pose(trajectory, stamp)
                    if pose is None:
                        continue
                    local = (world - pose[:3, 3]) @ pose[:3, :3]
                    extent = min(trajectory, key=lambda r: abs(r[0] - stamp))[2] / 2
                    annotated |= np.all(np.abs(local) <= extent + .1, axis=-1)
                kept = world[~annotated & ~near]
                integration_tick = time.monotonic()
                if len(kept):
                    volume.integrate(np.ascontiguousarray(kept, np.float64), np.ascontiguousarray(origin, np.float64))
                record = {'sample_index': build['sample_index'], 'sample_id': sample['token'],
                          'all_build_returns': len(world), 'known_object_returns': int(annotated.sum()),
                          'additional_sensor_near_removed': int((near & ~annotated).sum()),
                          'integrated_background_returns': len(kept), 'integration_s': time.monotonic() - integration_tick}
                records.append(record)
                save('status.json', {'status': 'running', 'phase': 'integrate_build', 'scene': name,
                                     'scenes_done': len(output), **record})
                print(json.dumps({'scene': name, **record}), flush=True)
            save('status.json', {'status': 'running', 'phase': 'extract_observed_mesh', 'scene': name})
            vertices, faces = volume.extract_triangle_mesh(fill_holes=False, min_weight=0.)
            vertices = vertices.astype(np.float32).reshape(-1, 3)
            faces = faces.astype(np.int32).reshape(-1, 3)
            np.savez(folder / 'background.npz', vertices_world_m=vertices, faces=faces)
            volume.extract_vdb_grids(str(folder / 'background.vdb'))
            del volume
            for frame in scene['frames']:
                (folder / frame['file']).symlink_to((args.scene_data / name / frame['file']).resolve())
            entry = {**scene, 'background_triangles': len(faces),
                     'tsdf': {'vertices': len(vertices), 'triangles': len(faces),
                              'voxel_size_m': args.voxel_size_m, 'sdf_trunc_m': args.sdf_trunc_m,
                              'space_carving': True, 'fill_holes': False, 'min_weight': 0.,
                              'build': records, 'wall_s': time.monotonic() - tick}}
            (folder / 'scene.json').write_text(json.dumps(entry, indent=2) + '\n')
            output.append(entry)
            print(json.dumps({'scene': name, 'vertices': len(vertices), 'triangles': len(faces),
                              'wall_s': entry['tsdf']['wall_s']}), flush=True)
            del vertices, faces
        result = {'status': 'done', 'scenes': output, 'wall_s': time.monotonic() - started,
                  'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                  'boundary': 'TSDF background from same static build returns; old heldout values not read; no Actor surface changes'}
        save('index.json', result)
        save('status.json', {'status': 'done'})
        print(json.dumps({k: v for k, v in result.items() if k != 'scenes'}), flush=True)
    except Exception as exc:
        save('status.json', {'status': 'failed', 'exception': type(exc).__name__, 'message': str(exc)})
        (args.output / 'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
