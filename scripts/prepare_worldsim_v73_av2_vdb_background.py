"""Same AV2 build returns, official TSDF, exact per-return origin grouping.

Known Actor poses and heldout beam files come unchanged from the parent scene
data. Only build sweeps are read. Both native and build-carved meshes are saved;
this constructs a background candidate, without evaluating external quality.
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
import vdbfusion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from motion_proj.worldsim_v73.av2_geometry import AV2MetricLog, sizes_at
from motion_proj.worldsim_v73.vdb_integration import integrate_per_origin
from prepare_worldsim_v73_av2_scene_geometry import carve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-data', type=Path, required=True)
    parser.add_argument('--sensor-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-size-m', type=float, default=.1)
    parser.add_argument('--sdf-trunc-m', type=float, default=.3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    source = json.loads((args.scene_data/'index.json').read_text())
    save('manifest.json', {
        'task_id': 'WS-V73-M4-AV2-SCENE-DATA-01',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'parent_scene_data': str(args.scene_data), 'sensor_root': str(args.sensor_root),
        'log_ids': [scene['log_id'] for scene in source['scenes']],
        'implementation': 'official vdbfusion wheel',
        'vdbfusion_version': importlib.metadata.version('vdbfusion'),
        'voxel_size_m': args.voxel_size_m, 'sdf_trunc_m': args.sdf_trunc_m,
        'space_carving': True, 'fill_holes': False, 'min_weight': 0.,
        'input_selection': 'same parent build sweeps, valid sensor poses, outside every valid-time known box+0.1m; no sensor-near removal or input cap',
        'endpoints': 'published reference-ego compensated xyz transformed to world once; never apply per-point ego transform again to these endpoints',
        'origins': 'actual calibrated per-return world sensor origins; group exact identical float64 origins without time/position quantization or frame-origin substitution',
        'grouping': 'sort/partition once; preserve point order within each origin group; no repeated N-by-number-of-origins scan',
        'integration_weights': 'one unit per retained original return; parent PCA deduplicates float32 endpoints, so raw integration count can differ from its unique center count',
        'unknown_boundary': 'all 8 meshing-cell corners observed; no unknown-region closure or learned opacity',
        'carving': 'reuse common one-pass whole-triangle exclusion before all original build first-return ranges minus0.2m, with the same per-return origins',
        'readout_inputs': 'parent heldout beams, owners, sensor-known flags and readonly Actor trajectories linked unchanged; heldout values not evaluated or read for fusion',
        'model_quality_computed': False, 'optimizer_updates': 0,
        'external_build_values_processed': any(scene['role']=='external_confirmation' for scene in source['scenes']),
        'references': ['https://github.com/PRBonn/vdbfusion',
                       'https://raw.githubusercontent.com/PRBonn/vdbfusion/main/src/vdbfusion/vdbfusion/VDBVolume.cpp',
                       'https://github.com/argoverse/av2-api/blob/main/src/av2/structures/sweep.py'],
    })
    save('status.json', {'status': 'running', 'phase': 'build_metadata', 'logs_done': 0})
    scenes = []
    try:
        for parent in source['scenes']:
            tick = time.monotonic()
            log_id = parent['log_id']
            folder = args.output/parent['scene']
            folder.mkdir()
            data = AV2MetricLog(args.sensor_root/log_id)
            tracks = data.load_tracks()
            volume = vdbfusion.VDBVolume(args.voxel_size_m, args.sdf_trunc_m, space_carving=True)
            build_rays = []
            records = []
            for build in parent['build']:
                scan_tick = time.monotonic()
                stamp = int(build['sample_id'])
                sweep = data.sweep(Path('sensors/lidar')/(str(stamp)+'.feather'))
                annotated = np.zeros(sweep['raw_points'], bool)
                for owner, trajectory in tracks.items():
                    local, _, _, known = data.actor_coordinates(owner, sweep)
                    extent = sizes_at(trajectory, sweep['point_timestamps_ns'])/2
                    annotated |= known & np.all(np.abs(local) <= extent+.1, axis=1)
                sensor_known = sweep['valid_sensor_pose']
                keep = sensor_known & ~annotated
                save('status.json', {'status': 'running', 'phase': 'integrate_per_return',
                                     'current_log': log_id, 'sample_index': build['sample_index'],
                                     'logs_done': len(scenes), 'background_returns': int(keep.sum())})
                integration_tick = time.monotonic()
                count = integrate_per_origin(volume, sweep['points_world_m'][keep], sweep['origins_world_m'][keep])
                build_rays.append((sweep['origins_world_m'][sensor_known].astype(np.float32),
                                   sweep['directions_world'][sensor_known].astype(np.float32),
                                   sweep['observed_range_m'][sensor_known].astype(np.float32)))
                record = {'sample_index': build['sample_index'], 'sample_id': str(stamp),
                          'raw_returns': sweep['raw_points'], 'invalid_sensor_pose': int((~sensor_known).sum()),
                          'excluded_known_objects': int(annotated.sum()), **count,
                          'parent_retained_returns': build['background_points'],
                          'integration_s': time.monotonic()-integration_tick,
                          'scan_wall_s': time.monotonic()-scan_tick}
                records.append(record)
                print(json.dumps({'log_id': log_id, **record}), flush=True)
                del sweep, annotated, keep, sensor_known
            save('status.json', {'status': 'running', 'phase': 'extract_observed_mesh',
                                 'current_log': log_id, 'logs_done': len(scenes)})
            vertices, faces = volume.extract_triangle_mesh(fill_holes=False, min_weight=0.)
            vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
            faces = np.asarray(faces, np.uint32).reshape(-1, 3)
            del volume
            np.savez(folder/'background_uncarved.npz', vertices_world_m=vertices, faces=faces)
            save('status.json', {'status': 'running', 'phase': 'build_free_carving',
                                 'current_log': log_id, 'logs_done': len(scenes), 'triangles': len(faces)})
            retained, carving = carve(vertices, faces, build_rays)
            np.savez(folder/'background.npz', vertices_world_m=vertices, faces=faces[retained], retained_parent_face_ids=retained)
            for frame in parent['frames']:
                (folder/frame['file']).symlink_to((args.scene_data/parent['scene']/frame['file']).resolve())
            entry = {**parent, 'background_triangles': len(retained),
                     'background_points': parent['background_points'], 'carving': carving,
                     'tsdf': {'vertices': len(vertices), 'native_triangles': len(faces),
                              'integrated_returns': sum(row['integrated_points'] for row in records),
                              'origin_groups': sum(row['origin_groups'] for row in records),
                              'voxel_size_m': args.voxel_size_m, 'sdf_trunc_m': args.sdf_trunc_m,
                              'build': records, 'wall_s': time.monotonic()-tick}}
            (folder/'scene.json').write_text(json.dumps(entry, indent=2)+'\n')
            scenes.append(entry)
            save('status.json', {'status': 'running', 'phase': 'scene_complete', 'current_log': log_id,
                                 'logs_done': len(scenes), 'elapsed_s': time.monotonic()-started})
            print(json.dumps({'log_id': log_id, 'logs_done': len(scenes), **carving,
                              'wall_s': entry['tsdf']['wall_s']}), flush=True)
            del data, tracks, build_rays, vertices, faces
        result = {'status': 'done', 'scenes': scenes, 'wall_s': time.monotonic()-started,
                  'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  'model_quality_computed': False, 'optimizer_updates': 0,
                  'boundary': 'build-only background candidate; parent heldout beams and known Actor trajectories unchanged; no new-log quality evaluation'}
        save('index.json', result)
        save('status.json', {'status': 'done', 'logs_done': len(scenes)})
        print(json.dumps({key: value for key, value in result.items() if key != 'scenes'}), flush=True)
    except Exception as exc:
        save('status.json', {'status': 'failed', 'logs_done': len(scenes),
                             'exception': type(exc).__name__, 'message': str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
