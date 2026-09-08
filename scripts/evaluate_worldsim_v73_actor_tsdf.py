"""Build-only canonical Actor TSDF with native meshes and common literal readout.

Official VDBFusion supplies integration/meshing. Unknown cell corners are not
closed. A second fixed surface removes whole triangles contradicted by any
original build first-return beam, using one intersection pass. Both surfaces
are retained; no target-time pruning, neural updates, or fallback surface.
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
import open3d as o3d
import torch
import vdbfusion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH, ray_statistics


def carve_build_faces(vertices, faces, build, margin=.2):
    bvh = SurfaceBVH(vertices, faces)
    removed = np.zeros(len(faces), bool)
    rays_count = 0
    contradictory_rays = 0
    for frame in build:
        origins = frame['origins_actor_m'].numpy()
        directions = frame['directions_actor'].numpy()
        observed = frame['observed_first_range_m'].numpy()
        rays_count += len(observed)
        if bvh.scene is None:
            continue
        for start in range(0, len(observed), 8192):
            rays = np.concatenate([origins[start:start+8192], directions[start:start+8192]], axis=1).astype(np.float32)
            hits = bvh.scene.list_intersections(o3d.core.Tensor(rays), nthreads=2)
            ids = hits['ray_ids'].numpy().astype(np.int64)
            depth = hits['t_hit'].numpy()
            triangles = hits['primitive_ids'].numpy().astype(np.int64)
            free = (depth > 1e-6) & (depth < observed[start+ids] - margin)
            removed[triangles[free]] = True
            contradictory_rays += len(np.unique(ids[free]))
    retained = np.flatnonzero(~removed)
    return faces[retained], retained, {
        'build_near_box_rays': rays_count, 'contradictory_build_rays_before': contradictory_rays,
        'original_triangles': len(faces), 'removed_triangles': int(removed.sum()),
        'retained_triangles': len(retained),
    }


def evaluate_surface(vertices, faces, rays):
    bvh = SurfaceBVH(vertices, faces)
    frames = []
    for frame in rays:
        origins = frame['origins_actor_m'].numpy()
        directions = frame['directions_actor'].numpy()
        observed = frame['observed_first_range_m'].numpy()
        positive = frame['positive_actor'].numpy()
        target = frame['points_actor_m'][frame['positive_actor']].numpy()
        depth = bvh.cast(origins, directions)
        distance = (bvh.scene.compute_distance(o3d.core.Tensor(target.astype(np.float32)), nthreads=2).numpy()
                    if len(target) and bvh.scene is not None else None)
        owned = ray_statistics(depth, observed, positive)
        all_rays = ray_statistics(depth, observed, np.ones(len(observed), bool))
        frames.append({
            'sample_index': frame['sample_index'], 'role': frame['role'], 'owned_ray': owned,
            'all_near_box_rays': len(observed),
            'mean_free_intrusion_m': all_rays['free_intrusion_m'] or 0.,
            'free_intrusion_rate': all_rays['early_rate'],
            'positive_points': len(target),
            'positive_surface_mean_m': float(distance.mean()) if distance is not None else None,
            'positive_surface_recall_02': float((distance <= .2).mean()) if distance is not None else (0. if len(target) else None),
        })
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-size-m', type=float, default=.1)
    parser.add_argument('--sdf-trunc-m', type=float, default=.3)
    parser.add_argument('--baseline-results', type=Path, help='reuse original same-cohort LiDAR PCA rows without rerunning its readout')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ['uncarved', 'carved']:
        (args.output / name).mkdir()
    started = time.monotonic()
    torch.set_num_threads(2)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    index = json.loads((args.actor_data / 'index.json').read_text())
    manifest = {
        'task_id': 'WS-V73-M2-TSDF-FUSION-01',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'actor_data': str(args.actor_data), 'cohort_actors': len(index['cases']),
        'roles': sorted({row['role'] for row in index['cases']}),
        'implementation': 'official vdbfusion wheel; native Marching Cubes mesh',
        'vdbfusion_version': importlib.metadata.version('vdbfusion'),
        'voxel_size_m': args.voxel_size_m, 'sdf_trunc_m': args.sdf_trunc_m,
        'space_carving': True, 'fill_holes': False, 'min_weight': 0.,
        'inputs': 'all original owned build returns with their canonical sensor origins; no input cap, images, learned weights or extra-time targets',
        'rigidity': 'use existing read-only Actor canonical endpoint/origin at each acquisition time; no registration fitting or independent per-frame corrections',
        'unknown_boundary': 'meshing requires observed weight at all 8 cell corners; no hole filling or PCA fallback when TSDF lacks support',
        'carving': 'second surface removes whole faces intersecting any original build central beam before range-0.2m; includes non-owned beams; one pass, coincident-hit residuals possible',
        'heldout_boundary': 'both surfaces constructed before heldout values enter common CPU hard evaluation; no heldout construction or surface rejection',
        'surface_density': 'native TSDF triangles are not fixed-size query patches; report vertex/face counts and area, not a matched output-budget claim',
        'legacy_surface_patches_field': 'counts TSDF triangle elements solely for common schema; never interpret these as learned 8-triangle patches',
        'baseline_results': str(args.baseline_results) if args.baseline_results else None,
        'optimizer_updates': 0,
        'external_test_read': any(row['role'] == 'external_confirmation' for row in index['cases']),
        'references': ['https://lightfield.stanford.edu/papers/volrange/',
                       'https://github.com/PRBonn/vdbfusion',
                       'https://raw.githubusercontent.com/PRBonn/vdbfusion/main/src/vdbfusion/vdbfusion/MarchingCubes.cpp'],
    }
    save('manifest.json', manifest)
    save('cohort.json', index['cases'])
    save('status.json', {'status': 'running', 'phase': 'build_fusion'})
    outputs = {'uncarved': [], 'carved': []}
    construction = []
    try:
        for entry in index['cases']:
            tick = time.monotonic()
            case = torch.load(args.actor_data / entry['file'], map_location='cpu', weights_only=False)
            build = [frame for frame in case['rays'] if frame['role'] == 'build']
            volume = vdbfusion.VDBVolume(args.voxel_size_m, args.sdf_trunc_m, space_carving=True)
            integrated = 0
            origins_count = 0
            for frame in build:
                positive = frame['positive_actor']
                endpoints = frame['points_actor_m'][positive].numpy().astype(np.float64)
                origins = frame['origins_actor_m'][positive].numpy().astype(np.float64)
                if not len(endpoints):
                    continue
                unique_origins, group = np.unique(origins, axis=0, return_inverse=True)
                for i, origin in enumerate(unique_origins):
                    points = endpoints[group == i]
                    volume.integrate(np.ascontiguousarray(points), np.ascontiguousarray(origin))
                integrated += len(endpoints)
                origins_count += len(unique_origins)
            if integrated:
                vertices, faces = volume.extract_triangle_mesh(fill_holes=False, min_weight=0.)
                vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
                faces = np.asarray(faces, np.int64).reshape(-1, 3)
            else:
                vertices = np.empty((0, 3), np.float32)
                faces = np.empty((0, 3), np.int64)
            del volume
            carved, retained, carving = carve_build_faces(vertices, faces, build)
            record = {'actor': entry, 'integrated_owned_build_returns': integrated,
                      'integration_origin_groups': origins_count, 'vertices': len(vertices),
                      'carving': carving, 'build_s': time.monotonic() - tick}
            for name, triangles in [('uncarved', faces), ('carved', carved)]:
                elements = vertices[triangles]
                centers = elements.mean(1) if len(triangles) else np.empty((0, 3), np.float32)
                area = .5 * np.linalg.norm(np.cross(elements[:, 1]-elements[:, 0], elements[:, 2]-elements[:, 0]), axis=1).sum()
                torch.save({'vertices_actor_m': torch.from_numpy(vertices),
                            'faces': torch.from_numpy(triangles),
                            'centers_actor_m': torch.from_numpy(centers),
                            'surface_representation': 'native_tsdf_mesh',
                            'retained_parent_face_ids': torch.from_numpy(retained) if name == 'carved' else torch.arange(len(faces))},
                           args.output / name / (entry['owner'] + '_surface.pt'))
                frames = evaluate_surface(vertices, triangles, case['rays'])
                outputs[name].append({
                    'actor': entry, 'surface_patches': len(triangles), 'surface_element_unit': 'native_tsdf_triangle',
                    'surface_vertices': len(vertices), 'surface_active_vertices': len(np.unique(triangles)),
                    'surface_area_m2': float(area),
                    'frames': frames, 'extra_time_usage': 'evaluation_only',
                    'seed_support': {'lidar_fallback': False, 'prediction_unavailable': not len(triangles),
                                     'reason': 'no_observed_tsdf_zero_crossing_mesh' if not len(triangles) else None,
                                     'integrated_owned_build_returns': integrated},
                })
            record['wall_s'] = time.monotonic() - tick
            construction.append(record)
            save('status.json', {'status': 'running', 'phase': 'actor_complete', 'actors_done': len(construction),
                                'scene': entry['scene'], 'owner': entry['owner'], 'elapsed_s': time.monotonic()-started})
            print(json.dumps({'actors_done': len(construction), 'scene': entry['scene'],
                              'integrated_returns': integrated, **carving, 'wall_s': record['wall_s']}), flush=True)
            del case, vertices, faces, carved, build, elements, centers
        resources = {'wall_s': time.monotonic()-started,
                     'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        baseline = {'baseline': json.loads(args.baseline_results.read_text())} if args.baseline_results else {}
        for name, rows in outputs.items():
            save(name + '/summary.json', {'status': 'done', 'final': rows, **baseline, 'optimizer_updates': 0,
                'fit_label_times': 'evaluation_only', 'boundary': 'build-only canonical TSDF '+name+'; native mesh density; original full cohort; no external confirmation',
                'resources_boundary': 'shared construction and two readouts counted once in parent summary',
                'parent_run': str(args.output)})
        save('summary.json', {'status': 'done', 'methods': {name: str(args.output/name) for name in outputs},
                             'construction': construction, **resources, 'optimizer_updates': 0})
        save('status.json', {'status': 'done'})
        print(json.dumps({'status': 'done', 'actors': len(construction), **resources}), flush=True)
    except Exception as exc:
        save('status.json', {'status': 'failed', 'exception': type(exc).__name__, 'message': str(exc)})
        (args.output/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
