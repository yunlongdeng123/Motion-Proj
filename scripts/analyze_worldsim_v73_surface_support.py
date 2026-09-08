"""在固定表面上区分首交点错误、后方支持与仅邻近测量的表面。"""
import argparse
import json
import resource
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH


CATEGORIES = [
    'first_hit', 'early_with_later_hit', 'early_without_hit_but_near',
    'early_without_hit_or_near', 'late_but_near', 'late_without_near',
    'missing_but_near', 'missing_without_near',
]


def measure(vertices, faces, origins, directions, observed, tolerance=.2):
    count = len(observed)
    scene = SurfaceBVH(vertices, faces).scene
    first = np.full(count, np.inf, np.float32)
    nearest = np.full(count, np.inf, np.float32)
    any_hit = np.zeros(count, bool)
    early_layers = np.zeros(count, np.int64)
    layers = np.zeros(count, np.int64)
    if count and scene is not None:
        rays = o3d.core.Tensor(np.concatenate([origins, directions], -1).astype(np.float32))
        first = scene.cast_rays(rays, nthreads=2)['t_hit'].numpy()
        endpoints = (origins + directions * observed[:, None]).astype(np.float32)
        nearest = scene.compute_distance(o3d.core.Tensor(endpoints), nthreads=2).numpy()
        intersections = scene.list_intersections(rays, nthreads=2)
        ray_ids = intersections['ray_ids'].numpy().astype(np.int64)
        depth = intersections['t_hit'].numpy()
        take = np.isfinite(depth) & (depth >= 0)
        ray_ids, depth = ray_ids[take], depth[take]
        if len(depth):
            order = np.lexsort((depth, ray_ids))
            ray_ids, depth = ray_ids[order], depth[order]
            # 合并数值重合交点，计数代表沿束深度层，不代表拓扑片数。
            distinct = np.r_[True, (ray_ids[1:] != ray_ids[:-1]) | (np.diff(depth) > 1e-4)]
            layer_ids, layer_depth = ray_ids[distinct], depth[distinct]
            layers = np.bincount(layer_ids, minlength=count)
            early_ids = layer_ids[layer_depth < observed[layer_ids] - tolerance]
            early_layers = np.bincount(early_ids, minlength=count)
            good = np.abs(depth - observed[ray_ids]) <= tolerance
            any_hit[ray_ids[good]] = True
    returned = np.isfinite(first)
    early = returned & (first < observed - tolerance)
    hit = returned & (np.abs(first - observed) <= tolerance)
    late = returned & (first > observed + tolerance)
    missing = ~returned
    near = nearest <= tolerance
    masks = {
        'first_hit': hit,
        'early_with_later_hit': early & any_hit,
        'early_without_hit_but_near': early & ~any_hit & near,
        'early_without_hit_or_near': early & ~any_hit & ~near,
        'late_but_near': late & near,
        'late_without_near': late & ~near,
        'missing_but_near': missing & near,
        'missing_without_near': missing & ~near,
    }
    counters = {key: int(value.sum()) for key, value in masks.items()}
    counters.update({
        'rays': count, 'early': int(early.sum()), 'near_surface': int(near.sum()),
        'any_hit_band': int(any_hit.sum()), 'multiple_depth_layers': int((layers >= 2).sum()),
        'multiple_early_layers': int((early_layers >= 2).sum()),
        'max_depth_layers': int(layers.max()) if count else 0,
        'depth_layers_sum': int(layers.sum()),
    })
    return counters


def aggregate(rows):
    totals = defaultdict(int)
    per_log = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row['counts'].items():
            if key != 'max_depth_layers':
                totals[key] += value
        if row['counts']['rays']:
            for key, value in row['counts'].items():
                if key not in ['rays', 'max_depth_layers']:
                    per_log[row['actor']['log_id']][key].append(value / row['counts']['rays'])
    log_means = {log: {key: float(np.mean(value)) for key, value in metrics.items()}
                 for log, metrics in per_log.items()}
    keys = sorted({key for metrics in log_means.values() for key in metrics})
    return {
        'actors': len(rows), 'actors_without_owned_rays': sum(row['counts']['rays'] == 0 for row in rows),
        'empty_surfaces': sum(row['surface_faces'] == 0 for row in rows),
        'logs': len({row['actor']['log_id'] for row in rows}),
        'logs_with_owned_rays': len(log_means), 'raw_counts': dict(totals),
        'equal_log_means': {key: float(np.mean([m[key] for m in log_means.values()])) for key in keys},
        'per_log': log_means,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-data', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True, help='NAME=fixed completed run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    models = {name: Path(path) for name, path in (item.split('=', 1) for item in args.model)}
    manifest = {
        'task': 'WS-V73-M2-SURFACE-SUPPORT-01', 'code_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'actor_data': str(args.actor_data), 'models': {key: str(path) for key, path in models.items()},
        'scope': 'all 75 original development Actors; only original owned heldout first returns',
        'optimizer_updates': 0, 'neural_inference': False, 'tolerance_m': .2,
        'depth_coalescing_m': 1e-4, 'failure_ledger_refs': ['V73-F02', 'V73-F03', 'V73-F04'],
        'readout': 'existing CPU SurfaceBVH cast_rays plus list_intersections and unsigned compute_distance',
        'statistics': 'pool heldout frames within Actor; normalize each Actor by its owned rays, then mean Actors within log and mean logs; not a replacement for canonical main metrics',
        'boundary': 'no mesh edits, query deletion, new opacity, occupancy signs or target-based inference; later good intersection is diagnostic evidence, never substituted for the first return; coincident hits are not exact topology counts',
        'source': 'https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html',
    }
    save('manifest.json', manifest)
    save('status.json', {'status': 'running'})
    try:
        entries = [row for row in json.loads((args.actor_data / 'index.json').read_text())['cases']
                   if row['role'] == 'development']
        rows = {name: [] for name in models}
        for entry in entries:
            case = torch.load(args.actor_data / entry['file'], map_location='cpu', weights_only=False)
            frames = [frame for frame in case['rays'] if frame['role'] == 'heldout_time']
            for method, run in models.items():
                surface = torch.load(run / (entry['owner'] + '_surface.pt'), map_location='cpu', weights_only=False)
                vertices = surface['vertices_actor_m'].numpy()
                faces = surface['faces'].numpy()
                measured = []
                for frame_number, frame in enumerate(frames):
                    owned = frame['positive_actor'].numpy().astype(bool)
                    result = measure(vertices, faces, frame['origins_actor_m'].numpy()[owned],
                                     frame['directions_actor'].numpy()[owned],
                                     frame['observed_first_range_m'].numpy()[owned])
                    measured.append({'heldout_frame_index': frame_number, 'counts': result})
                counts = {key: sum(frame['counts'][key] for frame in measured)
                          for key in (measured[0]['counts'] if measured else CATEGORIES + [
                              'rays', 'early', 'near_surface', 'any_hit_band', 'multiple_depth_layers',
                              'multiple_early_layers', 'depth_layers_sum', 'max_depth_layers'])}
                counts['max_depth_layers'] = max((f['counts']['max_depth_layers'] for f in measured), default=0)
                rows[method].append({'actor': entry, 'surface_faces': len(faces), 'counts': counts, 'frames': measured})
            save('status.json', {'status': 'running', 'actors_done': len(rows[next(iter(models))]),
                                 'actors_total': len(entries)})
        summary = {'status': 'done', 'actors': rows,
                   'statistics': {name: aggregate(value) for name, value in rows.items()},
                   'wall_s': time.monotonic() - started,
                   'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                   'failure_ledger_delta': 'pending result interpretation'}
        save('summary.json', summary)
        save('status.json', {'status': 'done'})
        print(json.dumps({key: summary[key] for key in ['status', 'statistics', 'wall_s', 'peak_rss_gib']}), flush=True)
    except Exception as exc:
        save('status.json', {'status': 'failed', 'exception': type(exc).__name__, 'message': str(exc)})
        (args.output / 'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
