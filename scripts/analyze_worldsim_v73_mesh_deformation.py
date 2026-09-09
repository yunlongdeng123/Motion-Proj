"""读取保存的Q-v2网格，区分局部形变与非邻接三角相交；不修改表面。"""
import argparse
import json
import resource
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [('min', 0), ('p05', .05), ('p50', .5), ('p95', .95), ('max', 1)]


def deformation(vertices, faces, reference):
    tri, ref = vertices[faces], reference[faces]
    b1, b2 = ref[:, 1] - ref[:, 0], ref[:, 2] - ref[:, 0]
    edge1 = np.linalg.norm(b1, axis=-1)
    u = b1 / edge1[:, None]
    projection = (b2 * u).sum(-1)
    height = np.linalg.norm(b2 - projection[:, None] * u, axis=-1)
    basis = np.zeros((len(faces), 2, 2), dtype=np.float64)
    basis[:, 0, 0], basis[:, 0, 1], basis[:, 1, 1] = edge1, projection, height
    actual_edges = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], axis=-1)
    # 3×2映射的奇异值对整体刚体转动不变；参考是解析模板，不是表面真值。
    jacobian = actual_edges @ np.linalg.inv(basis)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    smax, smin = singular[:, 0], singular[:, 1]
    area_ratio = smax * smin
    anisotropy = smax / np.maximum(smin, 1e-12)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=-1)
    initial_lengths = np.linalg.norm(reference[edges[:, 1]] - reference[edges[:, 0]], axis=-1)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    pairs = np.asarray(mesh.get_self_intersecting_triangles()).astype(np.int64).reshape(-1, 2)
    intersected = np.unique(pairs)
    raw = {'smin': smin, 'smax': smax, 'area_ratio': area_ratio, 'anisotropy': anisotropy,
           'edge_ratio': lengths / initial_lengths, 'edge_length_m': lengths,
           'intersection_pairs': pairs}
    stats = {}
    for key, values in raw.items():
        if key != 'intersection_pairs':
            stats.update({key + '_' + label: float(np.quantile(values, quantile)) for label, quantile in QUANTILES})
    ref_area = edge1 * height / 2
    stats.update({
        'smin_below_01_fraction': float(np.mean(smin < .1)),
        'smax_above_3_fraction': float(np.mean(smax > 3)),
        'area_ratio_below_01_fraction': float(np.mean(area_ratio < .1)),
        'anisotropy_above_10_fraction': float(np.mean(anisotropy > 10)),
        'total_area_ratio': float((ref_area * area_ratio).sum() / ref_area.sum()),
        'nonadjacent_intersection_pairs': len(pairs),
        'intersected_face_fraction': len(intersected) / len(faces),
        'any_nonadjacent_intersection': int(len(pairs) > 0),
    })
    return stats, raw


def aggregate(rows):
    logs = defaultdict(list)
    for row in rows:
        if row['metrics'] is not None:
            logs[row['actor']['log_id']].append(row['metrics'])
    per_log = {key: {metric: float(np.mean([x[metric] for x in values])) for metric in values[0]}
               for key, values in logs.items()}
    return {'actors': len(rows), 'nonempty_actors': sum(x['metrics'] is not None for x in rows),
            'empty_actors': sum(x['metrics'] is None for x in rows),
            'actors_with_nonadjacent_intersection': sum(bool(x['metrics'] and x['metrics']['any_nonadjacent_intersection']) for x in rows),
            'logs_with_nonempty': len(per_log), 'per_log': per_log,
            'equal_log_means_of_actor_statistics': {metric: float(np.mean([x[metric] for x in per_log.values()]))
                                                    for metric in next(iter(per_log.values()))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-data', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    models = {name: Path(path) for name, path in (item.split('=', 1) for item in args.model)}

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    manifest = {'task_id': 'WS-V73-Q-V2-MESH-DIAGNOSTIC-01', 'run_id': args.output.name,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'config': {'actor_data': str(args.actor_data), 'models': {k: str(v) for k, v in models.items()}},
                'failure_ledger_refs': ['V73-F02', 'V73-F03', 'V73-F04'],
                'scope': 'all 75 original development Actors; saved final meshes only; no optimization or neural inference',
                'reference': 'original checkpoint template_vertices multiplied by read-only Actor size/2; neither GT nor initial neural output',
                'statistics': 'face/edge quantiles per nonempty Actor, then mean Actors per log and mean logs; eight empty predictions retained as unavailable for distortion, not zero distortion',
                'interpretation': 'descriptive distortion bins are not quality gates; 3x2 singular values invariant under rigid rotations; anisotropy denominator numerical floor 1e-12',
                'intersection': 'Open3D0.19 get_self_intersecting_triangles skips pairs sharing ANY vertex; does not certify absence of adjacent folding or causal physics attribution',
                'sources': ['https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/geometry/TriangleMesh.cpp',
                            'https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh'],
                'failure_ledger_delta': 'pending interpretation'}
    save('manifest.json', manifest)
    save('status.json', {'status': 'running'})
    try:
        templates = {}
        for name, run in models.items():
            state = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
            templates[name] = state['query_decoder']['template_vertices'].numpy().astype(np.float64)
            del state
            (args.output / name).mkdir()
        entries = [x for x in json.loads((args.actor_data / 'index.json').read_text())['cases'] if x['role'] == 'development']
        rows = {name: [] for name in models}
        for entry in entries:
            case = torch.load(args.actor_data / entry['file'], map_location='cpu', weights_only=False)
            size = case['size_lwh_m'].numpy().astype(np.float64)
            for name, run in models.items():
                surface = torch.load(run / (entry['owner'] + '_surface.pt'), map_location='cpu', weights_only=False)
                vertices = surface['vertices_actor_m'].numpy().astype(np.float64)
                faces = surface['faces'].numpy().astype(np.int64)
                stats = None
                if len(faces):
                    stats, raw = deformation(vertices, faces, templates[name] * size / 2)
                    np.savez_compressed(args.output / name / (entry['owner'] + '.npz'), **raw)
                rows[name].append({'actor': entry, 'vertices': len(vertices), 'faces': len(faces), 'metrics': stats})
            save('status.json', {'status': 'running', 'actors_done': len(rows[next(iter(models))]), 'actors_total': len(entries)})
        summary = {'status': 'done', 'actors': rows, 'statistics': {k: aggregate(v) for k, v in rows.items()},
                   'wall_s': time.monotonic() - started,
                   'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                   'failure_ledger_delta': 'pending interpretation'}
        save('summary.json', summary)
        save('status.json', {'status': 'done'})
        print(json.dumps({k: summary[k] for k in ['status', 'statistics', 'wall_s', 'peak_rss_gib']}), flush=True)
    except Exception as exc:
        save('status.json', {'status': 'failed', 'exception': type(exc).__name__, 'message': str(exc)})
        (args.output / 'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
