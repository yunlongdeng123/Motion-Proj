"""Read saved J/21K arrays: exact old domains, diagnostic worst pixels and context."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fixed_domains import fixed_domains

GROUPS = ['asset_C2', 'native_a37', 'oldroad', 'native_static',
          'semantic_a5', 'unused18', 'unused26', 'newroad']
ACTORS = {0: None, 1: 5, 2: 26, 3: 18, 4: 6, 5: 37}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def image(rgb):
    return Image.fromarray(np.asarray(np.clip(rgb, 0, 1) * 255, np.uint8))


def role_counts(roles, selected):
    return {name: int((mask & selected).sum()) for name, mask in roles.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--fixed-reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Do not overwrite a prior audit')
    a.output.mkdir(parents=True)
    ground = load(a.fixed_reference)['depth_m'][:, 3] > 0
    inputs = [a.fixed_reference, Path(__file__), Path(__file__).with_name('fixed_domains.py')]
    sources = {}
    records = []
    font = ImageFont.load_default()
    for raw in [128, 137]:
        op = a.observations / f'{raw:03d}.npz'
        bp = a.root / f'baseline4/frames/{raw:03d}/delete_float.npz'
        jp = a.root / f'J4_road_parent_operator/lp_candidate_render_v1/frames/{raw:03d}/delete_float.npz'
        for path in [op, bp, jp]:
            sources[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                                  'mtime_ns': path.stat().st_mtime_ns}
        obs, base, j = load(op), load(bp), load(jp)
        gt = np.asarray(obs['rgb'], np.float64)
        assert gt.shape == (320, 480, 3)
        b = np.moveaxis(base['rgb'], 0, -1).astype(np.float64)
        c = np.moveaxis(j['rgb'], 0, -1).astype(np.float64)
        db = fixed_domains(obs)
        labels = np.asarray(obs['actor_labels']).squeeze()
        roles = {**db, 'full_actor37_label5': labels == 5,
                 'any_actor_label': labels > 0, 'ground_beneath_a37_proxy': ground[raw - 89]}
        weights_b = np.asarray(base['group_alpha_T'])
        if len(weights_b) == 7:
            weights_b = np.concatenate([weights_b, np.zeros_like(weights_b[:1])])
        weights_j = np.asarray(j['group_alpha_T'])
        assert weights_b.shape == weights_j.shape == (8, 320, 480)
        metrics = [
            ('protected_nonroad_error_increase', db['protected_nonroad'],
             np.abs(c - gt).mean(2) - np.abs(b - gt).mean(2), 'RGB MAE increase vs observed'),
            ('originalroad_error_increase', db['road'],
             np.abs(c - gt).mean(2) - np.abs(b - gt).mean(2), 'RGB MAE increase vs observed'),
            ('guard_alpha_abs_difference', db['fixed_nonroad_geometry_guard'],
             abs(j['alpha'] - base['alpha']), 'alpha absolute change vs latest21K'),
            ('guard_depth_abs_difference', db['fixed_nonroad_geometry_guard'],
             abs(j['depth_m'] - base['depth_m']), 'metres absolute depth change vs latest21K'),
        ]
        for name, domain, score, meaning in metrics:
            valid = domain & np.isfinite(score)
            flat = np.flatnonzero(valid)
            ordered = flat[np.argsort(-score.ravel()[flat], kind='stable')[:20]]
            selected = np.zeros((320, 480), bool)
            selected.ravel()[ordered] = True
            points = []
            for rank, index in enumerate(ordered, 1):
                y, x = np.unravel_index(index, (320, 480))
                points.append({
                    'rank': rank, 'pixel_xy': [int(x), int(y)], 'score': finite(score[y, x]),
                    'actor_label': int(labels[y, x]), 'source_instance_actor_id': ACTORS.get(int(labels[y, x])),
                    'original_roles': {k: bool(v[y, x]) for k, v in roles.items()},
                    'observed_RGB': gt[y, x].tolist(), 'latest21K_RGB': b[y, x].tolist(), 'J_RGB': c[y, x].tolist(),
                    'latest21K_observed_RGB_error': finite(abs(b[y, x] - gt[y, x]).mean()),
                    'J_observed_RGB_error': finite(abs(c[y, x] - gt[y, x]).mean()),
                    'latest21K_alpha': finite(base['alpha'][y, x]), 'J_alpha': finite(j['alpha'][y, x]),
                    'latest21K_depth_m': finite(base['depth_m'][y, x]), 'J_depth_m': finite(j['depth_m'][y, x]),
                    'latest21K_group_alpha_T': dict(zip(GROUPS, map(float, weights_b[:, y, x]))),
                    'J_group_alpha_T': dict(zip(GROUPS, map(float, weights_j[:, y, x]))),
                    'latest21K_dominant_source_group': GROUPS[int(weights_b[:, y, x].argmax())],
                    'J_dominant_source_group': GROUPS[int(weights_j[:, y, x].argmax())],
                })
            x0, y0 = points[0]['pixel_xy']
            crop = (max(0, x0 - 24), max(0, y0 - 18), min(480, x0 + 25), min(320, y0 + 19))
            board = Image.new('RGB', (1500, 650), '#f4f7f8')
            draw = ImageDraw.Draw(board)
            draw.text((15, 8), f'raw{raw} DELETE / {name} / original domain; diagnostic top20 only', font=font, fill='#173b4a')
            for col, (label, rgb) in enumerate([('Observed O (not hidden delete GT)', gt), ('Delivered21K', b), ('Actual J', c)]):
                draw.text((15 + col * 495, 30), label, font=font, fill='#173b4a')
                native = image(rgb)
                marked = native.copy()
                pen = ImageDraw.Draw(marked)
                for point in points:
                    x, y = point['pixel_xy']
                    pen.ellipse((x - 2, y - 2, x + 2, y + 2), outline='#f4409e')
                board.paste(marked, (15 + col * 495, 48))
                zoom = native.crop(crop).resize((294, 222), Image.Resampling.NEAREST)
                board.paste(zoom, (15 + col * 495, 402))
                draw.text((15 + col * 495, 380), f'Unmarked same crop xyxy={crop}; nearest display only', font=font, fill='#173b4a')
            figure = a.output / f'raw{raw}_delete_{name}.png'
            board.save(figure)
            row = {'raw': raw, 'state': 'delete', 'metric': name, 'meaning': meaning,
                   'original_domain_pixels': int(domain.sum()), 'finite_domain_pixels': int(valid.sum()),
                   'positive_score_pixels': int((valid & (score > 0)).sum()),
                   'original_domain_role_intersections': role_counts(roles, domain),
                   'top20_role_intersections': role_counts(roles, selected),
                   'top20': points, 'display_crop_xyxy': list(crop),
                   'figure': {'path': str(figure), 'sha256': sha(figure), 'size': [1500, 650]}}
            records.append(row)
    for path in inputs:
        sources[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                              'mtime_ns': path.stat().st_mtime_ns}
    assert all(sha(Path(path)) == value['sha256'] and Path(path).stat().st_mtime_ns == value['mtime_ns']
               for path, value in sources.items())
    report = {'schema': 'J_SAVED_PHOTO_GEOMETRY_FAILURE_PIXELS_CPU_V1',
              'status': 'READ_ONLY_TOP_PIXEL_AND_ROLE_DIAGNOSTIC_OLD_GATES_UNCHANGED',
              'records': records, 'source_integrity': {'status': 'PASS_BYTES_MTIME_SHA_UNCHANGED', 'files': sources},
              'new_GPU_forwards': 0, 'new_solver_calls': 0, 'new_optimizer_updates': 0,
              'quality_accepted': False, 'human_verdict': None,
              'limits': ['Original protected_nonroad/road/geometry domains are not redefined or rescored.',
                         'Top20 and display crops are diagnostic selections; original complete-domain failures remain.',
                         'Observed contains the source a37 and is not ground truth for the hidden DELETE road.',
                         'Instance/source-group annotations do not prove current 3D semantic truth; depth changes do not alone identify invalid geometry.']}
    output = a.output.parent / (a.output.name + '.json')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'records': len(records), 'new_GPU_forwards': 0,
                      'top': [{'raw': r['raw'], 'metric': r['metric'], 'point': r['top20'][0]} for r in records]}))


if __name__ == '__main__':
    main()
