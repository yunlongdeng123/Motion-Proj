"""Read-only local artifact diagnostics against one frozen, reliable road render."""
from pathlib import Path
import json
import numpy as np

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/a20_scene_residual_cleanup_20261122_r1')
LUMA = np.array([.2126, .7152, .0722], np.float32)[:, None, None]
rois = json.loads((ROOT/'diagnostic_rois.json').read_text())
runs = ['cleanup_visible_red_r1', 'cleanup_red_black_r2_retry1', 'cleanup_dynamic_strays_r3', 'cleanup_dark_static_r4', 'cleanup_rgb_residual_r5']
report = {'reference': 'fixed actual road RGB normalized only at alpha >= .95; diagnostic, not ground truth', 'views': []}
for raw in (86, 110):
    reference = np.load(ROOT/'dark_query_r1/evidence'/f'raw{raw:03d}_actual_float_reference.npz')
    mask = reference['reliable_ground']; target = reference['road_RGB_normalized_stable']
    road_luma = reference['road_luma']; views = []
    paths = [('original', ROOT/'cleanup_dark_static_r4/evaluation'/f'raw{raw:03d}_before.npz')]
    paths += [(run, ROOT/run/'evaluation'/f'raw{raw:03d}_after.npz') for run in runs]
    for name, path in paths:
        if not path.exists(): continue
        z = np.load(path); rgb = z['rgb']; gap = road_luma-(rgb*LUMA).sum(0)
        error = np.abs(rgb-target).max(0)
        views.append({'stage': name, 'reliable_road_pixels': int(mask.sum()),
            'dark_gap_gt_08_pixels': int(((gap>.08)&mask).sum()),
            'bright_gap_gt_08_pixels': int(((gap<-.08)&mask).sum()),
            'absolute_RGB_gap_p95': float(np.percentile(error[mask], 95)),
            'diagnostic_RGB_L1': float(np.abs(rgb-target)[:, mask].mean())})
    report['views'].append({'raw': raw, 'stages': views})
(ROOT/'residual_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
