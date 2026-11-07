"""Independent CPU spot check; not a replacement for the full frozen review."""
import json
import sys
from pathlib import Path
import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B / 'scene008_v44_a5_road_fusion_integration_20261107_r1'
F = B / 'scene008_road_fusion_bake_20261107_r1'
sys.path.insert(0, str(F / 'red_review'))
from evaluate_fusion_writeback import edge_metrics
from prepare_world_masks49 import pair_summary


def load(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


registry = json.loads((F / 'red_review/world_masks49_v1/registry.json').read_text())
meta = {r['raw']: r for r in registry['frames']}
primary = load(F / 'red_review/world_masks49_v1/primary_raw128/fixed_masks_raw128.npz')
primary_world = {'local_xyz': primary['local_xyz']}
primary_world.update({e+'_'+s: primary['band_'+e+'_'+s] for e in ['near_x', 'far_x', 'lateral_y'] for s in ['minus', 'plus']})
primary_registration = pair_summary(primary_world)
rows = []
for raw in [94, 105, 128, 137]:
    obs = load(Path(meta[raw]['obs_path']))
    gt = obs['rgb'].astype(np.float32).transpose(2, 0, 1)
    if gt.max() > 1:
        gt /= 255
    a5 = obs['actor_labels'] == 1
    a6 = obs['visible'][0].astype(bool)
    a37 = obs['visible'][1].astype(bool)
    upper = np.zeros_like(a5)
    yy, xx = np.nonzero(a5)
    if len(yy):
        q = yy <= yy.min() + .25 * (yy.max() - yy.min())
        upper[yy[q], xx[q]] = True
    guard = ~obs['road'].astype(bool) & ~obs['road_edit'].astype(bool)
    protected = obs['protect'].astype(bool) & ~a6 & ~a37 & guard
    geometry_guard = guard & (obs['protect'].astype(bool) | a5 | a6 | a37)
    for state in ['keep', 'delete']:
        base = load(R / f'baseline4/frames/{raw:03d}/{state}_float.npz')
        base_err = np.abs(base['rgb'] - gt).mean(0)
        for variant in ['AA', 'C']:
            val = load(R / f'{variant}4_retry1/frames/{raw:03d}/{state}_float.npz')
            err = np.abs(val['rgb'] - gt).mean(0)
            domains = dict(a5=a5, upper=upper, a6=a6, protected_nonroad=protected, road=obs['road'].astype(bool))
            if state == 'keep':
                domains['a37'] = a37
            metrics = {k: dict(count=int(m.sum()), before=float(base_err[m].mean()) if m.any() else None,
                               after=float(err[m].mean()) if m.any() else None,
                               delta=float((err[m].astype(float)-base_err[m]).mean()) if m.any() else None,
                               RGB_max_difference=float(np.abs(val['rgb'][:, m]-base['rgb'][:, m]).max()) if m.any() else None)
                       for k, m in domains.items()}
            depth_mask = obs['depth_valid'][0].astype(bool)
            target_depth = obs['depth_m'][0]
            metrics['a6_depth_delta'] = float((np.abs(val['depth_m'][depth_mask]-target_depth[depth_mask]).astype(float)-np.abs(base['depth_m'][depth_mask]-target_depth[depth_mask])).mean()) if depth_mask.any() else None
            metrics['guard_alpha_max_difference'] = float(np.abs(val['alpha'][geometry_guard]-base['alpha'][geometry_guard]).max())
            metrics['guard_depth_max_difference'] = float(np.abs(val['depth_m'][geometry_guard]-base['depth_m'][geometry_guard]).max())
            record = dict(raw=raw, state=state, variant=variant, metrics=metrics)
            if raw == 128:
                record['primary_near'] = {k: edge_metrics(v['rgb'], gt, primary_world, primary_registration)['near_x']['equal_bin_abs_residual_jump_mean']
                                          for k, v in [('baseline', base), ('candidate', val)]}
            rows.append(record)
result = dict(status='CPU_SPOT_CHECK_NOT_FULL_ACCEPTANCE', rows=rows, native_forwards=0)
path = R / 'review/root_quick_pilot.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
