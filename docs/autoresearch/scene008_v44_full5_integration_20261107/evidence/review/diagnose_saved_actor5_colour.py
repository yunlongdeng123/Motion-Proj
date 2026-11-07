"""Two-frame CPU-only diagnostic. Never changes the registered metric domains."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
FULL5 = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1/formal10k_keep_all49')
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44')
OBS = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_instances_20261105_r1/observations/frames')
GROUPS = ['asset', 'a37', 'road', 'native_static', 'a5', 'a18', 'a26']
sources = {}

def read(path):
    b = path.read_bytes()
    sources[str(path)] = {'bytes': len(b), 'mtime_ns': path.stat().st_mtime_ns,
                          'sha256': hashlib.sha256(b).hexdigest()}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}

def rgb_summary(rgb, gt):
    rgb = np.asarray(rgb, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return {'pixels': len(rgb), 'mean_rgb': rgb.mean(0).tolist(),
            'gt_mean_rgb': gt.mean(0).tolist(),
            'mae_fixed_domain': float(np.abs(rgb - gt).mean()),
            'mean_signed_rgb_error': (rgb - gt).mean(0).tolist()}

def region_summary(mask, mixed, base, native, gt, alpha7, rgb5T, native_alpha):
    mixed, base, native, gt = [x[mask] for x in [mixed, base, native, gt]]
    alpha7 = alpha7[mask].astype(np.float64)
    rgb5T = rgb5T[mask].astype(np.float64)
    native_alpha = native_alpha[mask].astype(np.float64)
    a5 = alpha7[:, 4]
    valid = a5 > 1e-6
    ratio = rgb5T[valid] / a5[valid, None]
    diagnostics = {}
    for threshold in [1e-6, .1, .5, .9]:
        select = a5 > threshold
        diagnostics[str(threshold)] = {
            'valid_pixels': int(select.sum()), 'fraction_of_original_region': float(select.mean()),
            'effective_colour_rgbT_over_alphaT': rgb_summary(rgb5T[select] / a5[select, None], gt[select]) if select.any() else None,
            'mixed_final_same_valid_pixels': rgb_summary(mixed[select], gt[select]) if select.any() else None,
            'native_full5_final_same_valid_pixels': rgb_summary(native[select], gt[select]) if select.any() else None,
        }
    return {
        'fixed_domain_pixels': len(gt), 'mixed_final': rgb_summary(mixed, gt),
        'v44_baseline_final': rgb_summary(base, gt), 'original_full5_keep_final': rgb_summary(native, gt),
        'mixed_mean_alphaT_by_group': dict(zip(GROUPS, alpha7.mean(0).tolist())),
        'mixed_owner5_rgbT_mean': rgb5T.mean(0).tolist(),
        'mixed_owner5_alpha_weighted_effective_colour': (rgb5T.sum(0) / a5.sum()).tolist() if a5.sum() > 0 else None,
        'mixed_owner5_effective_colour_valid_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(ratio, [.1, .5, .9], axis=0).tolist())) if len(ratio) else None,
        'native_full5_mean_alphaT_by_channel': dict(zip(['static', '5', '6', '18', '26', '37'], native_alpha.mean(0).tolist())),
        'effective_alpha_diagnostics_only': diagnostics,
        'mixed_static_alpha_exceeds_owner5_count': int((alpha7[:, 3] > a5).sum()),
    }

frames = []
for raw in [94, 105]:
    obs = read(OBS / f'{raw:03d}.npz')
    full = read(FULL5 / f'a5_float/{raw:03d}.npz')
    yx = np.column_stack(np.nonzero(obs['actor_labels'] == 1)).astype(np.int16)
    assert np.array_equal(yx, full['yx'])
    gt = obs['rgb'][yx[:, 0], yx[:, 1]]
    assert np.array_equal(gt, full['gt_rgb'])
    assert full['actor_channel_ids'].tolist() == [0, 5, 6, 18, 26, 37]
    upper = yx[:, 0] <= yx[:, 0].min() + .25 * (yx[:, 0].max() - yx[:, 0].min())
    assert np.array_equal(upper, full['upper_selected'])
    for state in ['keep', 'delete']:
        cand = read(ROOT / f'candidate_a5_49/frames/{raw:03d}/{state}_float.npz')
        baseline = read(BASE / f'{raw:03d}/{state}_float.npz')
        mix_rgb = cand['rgb'][:, yx[:, 0], yx[:, 1]].T
        base_rgb = baseline['rgb'][:, yx[:, 0], yx[:, 1]].T
        groups = cand['group_alpha_T'][:, yx[:, 0], yx[:, 1]].T
        rgbT = cand['added_owner_rgb_T'][0, :, yx[:, 0], yx[:, 1]]
        if rgbT.shape != gt.shape:
            rgbT = rgbT.T
        assert rgbT.shape == gt.shape
        row = {'raw': raw, 'state': state, 'fixed_gt_and_yx_exact': True}
        for name, region in [('full', np.ones(len(yx), dtype=bool)), ('upper', upper)]:
            row[name] = region_summary(region, mix_rgb, base_rgb, full['keep_rgb'], gt, groups, rgbT, full['joint_alphaT'])
        frames.append(row)

for path, before in sources.items():
    p = Path(path)
    assert p.stat().st_mtime_ns == before['mtime_ns']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before['sha256']
result = {
    'status': 'PASS_TWO_FRAME_SAVED_FLOAT_CPU_DIAGNOSTIC', 'raw_ids': [94, 105],
    'gpu_forwards': 0, 'optimization_updates': 0, 'fixed_main_domains_unchanged': True,
    'effective_colour_note': 'RGB_T/alpha_T is the depth-weighted effective colour of the added owner in this mixed sort. Alpha filtering is diagnostic only and does not replace full or upper metric domains.',
    'comparison_note': 'Original Full5 is its saved KEEP output, compared against the exact same fixed a5 pixels and GT for both mixed states. All RGB values are consumed as saved, without an additional clamp.',
    'causal_conclusion': None, 'frames': frames,
    'source_integrity': {'status': 'PASS_UNCHANGED_BYTES_MTIME_SHA256', 'files': sources},
}
out = ROOT / 'review/two_frame_added_owner_colour.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'output': str(out), 'frames': [
    {'raw': x['raw'], 'state': x['state'], 'mixed_MAE': x['full']['mixed_final']['mae_fixed_domain'],
     'native_full5_MAE': x['full']['original_full5_keep_final']['mae_fixed_domain'],
     'alpha7': x['full']['mixed_mean_alphaT_by_group'],
     'added5_effective_rgb': x['full']['mixed_owner5_alpha_weighted_effective_colour'],
     'mixed_final_rgb': x['full']['mixed_final']['mean_rgb'],
     'native_final_rgb': x['full']['original_full5_keep_final']['mean_rgb'],
     'gt_rgb': x['full']['mixed_final']['gt_mean_rgb']}
    for x in frames]}))
