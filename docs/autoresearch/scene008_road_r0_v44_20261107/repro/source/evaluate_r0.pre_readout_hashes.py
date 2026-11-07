"""Real paired R0 rendering with fixed observation-derived evaluation domains."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def summary(array, mask):
    import numpy as np
    a = np.asarray(array, np.float64)[mask]
    return dict(pixels=len(a), sum=float(a.sum()), mean=float(a.mean()) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None, max=float(a.max()) if len(a) else None)


def image(path, rgb):
    import numpy as np
    from PIL import Image
    array = np.asarray(rgb)
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    Image.fromarray(np.uint8(np.clip(array, 0, 1)*255+.5)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=['pilot', 'full49'], default='pilot')
    ap.add_argument('--retirement', type=Path)
    ap.add_argument('--material', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_snapshot = args.output/'source_snapshot'
    source_snapshot.mkdir()
    source_identities = {}
    for file_name in ('evaluate_r0.py', 'r0_runtime.py'):
        source = Path(__file__).resolve().parent/file_name
        shutil.copy2(source, source_snapshot/file_name)
        source_identities[file_name] = hashlib.sha256(source.read_bytes()).hexdigest()
    import numpy as np
    import torch
    from scipy.ndimage import binary_dilation
    from r0_runtime import R0Runtime, C
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.55)
    start = time.time()
    model = R0Runtime(args.asset, retirement=args.retirement, material=args.material)
    rt = model.rt
    raws = [105, 113, 130, 137] if args.mode == 'pilot' else list(range(89, 138))
    variants = ['baseline', 'extension'] + (['calibrated'] if args.material else []) + (['replacement'] if args.retirement else [])
    records, baseline_parity = [], []
    for raw in raws:
        data = rt.raw(raw)
        obs = data['obs']
        rgb_gt = np.asarray(obs['rgb'], np.float32).transpose(2, 0, 1)
        if rgb_gt.max() > 1:
            rgb_gt /= 255
        view_path = rt.road_run/'inputs/views'/f'{raw:03d}.npz'
        with np.load(view_path) as z:
            other_guard = z['other_vehicle_guard'].copy()
        masks = dict(a6=obs['visible'][0].astype(bool), a37=obs['visible'][1].astype(bool),
                     measured_a6=obs['depth_valid'][0].astype(bool), road=obs['road'].astype(bool),
                     measured_road=obs['road'].astype(bool)&(obs['scene_depth_m']>0),
                     protected=obs['protect'].astype(bool)&~obs['visible'].any(0),
                     outside=obs['protect'].astype(bool)&~obs['visible_reviewed'].any(0)&~obs['box_candidate'].any(0))
        fixed = np.zeros((320, 480), bool)
        fixed[204:222, 230:270] = True
        silver = np.zeros_like(fixed)
        if raw == 137:
            silver[213:233, 210:290] = True
        elif raw == 130:
            silver[192:244, 202:298] = True
        folder = args.output/f'{raw:03d}'
        folder.mkdir()
        image(folder/'observed.png', rgb_gt)
        with torch.no_grad():
            solo_out = model.render(raw, solo=True)
            solo = solo_out['alpha'][0].cpu().numpy()
            solo_depth = (solo_out['depth_sum'][0]/solo_out['alpha'][0].clamp_min(1e-8)/rt.scale).cpu().numpy()
            # This is a declared mechanism domain from measured road geometry,
            # not a hidden-GT accuracy mask. Known a6 and other vehicles excluded.
            eligible = (solo >= .98)&~binary_dilation(obs['box_candidate'][0], iterations=2)&~other_guard
            image(folder/'extension_solo_alpha.png', solo)
            image(folder/'extension_solo_rgb.png', solo_out['rgb'].cpu().numpy())
            del solo_out
            by_state = {}
            for state in ('keep', 'delete'):
                by_variant, outputs = {}, {}
                for variant in variants:
                    out = model.render(raw, delete=(state == 'delete'), extension=variant != 'baseline',
                                       retire=variant == 'replacement', material=variant == 'calibrated')
                    rgb = out['rgb'].cpu().numpy()
                    alpha = out['alpha'][0].cpu().numpy()
                    weights = out['group_alpha_T'].cpu().numpy()
                    ext = out['extension_alpha_T'].cpu().numpy()
                    depth = (out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/rt.scale).cpu().numpy()
                    error = np.abs(rgb-rgb_gt).mean(0)
                    domains = dict(fixed720=fixed, silver_fixed=silver,
                                   measured_extension_rays=eligible)
                    item = dict(a6_RGB=summary(error, masks['a6']),
                        a6_depth_m=summary(abs(depth-obs['depth_m'][0]), masks['measured_a6']),
                        observed_road_depth_m=summary(abs(depth-obs['scene_depth_m']), masks['measured_road']),
                        observed_road_RGB=summary(error, masks['road']),
                        protected_RGB=summary(error, masks['protected']),
                        a6_alpha=summary(weights[0], masks['a6']),
                        alpha_conservation_maxabs=out['group_conservation_maxabs'],
                        RGB_conservation_maxabs=out['color_reconstruction_maxabs'],
                        native_asset_rows_hash=out['meta']['native_asset_row_hash'], domains={})
                    if state == 'keep':
                        item['a37_RGB'] = summary(error, masks['a37'])
                    for label, domain in domains.items():
                        item['domains'][label] = dict(
                            extension_solo_alpha=summary(solo, domain),
                            alpha=summary(alpha, domain), road_alphaT=summary(weights[2], domain),
                            static_alphaT=summary(weights[3], domain), asset_alphaT=summary(weights[0], domain),
                            a37_alphaT=summary(weights[1], domain), environment_T=summary(1-alpha, domain),
                            extension_alphaT=summary(ext, domain),
                            extension_visibility_ratio=summary(ext/np.maximum(solo, 1e-8), domain),
                            road_only_depth_residual_m=summary(abs(depth-solo_depth), domain))
                    if variant == 'baseline':
                        old = C/'scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44'/f'{raw:03d}'/(state+'_float.npz')
                        with np.load(old) as z:
                            delta = float(abs(rgb-z['rgb']).max())
                            assert delta < 2e-6, ('V44 baseline differs', raw, state, delta)
                        baseline_parity.append(dict(raw=raw, state=state, maxabs=delta))
                    if args.material and variant == 'extension' and raw in (105,113,130,137):
                        previous = C/'scene008_road_r0_v44_20261107_r1/results/full49_v2'/f'{raw:03d}'/('extension_'+state+'_float.npz')
                        with np.load(previous) as z:
                            delta = float(abs(rgb-z['rgb']).max())
                            assert delta == 0., ('Uncalibrated numeric path changed', raw, state, delta)
                        item['uncalibrated_v2_saved_float_maxabs'] = delta
                    if variant != 'baseline':
                        b_rgb, b_depth = outputs['baseline']
                        item['versus_baseline'] = {name:dict(
                            RGB_change=summary(abs(rgb-b_rgb).mean(0), m),
                            RGB_max_channel_gt002=int((abs(rgb-b_rgb).max(0)[m] > .02).sum()),
                            depth_change_m=summary(abs(depth-b_depth), m))
                            for name,m in masks.items() if name not in ('measured_a6','measured_road')}
                    image(folder/(variant+'_'+state+'.png'), rgb)
                    image(folder/(variant+'_'+state+'_road_alphaT.png'), weights[2])
                    if raw in [105, 113, 130, 137]:
                        np.savez_compressed(folder/(variant+'_'+state+'_float.npz'), rgb=rgb, alpha=alpha,
                            depth_m=depth, group_alpha_T=weights, extension_alpha_T=ext)
                    outputs[variant] = (rgb, depth)
                    by_variant[variant] = item
                    del out
                by_state[state] = by_variant
                for variant in variants[1:]:
                    assert by_variant[variant]['native_asset_rows_hash'] == by_variant['baseline']['native_asset_rows_hash']
                del outputs
            records.append(dict(raw=raw, states=by_state, mechanism_domain_pixels=int(eligible.sum())))
        write(folder/'metrics.json', records[-1])
        write(args.output/'progress.json', dict(completed=[r['raw'] for r in records]))
        rt._data.clear()
        torch.cuda.empty_cache()
        print(json.dumps(dict(raw=raw, completed=len(records), seconds=time.time()-start)), flush=True)
    pooled = []
    for state in ('keep','delete'):
        for variant in variants:
            for metric in ('a6_RGB','a6_depth_m','observed_road_depth_m','observed_road_RGB','protected_RGB','a37_RGB'):
                values=[r['states'][state][variant][metric] for r in records if metric in r['states'][state][variant]]
                n=sum(x['pixels'] for x in values)
                if values:
                    pooled.append(dict(state=state, variant=variant, metric=metric, pixels=n,
                        mean=sum(x['sum'] for x in values)/n if n else None))
    write(args.output/'receipt.json', dict(status='R0_PAIRED_MECHANISM_RESULTS_NOT_HUMAN_ACCEPTED',
        mode=args.mode, raws=raws, variants=variants, seconds=time.time()-start,
        executed_source_sha256=source_identities,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), runtime=model.receipt(),
        baseline_parity=baseline_parity, pooled=pooled, frames=records,
        independent_hidden_truth=False, learned_prior=False, optimizer_updates=0, shutdown_performed=False))


if __name__ == '__main__':
    main()
