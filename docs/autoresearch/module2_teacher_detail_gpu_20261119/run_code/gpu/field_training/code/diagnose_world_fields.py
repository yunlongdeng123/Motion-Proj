"""有限 CPU 世界场诊断；不渲染、不优化，不把描述谱当质量门。"""
from pathlib import Path
import json
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
from PIL import Image

Q = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
RUN = Q / 'gpu_20261119'


def summary(value, mask):
    values = np.asarray(value)[mask]
    return dict(RMS=float(np.sqrt(np.mean(values.astype(np.float64)**2))),
                abs_max=float(np.abs(values).max()), abs_p50_p95_p99=np.quantile(np.abs(values), [.5, .95, .99]).tolist())


def derivative(value, mask, axis):
    if axis == 1:
        good = mask[:, 1:-1] & mask[:, :-2] & mask[:, 2:]
        return (value[:, 2:] - 2 * value[:, 1:-1] + value[:, :-2])[good]
    good = mask[1:-1] & mask[:-2] & mask[2:]
    return (value[2:] - 2 * value[1:-1] + value[:-2])[good]


def direction_stats(value, mask):
    out = {}
    for axis, name in ((1, 'u'), (0, 'v')):
        d = derivative(value, mask, axis).astype(np.float64)
        out[name + '_second_difference_energy'] = float(np.mean(d*d))
        out[name + '_stencil_nodes'] = len(d)
    out['v_over_u_second_difference_energy'] = out['v_second_difference_energy'] / max(out['u_second_difference_energy'], 1e-30)
    return out


def spectrum(value, mask):
    # 共同mask固定；其边界也贡献谱能量，故仅作描述，不是无限平面带限证书。
    mean = value[mask].astype(np.float64).mean(0)
    signal = (value - mean) * mask[..., None]
    p = np.abs(np.fft.rfft2(signal, axes=(0, 1)))**2
    p = p.sum(-1)
    kv = np.abs(np.fft.fftfreq(mask.shape[0]))[:, None]
    ku = np.fft.rfftfreq(mask.shape[1])[None, :]
    norm = max(float(p.sum()), 1e-30)
    # 频率按chart texel登记；名义2cm仅换算标签，不把chart当全局平面。
    radial = np.sqrt(ku**2 + kv**2)
    edges = [0, 1/12, 1/6, np.inf]
    return dict(DC_energy_fraction=float(p[0, 0]/norm),
                u_frequency_second_moment=float((p*ku**2).sum()/norm),
                v_frequency_second_moment=float((p*kv**2).sum()/norm),
                radial_band_fractions=[float(p[(radial>=lo)&(radial<hi)].sum()/norm) for lo, hi in zip(edges[:-1], edges[1:])],
                bands_cycles_per_texel=['[0,1/12)', '[1/12,1/6)', '[1/6,Nyquist-domain]'])


def save_rgb(path, value, valid):
    rgb = np.rint(np.clip(value, 0, 1)*255).astype(np.uint8)
    rgba = np.concatenate((rgb, (valid[..., None]*255).astype(np.uint8)), -1)
    Image.fromarray(rgba).save(path)


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    out = RUN / 'field_training/world_field_diagnostic_r1'
    out.mkdir(parents=True, exist_ok=False)
    result = dict(description='world-field diagnostics only; no new quality thresholds',
        axes='u=chart columns, v=chart rows; these are not camera image horizontal/vertical',
        spectral_boundary='same common reg mask for every stage; mask boundaries contribute spectrum',
        display=dict(low_delta_limit=.30, effective_H_display_limit=.04,
                     display_clipping_is_not_model_bound=True, model_H_bound=.34), rows={})
    previous = {}
    for arm, kind in [('M11', 'pixel'), ('M10', 'avg')]:
        previous[arm] = {}
        for label, path in [('A200', RUN/'formal_r1'/('%s_common_A200'%kind)/'field.pt'),
                            ('B500', RUN/'formal_r1'/arm/'B500/field.pt'),
                            ('C800', RUN/'formal_r1'/arm/'C800/field.pt')]:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            state, settings = saved['state'], saved['settings']
            d = {key: value.numpy() for key, value in state.items()}
            base = d['base']; valid = d['valid'][..., 0]>0
            R = d['reg_support'][..., 0]>0; latent = d['latent_mask'][..., 0]>0
            low = settings['low_bound']*np.tanh(d['low'])*d['low_mask']
            raw = np.tanh(d['high'])*d['latent_mask']
            smooth = np.stack([gaussian_filter(raw[..., c], 3., mode='nearest', truncate=3.) for c in range(3)], -1)
            high = .5*settings['high_bound']*(raw-smooth)
            effective = d['H_gate']*high
            yy, xx = np.indices(valid.shape)
            low_fine = np.stack([map_coordinates(low[..., c], [yy/6, xx/6], order=1, mode='constant', prefilter=False) for c in range(3)], -1)
            combined_unclamped = base+low_fine+effective
            combined = np.clip(combined_unclamped, 0, 1)
            folder = out/arm/label; folder.mkdir(parents=True)
            save_rgb(folder/'A0.png', base, valid)
            save_rgb(folder/'low_delta_same_scale.png', .5+.5*low_fine/.30, valid)
            save_rgb(folder/'effective_H_same_scale.png', .5+.5*effective/.04, valid)
            save_rgb(folder/'combined.png', combined, valid)
            np.savez_compressed(folder/'world_fields.npz', low=low, effective_H=effective.astype(np.float32),
                                combined=combined.astype(np.float32), common_support=R)
            row = dict(field=str(path), low_nodes=int(d['low_mask'].sum()), high_latent_nodes=int(latent.sum()),
                common_reg_nodes=int(R.sum()), L=summary(low, d['low_mask'][..., 0]>0),
                H_unmodulated=summary(high, R), effective_H=summary(effective, R),
                low_tanh_ge_095_fraction=float((np.abs(np.tanh(d['low']))[d['low_mask'][..., 0]>0]>=.95).mean()),
                high_tanh_ge_095_fraction=float((np.abs(np.tanh(d['high']))[latent]>=.95).mean()),
                high_tanh_ge_099_fraction=float((np.abs(np.tanh(d['high']))[latent]>=.99).mean()),
                effective_H_weighted_DC=effective[R].mean(0).tolist(),
                A0_directions=direction_stats(base, R), L_directions=direction_stats(low_fine, R),
                H_directions=direction_stats(effective, R), combined_directions=direction_stats(combined, R),
                H_spectrum=spectrum(effective, R), clamp_fraction=float(((combined_unclamped<0)|(combined_unclamped>1))[R].mean()))
            row['world_second_difference_energy_decomposition'] = {}
            for axis, name in ((1, 'u'), (0, 'v')):
                a, l, h = [derivative(value, R, axis).astype(np.float64) for value in (base, low_fine, effective)]
                ea, el, eh = [float(np.mean(x*x)) for x in (a, l, h)]
                crosses = dict(A0_L=2*float(np.mean(a*l)), A0_H=2*float(np.mean(a*h)), L_H=2*float(np.mean(l*h)))
                total = float(np.mean((a+l+h)**2))
                row['world_second_difference_energy_decomposition'][name] = dict(A0=ea, L=el, H=eh,
                    cross_terms=crosses, unclamped_total=total, identity_error=abs(total-ea-el-eh-sum(crosses.values())))
            if 'A200' in previous[arm]:
                old = previous[arm]['A200']
                row['drift_from_A200'] = dict(low_raw_max=float(np.max(np.abs(d['low']-old['low']))),
                    high_raw_max=float(np.max(np.abs(d['high']-old['high']))), effective_H_RMS=float(np.sqrt(np.mean((effective-old['effective'])[R]**2))))
            if label == 'C800':
                old = previous[arm]['B500']
                row['drift_from_B500'] = dict(low_raw_max=float(np.max(np.abs(d['low']-old['low']))),
                    high_raw_max=float(np.max(np.abs(d['high']-old['high']))), effective_H_RMS=float(np.sqrt(np.mean((effective-old['effective'])[R]**2))))
            previous[arm][label] = dict(low=d['low'].copy(), high=d['high'].copy(), effective=effective.copy())
            result['rows'][arm+'_'+label] = row
    result.update(seconds=time.perf_counter()-start, native_forward_calls=0, optimizer_updates=0, GPU_operations=0,
                  numerical_fields='NumPy/SciPy same separable Gaussian formula, not additional native render',
                  no_energy_additivity_claim_without_cross_terms=True)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(out/'results.json'), seconds=result['seconds'], native_calls=0, optimizer_updates=0)))


if __name__ == '__main__':
    main()
