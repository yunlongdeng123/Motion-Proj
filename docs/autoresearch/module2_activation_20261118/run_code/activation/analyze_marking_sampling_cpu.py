"""只读既有理想教师/检查点，诊断细标线采样约定；不改变教师或门槛。"""
from pathlib import Path
import importlib.util
import json
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
CPU = CAMPAIGN / 'road_capacity_cpu_20261117_r1'
OLD = CAMPAIGN / 'road_capacity_gpu_20261118_r1/check2'
OUT = CAMPAIGN / 'road_activation_20261118_r1/qa'
ASSET = CAMPAIGN / 'road_asset_20261116_r1/phase2/formal_r1/asset'


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (keys if keys else z.files)}


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), np.quantile(values, [0, .1, .5, .9, 1]).tolist())) if values.size else None


def world_covariance(road, rows):
    """复现固定RoadRows最短旋转的静态协方差；不是光栅或可见性贡献。"""
    rotation = road['model_from_world'][:3, :3] / float(road['meter_to_model'])
    normal = road['normal_world'][rows] @ rotation.T
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    q = np.c_[1 + normal[:, 2], -normal[:, 1], normal[:, 0], np.zeros(len(normal))]
    singular = np.linalg.norm(q, axis=1) < 1e-8
    q[singular] = [0, 1, 0, 0]
    q /= np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                  2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                  2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], axis=1).reshape(-1, 3, 3)
    worldR = np.einsum('ij,njk->nik', rotation.T, R)
    return np.einsum('nik,nk,njk->nij', worldR, road['gaussian_sigma_m'][rows] ** 2, worldR)


def main():
    from PIL import Image
    import torch
    torch.set_num_threads(1)
    source = CPU / 'ideal_code/ideal_world_teacher.py'
    spec = importlib.util.spec_from_file_location('frozen_ideal_world_field_read_only', source)
    field = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(field)
    config = json.loads((CPU / 'ideal_targets_r2/field_config.json').read_text())
    road = load(ASSET / 'road.npz', ['active', 'gaussian_sigma_m', 'normal_world', 'model_from_world', 'meter_to_model'])
    out_rows = []
    snapshots = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('marking_w120', 'marking_w240'):
        case = next(c for c in config['cases'] if c['name'] == name)
        checkpoint_path = OLD / f'native_relaxed_r1/{name}/checkpoint_final.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint['state']
        colors = state['x'].numpy()
        gradient = state['g'].numpy()
        fw = float((gradient.astype(np.float64) * colors - np.minimum(gradient.astype(np.float64), 0)).sum())
        snapshots.append(dict(case=name, path=str(checkpoint_path), iteration=state['iteration'],
                              finite=bool(np.isfinite(colors).all() and np.isfinite(gradient).all()),
                              box_feasible=bool(((colors >= 0) & (colors <= 1)).all()),
                              source_active_rows=int(len(checkpoint['world_rows'])), optimized_rows=len(colors),
                              scalar_at_zero=int((colors == 0).sum()), scalar_at_one=int((colors == 1).sum()),
                              projected_gradient_max=float(np.abs(np.clip(colors-gradient, 0, 1)-colors).max()),
                              objective=state['f'], FW_gap_numeric_proxy=fw,
                              FW_gap_over_objective=fw / state['f'], strict_negative_certificate=False,
                              optimization_state='500-step feasible iterate; no converged global-bound claim'))
        for raw in (86, 110, 134, 179, 198):
            target = load(CPU / f'ideal_targets_r2/{name}/{raw:03d}.npz')
            geo = load(CPU / f'ideal_targets_r2/geometry/{raw:03d}.npz')
            mask = target['detail_evaluation_mask'] & geo['canonical_mesh_hit_valid'] & (geo['fixed_alpha'] > .9)
            original = target['paint_fraction'][mask].astype(np.float64)
            core, flank = original >= .9, original <= .1
            xyz = geo['surface_world_xyz'][mask]
            jac = geo['pixel_world_xy_jacobian'][mask]
            samples = []
            for dx in (-.375, -.125, .125, .375):
                for dy in (-.375, -.125, .125, .375):
                    point = xyz.copy()
                    point[:, :2] += dx * jac[:, 0] + dy * jac[:, 1]
                    samples.append(field.evaluate_world(point, case, config)[1])
            AA = np.mean(samples, axis=0)
            contrast = float(original[core].mean() - original[flank].mean())
            AAcontrast = float(AA[core].mean() - AA[flank].mean())
            _, _, _, gv = field.curve_coordinates(xyz, config)
            screen_gradient = np.einsum('nw,npw->np', gv, jac)
            normal_screen = screen_gradient / np.linalg.norm(screen_gradient, axis=1)[:, None]
            ids = geo['triangle_world_rows'][mask]
            weights = geo['triangle_weights'][mask]
            valid = (ids >= 0).all(1)
            covariance = np.zeros((len(ids), 3, 3))
            covariance[valid] = np.einsum('nk,nkij->nij', weights[valid],
                                         world_covariance(road, ids[valid].reshape(-1)).reshape(-1, 3, 3, 3))
            camera = geo['world_to_camera']
            p = xyz @ camera[:3, :3].T + camera[:3, 3]
            fx, fy = geo['camera_K'][0, 0], geo['camera_K'][1, 1]
            J = np.zeros((len(p), 2, 3))
            J[:, 0, 0] = fx / p[:, 2]
            J[:, 1, 1] = fy / p[:, 2]
            J[:, 0, 2] = -fx * p[:, 0] / p[:, 2] ** 2
            J[:, 1, 2] = -fy * p[:, 1] / p[:, 2] ** 2
            J = np.einsum('nij,jk->nik', J, camera[:3, :3])
            cov2 = np.einsum('nik,nkl,njl->nij', J, covariance, J)
            cov2[:, 0, 0] += .3
            cov2[:, 1, 1] += .3
            cross_sigma = np.sqrt(np.einsum('ni,nij,nj->n', normal_screen, cov2, normal_screen))
            triangle_ids = np.unique(ids[(weights > 1e-12) & (ids >= 0)])
            old = json.loads((OLD / f'native_relaxed_r1/{name}/evaluation/evaluation.json').read_text())
            native_detail = next(r for r in old['detail']['frames'] if r['raw'] == raw)
            nativeP = load(OLD / f'native_relaxed_r1/{name}/evaluation/{raw:03d}.npz', ['road_only_rgb'])['road_only_rgb']
            asphalt_case = dict(case, marking=None)
            asphalt = field.evaluate_world(xyz, asphalt_case, config)[0]
            paint_rgb = np.asarray(config['paint_srgb']) - asphalt
            alpha = geo['fixed_alpha'][mask].astype(np.float64)
            fraction = ((nativeP[:, mask].T - alpha[:, None] * asphalt) * paint_rgb).sum(1) / (alpha * (paint_rgb ** 2).sum(1))
            width = target['width_px'][mask]
            core_ratio = (case['marking']['width_m'] - 2 * case['marking']['cross_edge_m']) / case['marking']['width_m']
            row = dict(case=name, raw=raw, detail_pixels=int(mask.sum()), core_pixels=int(core.sum()), flank_pixels=int(flank.sum()),
                       outer_width_core_px=quantiles(width[core]), flat_top_width_core_px=quantiles(width[core] * core_ratio),
                       flat_top_core_below3px=int((width[core] * core_ratio < 3).sum()),
                       native_metric=native_detail,
                       center_to_local_4x4_AA=dict(method='fixed canonical hit plus first-order local surface ray Jacobian,16 pixel footprint samples',
                                                approximate=True, target_changed=False, core_mean=float(AA[core].mean()),
                                                flank_mean=float(AA[flank].mean()), contrast_ratio=AAcontrast / contrast,
                                                paint_fraction_mean_abs_delta=float(np.abs(original-AA).mean()),
                                                native_contrast_relative_to_AA=native_detail['contrast_retention'] / (AAcontrast / contrast)),
                       static_anchor_footprint=dict(role='barycentric canonical triangle covariance proxy; not actual Gaussian alphaT attribution or render',
                                                    source_triangle_rows=len(triangle_ids), inactive_triangle_rows=int((~road['active'][triangle_ids]).sum()),
                                                    sigma_tangent_m=quantiles(road['gaussian_sigma_m'][triangle_ids, :2]),
                                                    cross_sigma_core_px=quantiles(cross_sigma[core]),
                                                    cross_FWHM_core_px=quantiles(2.354820045 * cross_sigma[core]),
                                                    outer_width_over_FWHM_core=quantiles(width[core] / (2.354820045 * cross_sigma[core])),
                                                    lowpass_variance_px2=.3),
                       metric_clipping=dict(unclipped_core_projection=quantiles(fraction[core]),
                                            core_below0=int((fraction[core] < 0).sum()), core_above1=int((fraction[core] > 1).sum())),
                       known_limits='center-sampled target and finite Gaussian footprint differ; AA proxy does not isolate loss of contrast or establish a representation limit')
            out_rows.append(row)
            if raw == 86:
                image = np.zeros((*mask.shape, 3), np.uint8)
                yy, xx = np.nonzero(mask)
                image[yy, xx, 0] = np.round(np.clip(original, 0, 1) * 255).astype(np.uint8)
                image[yy, xx, 1] = np.round(np.clip(fraction, 0, 1) * 255).astype(np.uint8)
                image[yy, xx, 2] = np.round(np.clip(AA, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(image).save(OUT / f'{name}_086_center_native_AA_channels.png')
        del checkpoint, colors, gradient
    result = dict(schema='existing_fine_marking_sampling_CPU_diagnostic.v1', GPU_executed=False,
                  native_render_executed=False, optimizer_updates=0, old_targets_modified=False,
                  old_acceptance_thresholds_modified=False, old_checkpoints=snapshots, rows=out_rows,
                  lowpass_source='/root/autodl-tmp/drivingeditor_repro/deps/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:110',
                  covariance_source='v44_factory.RoadRows fixed physical sigma and shortest normal rotation',
                  conclusions=[
                      '12cm raw86 satisfies registered outer-width>=3px, while part of its flat top is below3px.',
                      'Local pixel-footprint averaging changes the reference contrast, but does not by itself explain native0.8169.',
                      'Raw86 center-vs-AA convention and finite footprint merit a fixed sampling control before any representation upper-limit inference.',
                      'No sigma/density/model sweep, new target, new solver or native render was performed.'
                  ], human_verdict=None, visual_verdict=None)
    (OUT / 'marking_sampling_receipt.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(receipt=str(OUT / 'marking_sampling_receipt.json'), rows=len(out_rows), GPU_executed=False,
                          native_render_executed=False, torch_CUDA_initialized=torch.cuda.is_initialized())))


if __name__ == '__main__':
    main()
