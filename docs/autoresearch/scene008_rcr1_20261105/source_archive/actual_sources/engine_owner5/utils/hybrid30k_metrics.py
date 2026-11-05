"""同一实际前向的分层与局部误差；指标不先裁动态 alpha 或 RGB。"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def region_metrics(arrays, actor, sky):
    actor, sky = np.asarray(actor, dtype=bool).squeeze(), np.asarray(sky, dtype=bool).squeeze()
    shape = arrays['gt'].shape[-2:]
    if actor.shape != shape or sky.shape != shape:
        raise ValueError('评价mask尺寸与真实GT不一致')
    inside_distance = distance_transform_edt(actor)
    outside_distance = distance_transform_edt(~actor)
    regions = {'full': np.ones(shape, dtype=bool), 'actor': actor,
               'non_sky_outside_actor': ~actor & ~sky,
               'actor_core_5px': actor & (inside_distance > 5),
               'actor_inner_boundary_5px': actor & (inside_distance <= 5),
               'outside_boundary_5px': ~actor & ~sky & (outside_distance <= 5),
               'outside_far_5px': ~actor & ~sky & (outside_distance > 5)}
    gt = arrays['gt'].clip(0, 1).astype(np.float64)
    composite = arrays['composite'].clip(0, 1).astype(np.float64)
    static = arrays['static'].clip(0, 1).astype(np.float64)
    dynamic = arrays['dynamic'].clip(0, 1)
    alpha_d = arrays['alpha_d'].squeeze()
    alpha_s = arrays['alpha_s'].squeeze()
    rows = {}
    for name, mask in regions.items():
        count = int(mask.sum())
        row = {'pixel_count': count}
        if count:
            for channel, image in (('composite', composite), ('static', static)):
                mse = float(np.square(image[:, mask] - gt[:, mask]).mean())
                row[channel + '_mse'] = mse
                row[channel + '_psnr_db'] = -10 * float(np.log10(max(mse, 1e-12)))
            row.update(alpha_d_mean=float(alpha_d[mask].mean()), alpha_s_mean=float(alpha_s[mask].mean()),
                       dynamic_RGB_support_8bit_gt2=float((np.rint(dynamic[:, mask] * 255) > 2).any(axis=0).mean()),
                       composite_static_abs_mean=float(np.abs(composite[:, mask] - static[:, mask]).mean()))
        rows[name] = row
    return rows


def save_frame(outdir, frame_id, render_pkg, gt, dynamic_mask, sky_mask):
    arrays = {'gt': as_numpy(gt).astype(np.float32),
              'composite': as_numpy(render_pkg['render']).astype(np.float32),
              'static': as_numpy(render_pkg['render_s']).astype(np.float32),
              'dynamic': as_numpy(render_pkg['render_d']).astype(np.float32),
              'alpha_d': as_numpy(render_pkg['dynamic_alpha']).astype(np.float32),
              'alpha_s': as_numpy(render_pkg['static_alpha']).astype(np.float32),
              'depth_accum_model': as_numpy(render_pkg['depth']).astype(np.float32)}
    if arrays['gt'].shape != (3, 320, 480) or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('固定FP32/分辨率或评价finite检查失败')
    actor, sky = as_numpy(dynamic_mask).astype(bool), as_numpy(sky_mask).astype(bool)
    regions = region_metrics(arrays, actor, sky)
    root = Path(outdir) / 'hybrid_raw'
    root.mkdir(exist_ok=True)
    filename = frame_id.replace('/', '_')
    npz = root / (filename + '.npz')
    receipt = root / (filename + '.json')
    if npz.exists() or receipt.exists():
        raise FileExistsError('拒绝重复或覆盖hybrid评价帧')
    np.savez_compressed(npz, **arrays, actor_mask=actor, sky_mask=sky)
    row = {'frame_id': frame_id, 'regions': regions, 'arrays': str(npz.resolve()),
           'RGB_metrics': 'clamped RGB; float64 error from stored float32 arrays',
           'alpha_metrics': 'raw unmasked alpha; no body clipping',
           'depth_unit': 'accumulated model-unit depth, not normalized meter depth',
           'scope': 'all-observed reconstruction; no edited GT'}
    receipt.write_text(json.dumps(row, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return str(receipt.resolve())
