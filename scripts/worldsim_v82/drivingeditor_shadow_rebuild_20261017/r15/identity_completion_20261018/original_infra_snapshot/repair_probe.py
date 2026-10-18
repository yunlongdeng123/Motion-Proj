"""Saved native-size observations and per-ID deletion probes; no parameter updates."""
import json
from pathlib import Path


def save_repair_probe(output, step, cameras, static, instances, config, environment, spec):
    import numpy as np
    import torch
    from PIL import Image
    if __package__:
        from .renderer import render
        from .runtime import masks_for_frame
    else:
        from renderer import render
        from runtime import masks_for_frame
    root = Path(output) / 'repair_probes' / ('step_%06d' % step)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.no_grad():
        for frame in (0, 6, 10, 20, 28, 30, 34, 38, 48):
            camera = cameras[frame]
            gt = camera.original_image.to(static.get_xyz)
            masks, known = masks_for_frame(spec, instances, camera)
            union = masks[known].sum(0).clamp(0, 1).bool()
            options = dict(env_map=environment, instance_ids=instances.instance_ids, is_training=False)
            base = render(camera, static, instances, config, gt.new_zeros(3), **options)
            values = {'gt': gt, 'composite': base['render'], 'static': base['render_s'],
                      'vehicles': base['raw_render_d'], 'static_alpha': base['static_alpha']}
            for i, track in enumerate(instances.instance_ids):
                deleted = render(camera, static, instances, config, gt.new_zeros(3), edits={track: {'delete': True}}, **options)
                if not torch.equal(base['render_s'], deleted['render_s']):
                    raise AssertionError('per-ID deletion altered independent static background')
                if bool((deleted['instance_visible_contribution'][i] != 0).any()):
                    raise AssertionError('deleted ID retains visible contribution')
                values['delete%d' % i] = deleted['render']
                values['id%d_alpha' % i] = base['instance_visible_contribution'][i:i+1]
            fid = Path(camera.image_name).stem
            for name, value in values.items():
                v = value.detach().float().cpu().numpy()
                pixels = v[0] if v.shape[0] == 1 else v.transpose(1, 2, 0)
                Image.fromarray(np.round(np.clip(pixels, 0, 1)*255).astype(np.uint8)).save(root / (fid+'_'+name+'.png'))
            row = {'frame_index': frame, 'frame_id': fid,
                   'composite_MSE': float((base['render'].clamp(0, 1)-gt).square().mean()),
                   'static_outside_body_MSE': float((base['render_s'][:, ~union]-gt[:, ~union]).square().mean()),
                   'body_visible_alpha': [float(base['instance_visible_contribution'][i][masks[i].bool()].mean())
                                          if bool(masks[i].any()) else None for i in range(len(instances.instance_ids))]}
            if frame == 28:
                sl = (slice(None), slice(270, 305), slice(40, 110))
                row['road_roi_xyxy'] = [40,270,110,305]
                row['road_RGB_MAE_GT'] = float((base['render_s'][sl]-gt[sl]).abs().mean())
                row['road_RGB_mean'] = float(base['render_s'][sl].mean())
                row['road_GT_mean'] = float(gt[sl].mean())
                row['road_alpha_mean'] = float(base['static_alpha'][sl].mean())
            rows.append(row)
    (root / 'metrics.json').write_text(json.dumps({'step': step, 'frames': rows,
        'static_points': len(static.get_xyz), 'instance_points': instances.num_points,
        'optimizer_updates_in_probe': 0}, indent=2), encoding='utf-8')
    return rows
