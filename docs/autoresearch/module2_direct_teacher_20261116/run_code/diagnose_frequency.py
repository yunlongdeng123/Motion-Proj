"""Describe new image frequencies from actual fixed-camera float road renders.

The fixed support intersection excludes alpha boundaries. No threshold is used
to declare quality, and no reference image is invented. Color200 has unchanged
opacity/scales, providing an appearance-only stage attribution.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
NEW = CAMPAIGN / 'road_direct_teacher_20261116_r1'
OLD = CAMPAIGN / 'road_asset_20261116_r1'
PARENT = CAMPAIGN / 'scene065_closedloop_20261116_r1/handoff_coverage_r2'


def selected(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in ('road_only_rgb','road_only_alpha')}


def curvature(rgb, mask, axis):
    if axis == 'y':
        value = rgb[:,2:,:] - 2*rgb[:,1:-1,:] + rgb[:,:-2,:]
        valid = mask[2:,:] & mask[1:-1,:] & mask[:-2,:]
    else:
        value = rgb[:,:,2:] - 2*rgb[:,:,1:-1] + rgb[:,:,:-2]
        valid = mask[:,2:] & mask[:,1:-1] & mask[:,:-2]
    return dict(mean_abs=float(np.abs(value[:,valid]).mean()) if valid.any() else None,
                pixel_triplets=int(valid.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, default=NEW)
    p.add_argument('--output', type=Path, default=NEW / 'qa/frequency_diagnostic_r1')
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows = []
    for raw in (86,110,134,179,198):
        baseline = selected(OLD / f'phase2/formal_r1/rendered/{raw:03d}/arrays.npz')
        data = {'baseline':baseline}
        for arm in ('direct','control'):
            for stage in ('color','coverage','final'):
                data[arm+'_'+stage] = selected(args.campaign / f'public_render_r1/{arm}_{stage}/{raw:03d}/arrays.npz')
        with np.load(PARENT / f'contribution_diagnostics_20261116/float/{raw:03d}.npz') as values:
            mask = values['domain'].astype(bool)
        for value in data.values():
            mask &= value['road_only_alpha'].squeeze() > .99
        mask = binary_erosion(mask, iterations=2)
        metrics = {}
        for label,value in data.items():
            rgb = value['road_only_rgb'].astype(np.float64)
            metrics[label] = dict(y_second_difference=curvature(rgb,mask,'y'),
                                  x_second_difference=curvature(rgb,mask,'x'))
        for label,value in metrics.items():
            value['y_ratio_to_baseline'] = value['y_second_difference']['mean_abs'] / metrics['baseline']['y_second_difference']['mean_abs']
        # Native-size crop: no exposure adjustment, interpolation, denoise or sharpen.
        left, top, right, bottom = 60, 220, 450, 320
        width, height = right-left, bottom-top
        canvas = Image.new('RGB',(4*width,2*(height+22)),'#11151c')
        draw = ImageDraw.Draw(canvas)
        for r,arm in enumerate(('direct','control')):
            for c,stage in enumerate(('baseline','color','coverage','final')):
                label = 'baseline' if stage == 'baseline' else arm+'_'+stage
                pixels = data[label]['road_only_rgb'][:,top:bottom,left:right].transpose(1,2,0)
                tile = Image.fromarray(np.round(np.clip(pixels,0,1)*255).astype(np.uint8))
                canvas.paste(tile,(c*width,r*(height+22)+22))
                draw.text((c*width+4,r*(height+22)+4),arm+': '+stage,fill='#e6edf3')
        canvas.save(args.output / f'{raw:03d}_native_crop.png')
        rows.append(dict(raw=raw,role='training fidelity / pseudo teacher' if raw in (86,110) else 'diagnostic, not blind',
                         fixed_common_support_pixels=int(mask.sum()),metrics=metrics,
                         crop_xyxy=[left,top,right,bottom]))
    result = dict(schema='native_road_image_frequency_diagnostic.v1',frames=rows,
                  construction='actual premult sRGB float; same fixed evaluation domain and alpha>.99 support intersection, eroded two pixels',
                  statistic='mean absolute second finite difference per RGB channel; y variation describes horizontal stripes',
                  thresholds_for_quality=None,hidden_GT_used=False,new_training_or_GPU_forwards=0,human_verdict=None,
                  stage_attribution='color200 opacity and scales are bitwise unchanged from baseline by checkpoint audit; new color-stage frequencies arise from learned Gaussian RGB')
    (args.output / 'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(raw=row['raw'],y_ratios={key:round(value['y_ratio_to_baseline'],3) for key,value in row['metrics'].items()}) for row in rows]))


if __name__ == '__main__':
    main()
