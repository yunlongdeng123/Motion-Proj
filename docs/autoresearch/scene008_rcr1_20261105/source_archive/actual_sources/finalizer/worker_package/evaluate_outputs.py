"""Float readout first, PNGs second; no automatic visual-quality verdict."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

from calibration_adapter import evidence
from final_runtime import read, write
from sky_metrics import sky_statistics

DELETE_RAWS = (105, 113, 130, 137)
CROPS = {
    105: {'road_detail': (250, 170, 330, 235)},
    113: {'user_box': (276, 175, 309, 220), 'white_line': (287, 182, 299, 215)},
    130: {'road_detail': (250, 170, 330, 235)},
    137: {'right_patch': (276, 175, 309, 220), 'environment_band': (210, 213, 290, 233)},
}


def values(value, mask):
    selected = value[..., mask]
    if selected.numel() == 0:
        return dict(pixels=int(mask.sum()), mean=None, max=None, sum=0.)
    assert bool(torch.isfinite(selected).all())
    return dict(pixels=int(mask.sum()), mean=float(selected.mean()), max=float(selected.max()),
                sum=float(selected.double().sum()))


def png(rgb):
    array = rgb.detach().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.uint8(np.clip(array, 0, 1)*255+.5))


def float_roi(out, data, box):
    x0, y0, x1, y1 = box
    obs = data['obs']
    original_actors = (obs['actor_labels'] > 0) | obs['visible_reviewed'].any(0)
    rgb = out['rgb'][:, y0:y1, x0:x1].detach().cpu().numpy().astype(np.float32)
    groups = out['extra'][:4, y0:y1, x0:x1].detach().cpu().numpy().astype(np.float32)
    return dict(xyxy=np.asarray(box), rgb=rgb,
                target=data['rgb'][:, y0:y1, x0:x1].cpu().numpy().astype(np.float32),
                component_alpha_T=groups,
                alpha=out['alpha'][:, y0:y1, x0:x1].cpu().numpy().astype(np.float32),
                depth_sum=out['depth_sum'][:, y0:y1, x0:x1].cpu().numpy().astype(np.float32),
                original_observed_background=(~original_actors[y0:y1, x0:x1]),
                current_M2_support=(groups[0] > .01))


def compare_roi(candidate, reference_path):
    with np.load(reference_path, allow_pickle=False) as old:
        np.testing.assert_array_equal(old['xyxy'], candidate['xyxy'])
        np.testing.assert_array_equal(old['target'], candidate['target'])
        np.testing.assert_array_equal(old['original_observed_background'], candidate['original_observed_background'])
        # Same intersection mask on both endpoints; a changed occlusion cannot
        # improve a mean by dropping a difficult M2-overlapped pixel.
        valid = candidate['original_observed_background'] & ~candidate['current_M2_support'] & ~old['current_M2_support']
        count = int(valid.sum())
        record = dict(reference=str(reference_path), common_observed_background_pixels=count,
                      mask='Original actors excluded, current/reference M2 alphaT>.01 excluded on BOTH endpoints')
        for label, rgb in (('reference', old['rgb']), ('candidate', candidate['rgb'])):
            error = rgb-candidate['target']
            blue = error[2]-.5*(error[0]+error[1])
            record[label] = dict(rgb_mae=float(np.abs(error[:, valid]).mean()) if count else None,
                 signed_blue_error=float(blue[valid].mean()) if count else None,
                 absolute_blue_error=float(np.abs(blue[valid]).mean()) if count else None,
                 positive_blue_error_gt_05_pixels=int((blue[valid] > .05).sum()),
                 positive_blue_error_max=float(blue[valid].max()) if count else None)
        record['same_pixel_RGB_change_max'] = float(np.abs(candidate['rgb']-old['rgb']).max())
        return record


def evaluate(runtime, destination, label, reference=None, preflight=False):
    destination = Path(destination).resolve()
    destination.mkdir(exist_ok=False)
    (destination/'float_rois').mkdir()
    (destination/'crops').mkdir()
    reference = Path(reference).resolve() if reference else None
    if reference:
        reference_manifest = read(reference/'render_manifest.json')
        if not reference_manifest.get('accepted_current_M2_reference_only'):
            raise ValueError('Comparison reference must explicitly identify the accepted current M2 scene')
        if reference_manifest['M2_sha256'] != runtime.binding['M2_sha256']:
            raise ValueError('Reference and candidate must use the exact same complete M2 asset file')
        reference_rows = {(row['raw'], row['state']): row for row in read(reference/'float_metrics.json')['rows']}
        (destination/'comparisons').mkdir()
    rows = []
    comparisons = []
    raws = (105,) if preflight else tuple(range(89, 138))
    for raw in raws:
        data = runtime.raw(raw)
        for state in ('keep', 'delete'):
            if state == 'delete' and not (preflight or raw in DELETE_RAWS):
                continue
            delete = state == 'delete'
            out = runtime.render_scene(raw, delete=delete)
            assert all(bool(torch.isfinite(out[key]).all()) for key in ('rgb', 'alpha', 'depth_sum', 'extra'))
            ev = evidence(runtime, raw, delete)
            target_error = (out['rgb']-ev['target']).abs().mean(0)
            true_error = out['rgb']-data['rgb']
            blue_error = true_error[2]-.5*(true_error[0]+true_error[1])
            groups = out['extra'][:4]
            actors = torch.as_tensor((data['obs']['actor_labels'] > 0) | data['obs']['visible_reviewed'].any(0), device='cuda')
            background = ~actors & (groups[0] <= .01)
            weight = groups[2]
            affected = weight >= .02
            boundary = (weight > .15).cpu().numpy()
            seam = torch.as_tensor(binary_dilation(boundary, iterations=2) & ~binary_erosion(boundary, iterations=2), device='cuda')
            no_a37 = torch.zeros_like(ev['a37']) if delete else ev['a37']
            record = dict(raw=raw, state=state, timestamp=float(data['cam'].timestamp),
                          native_a6_rows_input=0, fixed_complete_M2=True,
                          regions=dict(road_target=values(target_error, ev['road']),
                                       road_affected=values(target_error, ev['road'] & affected),
                                       road_seam=values(target_error, ev['road'] & seam),
                                       a6_visible_RGB=values(true_error.abs().mean(0), ev['a6']),
                                       a37_visible_RGB=values(true_error.abs().mean(0), no_a37),
                                       observed_background_RGB=values(true_error.abs().mean(0), background)),
                          observed_background_blue=dict(signed=values(blue_error, background),
                                       absolute=values(blue_error.abs(), background),
                                       positive_gt_05_pixels=int(((blue_error > .05) & background).sum())),
                          target_role='Original RGB' if not delete else 'Reobserved-road proxy on fixed ground support; NOT deleted-object RGB GT')
            record.update(sky_statistics(runtime, raw, out['rgb'], data['rgb']))
            frame = png(out['rgb'])
            frame.save(destination/f'raw{raw}_{state}.png')
            if reference:
                old = reference_rows[raw, state]
                fixed = {}
                # These masks are the same observations at both endpoints.
                # Affected/seam/background conditional masks are intentionally
                # not used for aggregate endpoint comparisons.
                for name in ('road_target', 'a6_visible_RGB', 'a37_visible_RGB'):
                    before, after = old['regions'][name], record['regions'][name]
                    assert before['pixels'] == after['pixels'], (raw, state, name)
                    fixed[name] = dict(pixels=after['pixels'], reference_mean=before['mean'],
                                       candidate_mean=after['mean'],
                                       delta_sum=after['sum']-before['sum'])
                record['same_fixed_domain_reference_comparison'] = fixed
            for name, box in CROPS.get(raw, {}).items():
                saved = float_roi(out, data, box)
                np.savez_compressed(destination/'float_rois'/f'raw{raw}_{state}_{name}.npz', **saved)
                x0, y0, x1, y1 = box
                region = torch.zeros_like(background)
                region[y0:y1, x0:x1] = True
                mask = region & background
                record.setdefault('narrow_ROIs', {})[name] = dict(xyxy=box,
                    valid_observed_background_pixels=int(mask.sum()),
                    RGB=values(true_error.abs().mean(0), mask), blue=values(blue_error, mask),
                    absolute_blue=values(blue_error.abs(), mask),
                    positive_blue_gt_05_pixels=int(((blue_error > .05) & mask).sum()),
                    component_alpha_T_mean=[float(group[region].mean()) for group in groups],
                    scope='Raw renderer coordinates; pixels outside mask remain saved for direct visual checking')
                crop = frame.crop(box).resize(((x1-x0)*6, (y1-y0)*6), Image.Resampling.NEAREST)
                panel = Image.new('RGB', (max(660, crop.width), crop.height+55), 'white')
                draw = ImageDraw.Draw(panel)
                draw.text((6, 5), label, fill='black')
                draw.text((6, 22), f'raw{raw} / a37 {state.upper()} / {name} / x{x0}:{x1}, y{y0}:{y1}', fill='black')
                draw.text((6, 38), 'Complete fixed M2 in both states / renderer ROI / nearest 6x', fill='black')
                panel.paste(crop, ((panel.width-crop.width)//2, 55))
                panel.save(destination/'crops'/f'raw{raw}_{state}_{name}.png')
                if reference:
                    path = reference/'float_rois'/f'raw{raw}_{state}_{name}.npz'
                    comparisons.append(dict(raw=raw, state=state, region=name, **compare_roi(saved, path)))
                    with Image.open(reference/f'raw{raw}_{state}.png') as original:
                        old_crop = original.crop(box).resize(crop.size, Image.Resampling.NEAREST)
                    pair = Image.new('RGB', (panel.width*2, panel.height), 'white')
                    pair.paste(panel, (panel.width, 0))
                    pair.paste(old_crop, ((panel.width-crop.width)//2, 55))
                    draw = ImageDraw.Draw(pair)
                    draw.text((6, 5), 'CURRENT accepted M2 / original native10K+road1K+fine', fill='black')
                    draw.text((6, 22), f'raw{raw} / a37 {state.upper()} / {name} / x{x0}:{x1}, y{y0}:{y1}', fill='black')
                    draw.text((6, 38), 'Same raw camera and renderer-coordinate crop / nearest 6x', fill='black')
                    pair.save(destination/'comparisons'/f'raw{raw}_{state}_{name}.png')
            rows.append(record)
            print(f'RENDER raw{raw} {state}', flush=True)
    write(destination/'float_metrics.json', dict(float_before_PNG=True, rows=rows,
          reference_comparison=comparisons, quality_accepted=False,
          limitation='Narrow ROI metrics are not a whole-scene visual-quality certificate; road_edit_rgb is a proxy.',
          large_region_mean_does_not_override_narrow_strip_evidence=True))
    write(destination/'render_manifest.json', dict(label=label, binding_sha256=runtime.binding_sha256,
          M2_sha256=runtime.binding['M2_sha256'],
          accepted_current_M2_reference_only=False, preflight_only=bool(preflight),
          keep_raws=list(raws), delete_raws=list(raws) if preflight else list(DELETE_RAWS),
          fixed_complete_M2_in_both_states=True, only_a37_changes_with_delete=True,
          native_a6_rows_input=0, colour_field_shared_across_keep_delete=True,
          original_own_camera_per_raw=True, one_real_depth_order=True,
          float_metrics_before_PNG_encoding=True, narrow_xyxy_exclusive=CROPS,
          actual_native_reader_sky_statistics=True,
          observed_background_excludes_original_actors_and_M2_support=True,
          training_updates=0, quality_accepted=False))
    return rows
