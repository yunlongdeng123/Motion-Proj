"""Compare actual saved 3D renders on immutable A/C, coverage and full keep masks.

No render or optimization is performed. C agreement is pseudo-teacher agreement.
The other three cameras are diagnostics; this script never labels them blind tests.
"""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image, ImageDraw

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD = CAMPAIGN / 'road_asset_20261116_r1'
NEW = CAMPAIGN / 'road_direct_teacher_20261116_r1'
PARENT = CAMPAIGN / 'scene065_closedloop_20261116_r1/handoff_coverage_r2'
KEYS = (86, 110, 134, 179, 198)


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def weighted_mae(prediction, alpha, truth, mask, weight):
    q = np.asarray(weight, np.float64) * mask
    if q.sum() == 0:
        return None
    return float((np.abs(prediction.astype(np.float64) - alpha[None] * truth) * q[None]).sum() / (3 * q.sum()))


def panel(path, tiles):
    width, height = 480, 320
    columns = min(3, len(tiles)) if len(tiles) > 4 else min(2, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    image = Image.new('RGB', (columns * width, rows * (height + 24)), '#11151c')
    draw = ImageDraw.Draw(image)
    for index, (label, pixels) in enumerate(tiles[:6]):
        row, col = divmod(index, columns)
        a = np.asarray(pixels)
        if a.shape == (3, height, width):
            a = a.transpose(1, 2, 0)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)
        tile = Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8))
        image.paste(tile, (col * width, row * (height + 24) + 24))
        draw.text((col * width + 5, row * (height + 24) + 5), label, fill='#e6edf3')
    image.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, default=OLD / 'phase2/formal_r1/rendered')
    p.add_argument('--candidate', action='append', default=[], help='label=/absolute/public-render-directory')
    p.add_argument('--supervision', type=Path, default=NEW / 'supervision_r3')
    p.add_argument('--repo', type=Path, default=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo / 'scripts/worldsim_v82'))
    from drivingeditor_dynamic_edit.road_asset_quality import audit_render

    args.output.mkdir(parents=True, exist_ok=False)
    paths = dict(baseline=args.baseline)
    for argument in args.candidate:
        label, location = argument.split('=', 1)
        if label in paths:
            raise ValueError('Repeated render label')
        paths[label] = Path(location)
    checks = {}
    rows = []
    for raw in KEYS:
        domain = load(PARENT / f'contribution_diagnostics_20261116/float/{raw:03d}.npz')['domain']
        keep = load(PARENT / f'domains/{raw:03d}.npz')['K']
        data = {label: load(path / f'{raw:03d}/arrays.npz') for label, path in paths.items()}
        baseline = data['baseline']
        target = load(args.supervision / f'{raw:03d}.npz') if raw in (86, 110) else None
        stages = {}
        tiles = []
        support_tiles = []
        keep_tiles = []
        if target is not None:
            tiles.extend([('A: original visible RGB', target['target_A_srgb']),
                          ('C: frozen pseudo teacher', target['target_C_srgb'])])
        for label, value in data.items():
            record, _ = audit_render(
                value['road_only_alpha'], value['road_alphaT'], value['static_alphaT'], value['dynamic_alphaT'],
                value['total_alpha'], domain, rgb=value['rgb'], road_only_rgb=value['road_only_rgb'],
                parts={name: value[name] for name in ('static_rgb', 'dynamic_rgb', 'road_rgb', 'environment_rgb')},
                protected=keep, baseline_rgb=baseline['rgb'])
            metrics = record['metrics']
            stage = dict(mechanical_status=record['technical_status'],
                         gap_pixels=metrics['gap_pixels'], thin_pixels=metrics['thin_pixels'],
                         supported_pixels=metrics['supported_pixels'],
                         K_joint_max_difference=metrics['protected']['max_abs_rgb'],
                         alphaT_conservation_max_error=metrics['alphaT_conservation_max_error'],
                         RGB_conservation_max_error=metrics['RGB_conservation_max_error'])
            if target is not None:
                alpha = value['road_only_alpha'].squeeze().astype(np.float64)
                stage.update(A_real_premult_MAE=weighted_mae(value['road_only_rgb'], alpha, target['target_A_srgb'], target['A_mask'], target['q_A']),
                             C_pseudo_premult_MAE=weighted_mae(value['road_only_rgb'], alpha, target['target_C_srgb'], target['C_mask'], target['q_C']),
                             immutable_F_C_premult_MAE=weighted_mae(value['road_only_rgb'], alpha, target['F_control_C_srgb'], target['C_mask'], target['q_C']))
            stages[label] = stage
            native_alpha = value['road_only_alpha'].squeeze()
            support = np.full((*domain.shape, 3), .04, np.float32)
            support[domain & (native_alpha < .05)] = (.92, .2, .25)
            support[domain & (native_alpha >= .05) & (native_alpha <= .9)] = (.95, .65, .2)
            support[domain & (native_alpha > .9)] = (.1, .7, .5)
            support_tiles.append((label + ': fixed-domain alpha support', support))
            checks[f'{raw}_{label}_mechanical'] = record['technical_status'] == 'pass'
            checks[f'{raw}_{label}_joint_K_exact_zero'] = stage['K_joint_max_difference'] == 0
            if label != 'baseline':
                keep_error = np.max(np.abs(value['rgb'] - baseline['rgb']), axis=0)
                keep_image = np.zeros((*keep.shape, 3), np.float32)
                keep_image[keep, 0] = np.minimum(keep_error[keep] * 255, 1)
                keep_tiles.append((label + ': joint K delta / (1/255)', keep_image))
                checks[f'{raw}_{label}_support_nonregression'] = stage['supported_pixels'] >= stages['baseline']['supported_pixels']
                # Hard expansion is a parameter contract; report actual pixel regression too.
                stage['road_alpha_max_decrease'] = float(np.maximum(
                    baseline['road_only_alpha'].squeeze() - value['road_only_alpha'].squeeze(), 0).max())
                if target is not None:
                    base = stages['baseline']
                    stage['A_delta_vs_baseline'] = stage['A_real_premult_MAE'] - base['A_real_premult_MAE']
                    stage['C_fraction_improvement_vs_baseline'] = 1 - stage['C_pseudo_premult_MAE'] / base['C_pseudo_premult_MAE']
                    checks[f'{raw}_{label}_A_within_0.002'] = stage['A_delta_vs_baseline'] <= .002
            tiles.append((label + ': saved 3D road', value['road_only_rgb']))
        if target is None:
            for label, value in data.items():
                tiles.append((label + ': saved 3D joint', value['rgb']))
        else:
            mask_image = target['target_A_srgb'].transpose(1, 2, 0) * .25
            alpha_only = (target['coverage_A'] | target['coverage_C']) & ~(target['A_mask'] | target['C_mask'])
            mask_image[alpha_only] = (.1, .85, .85)
            mask_image[target['A_mask']] = (.1, .9, .3)
            mask_image[target['C_mask']] = (.9, .1, .8)
            mask_image[target['K']] = (.15, .35, 1.)
            tiles.append(('A green / C purple / alpha-only cyan / K blue', mask_image))
        panel(args.output / f'{raw:03d}_comparison.png', tiles)
        panel(args.output / f'{raw:03d}_alpha_support.png', support_tiles)
        if keep_tiles:
            panel(args.output / f'{raw:03d}_joint_K_difference.png', keep_tiles)
        rows.append(dict(raw=raw, role='training fidelity / pseudo teacher' if target is not None else 'diagnostic, not blind',
                         evaluation_pixels=int(domain.sum()), full_K_pixels=int(keep.sum()), stages=stages,
                         A_pixels=int(target['A_mask'].sum()) if target is not None else None,
                         C_common_pixels=int(target['C_mask'].sum()) if target is not None else None,
                         C_full_any_footprint_pixels=int(target['C_full_hole'].sum()) if target is not None else None,
                         C_full_geometry_pixels=int(target['C_teacher_hole'].sum()) if target is not None else None))

    totals = {label: {key: sum(row['stages'][label][key] for row in rows)
                     for key in ('gap_pixels', 'thin_pixels', 'supported_pixels')} for label in paths}
    checks['fixed_evaluation_38416'] = sum(row['evaluation_pixels'] for row in rows) == 38416
    checks['fixed_keep_77266'] = sum(row['full_K_pixels'] for row in rows) == 77266
    checks['baseline_supported_37404'] = totals['baseline']['supported_pixels'] == 37404
    for label in paths:
        checks[f'{label}_overall_supported_at_least_37404'] = totals[label]['supported_pixels'] >= 37404
    direct_goal = None
    if 'direct_C' in paths:
        direct_goal = all(row['stages']['direct_C']['C_fraction_improvement_vs_baseline'] >= .2 for row in rows[:2])
    result = dict(schema='independent_direct_teacher_public_render_comparison.v1',
                  render_mechanical_status='pass' if all(checks.values()) else 'failed',
                  direct_pseudo_teacher_20percent_goal='incomplete' if direct_goal is None else ('pass' if direct_goal else 'failed'),
                  matched_control_present='matched_F_control' in paths,
                  checks=checks, totals=totals, frames=rows,
                  fixed_evaluation_pixel_frames=sum(row['evaluation_pixels'] for row in rows),
                  full_K_pixel_frames=sum(row['full_K_pixels'] for row in rows),
                  A_meaning='real visible training fidelity', C_meaning='frozen pseudo-teacher agreement; not hidden road truth',
                  comparison_paths={key: str(value) for key, value in paths.items()}, supervision=str(args.supervision),
                  audit_GPU_forwards=0, audit_training_updates=0, human_verdict=None,
                  limitations=['scope/budget/source binding and lifecycle require separate audits',
                               'three diagnostic cameras are not world-isolated blind tests',
                               'numeric teacher agreement does not establish material or marking quality'])
    (args.output / 'acceptance.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('render_mechanical_status', 'direct_pseudo_teacher_20percent_goal', 'totals')}))


if __name__ == '__main__':
    main()
