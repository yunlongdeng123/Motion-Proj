"""Frozen-scene, pixel-aligned attribution of local blue patches and road seams.

No fitting, checkpoint writes, power operations or changes to the parent run.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT.parent
SOURCE = CAMPAIGN / 'scene008_a6_complete_asset_20261105/source'
sys.path.insert(0, str(SOURCE))
from scene_runtime import Runtime, tagged, select_rows
from editor_common import prepare_rasterization, merge_rows, raster
from full_pvd import FixedAsset
from utils.sh_utils import eval_sh

PILOTS = (105, 113, 130)
GROUPS = ('a6', 'a37', 'road', 'static')
BOXES = {
    'left_road_patch': (0, 205, 215, 320),
    'right_curb_patch': (315, 177, 480, 290),
    'left_trees': (15, 65, 160, 180),
    'right_building': (365, 85, 480, 185),
}


def cpu(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def blue(x):
    return x[2] - .5 * (x[0] + x[1])


def save(path, x):
    x = cpu(x)
    if x.ndim == 2:
        x = np.repeat(x[None], 3, axis=0)
    Image.fromarray(np.uint8(np.clip(x.transpose(1, 2, 0), 0, 1) * 255 + .5)).save(path)


def heat(x):
    """Signed blue error: red/orange is excess blue, cyan is less blue."""
    v = np.clip(x / .08, -1, 1)
    result = np.full((3, *v.shape), .12, np.float32)
    result[0] += np.maximum(v, 0) * .88
    result[1] += np.maximum(-v, 0) * .88
    result[2] += np.maximum(-v, 0) * .88
    return result


def stats(image, ref, mask):
    n = int(mask.sum())
    if not n:
        return dict(pixels=0)
    value = image[:, mask].astype(np.float64)
    delta = value - ref[:, mask]
    be = blue(delta)
    return dict(pixels=n, mean_rgb=value.mean(1).tolist(),
                signed_rgb_error=delta.mean(1).tolist(), mae=float(np.abs(delta).mean()),
                blue_index=float(blue(value).mean()), blue_error=float(be.mean()),
                blue_error_p95=float(np.quantile(be, .95)),
                excess_blue_fraction=float((be > .03).mean()))


def component_stats(parts, mask):
    if not mask.any():
        return {}
    return {name: dict(mean_rgb=x[:, mask].astype(np.float64).mean(1).tolist(),
                       blue=float(blue(x[:, mask].astype(np.float64)).mean()))
            for name, x in parts.items()}


def boundary(image, ref, domain, actors):
    jumps, errors = [], []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        crossing = (domain[a] != domain[b]) & ~actors[a] & ~actors[b]
        j = image[(slice(None),) + a] - image[(slice(None),) + b]
        r = ref[(slice(None),) + a] - ref[(slice(None),) + b]
        jumps.append(np.abs(j)[:, crossing])
        errors.append(np.abs(j-r)[:, crossing])
    j, e = np.concatenate(jumps, 1), np.concatenate(errors, 1)
    return dict(neighbor_pairs=j.shape[1], rgb_jump=float(j.mean()) if j.size else None,
                rgb_gradient_error=float(e.mean()) if e.size else None)


def draw(rt, raw, scene):
    """Group RGB attribution retains all opacity, geometry and sorting inputs."""
    rows = scene['rows']
    colors = rows['colors_precomp']
    if colors is None:
        directions = rows['means3D'] - scene['cam'].camera_center
        directions = directions / directions.norm(dim=1, keepdim=True)
        colors = (eval_sh(rt.n.static.active_sh_degree,
                          rows['shs'].transpose(1, 2), directions) + .5).clamp_min(0)
    features = rows['features'].clone()
    features[:, 6:18] = (features[:, 2:6, None] * colors[:, None, :]).reshape(-1, 12)
    kwargs = dict(rows, features=features)
    out = raster(scene['cam'], rt.pipe, rt.n.static.active_sh_degree, kwargs, scene['bg'])
    env = (1-out['alpha']) * scene['env']
    image = out['rgb'] + env
    components = out['extra'][4:16].reshape(4, 3, 320, 480)
    reconstructed = components.sum(0) + (1-out['alpha']) * scene['bg'][:, None, None] + env
    parity = float((reconstructed-image).abs().max())
    if parity > 2e-5:
        raise RuntimeError(f'Group color attribution does not reconstruct RGB: {parity}')
    parts = {name: cpu(x) for name, x in zip(GROUPS, components)}
    parts['environment'] = cpu(env)
    return dict(rgb=cpu(image), alpha=cpu(out['alpha'][0]),
                weights=cpu(out['extra'][:4]), parts=parts, parity=parity)


def scenes(rt, raw, asset, delete):
    data = rt.raw(raw)
    static = prepare_rasterization(data['cam'], rt.n.static, rt.pipe)
    static['means2D'] = torch.zeros_like(static['means3D'])
    dynamic = rt.n.actor_kwargs(data['cam'], delete_a37=delete)
    native_rows = merge_rows(tagged(static, 'legacy_static'),
                             tagged(dynamic, owners=rt.n.keep_a6))
    native = dict(rows=native_rows, cam=data['cam'], env=data['env'], bg=rt.n.bg)
    fine = rt.scene(raw, baseline=True, delete=delete)
    road_mask = fine['rows']['features'][:, 4] > 0
    gates = dict(fine, rows=select_rows(fine['rows'], ~road_mask))
    off_rows = dict(fine['rows'])
    original = rt.exp.repair.road.colors().detach()
    if off_rows['shs'] is not None:
        off_rows['shs'] = off_rows['shs'].clone()
        off_rows['shs'][road_mask, 0] = (original-.5)/.28209479177387814
    else:
        off_rows['colors_precomp'] = off_rows['colors_precomp'].clone()
        off_rows['colors_precomp'][road_mask] = original
    return dict(native=native, gates=gates, road_off=dict(fine, rows=off_rows),
                road_fine=fine, asset_fine=rt.scene(raw, asset, delete=delete))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--output', default='evaluation')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(exist_ok=False)
    (output / 'frames').mkdir()
    torch.set_num_threads(1)
    start = time.monotonic()
    rt = Runtime(fine_road=True)
    asset = FixedAsset.from_geometry(CAMPAIGN / 'scene008_a6_complete_asset_20261105/appearance/complete_a6_fitted.npz').cuda()
    raw_ids = PILOTS if args.pilot else range(89, 138)
    records = []
    y, x = np.indices((320, 480))
    for raw in raw_ids:
        data = rt.raw(raw)
        reference = cpu(data['rgb'])
        actors = cpu(data['actor_labels']) > 0
        for delete in ((False, True) if raw in PILOTS else (False,)):
            state = 'delete' if delete else 'keep'
            rendered = {name: draw(rt, raw, sc) for name, sc in scenes(rt, raw, asset, delete).items()}
            domain = rendered['road_fine']['weights'][2] > .05
            seam = binary_dilation(domain, iterations=2) & ~binary_erosion(domain, iterations=2) & ~actors
            masks = dict(full=np.ones((320, 480), bool), non_actor=~actors,
                         observed_road=cpu(data['road']).astype(bool) & ~actors,
                         road_seam=seam, road_interior=domain & ~seam & ~actors,
                         nonroad=(rendered['road_fine']['weights'][2] < 1e-6) & ~actors)
            for name, (x0, y0, x1, y1) in BOXES.items():
                masks[name] = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & ~actors
            row = dict(raw=raw, state=state, metrics={}, components={}, effects={}, boundary={})
            row['metrics']['raw'] = {name: stats(reference, reference, mask) for name, mask in masks.items()}
            for name, pkg in rendered.items():
                row['metrics'][name] = {label: stats(pkg['rgb'], reference, mask) for label, mask in masks.items()}
                row['components'][name] = {label: component_stats(pkg['parts'], mask) for label, mask in masks.items()}
                row['boundary'][name] = boundary(pkg['rgb'], reference, domain, actors)
            for label, after, before in (('gate', 'gates', 'native'), ('road_insert', 'road_off', 'gates'),
                                         ('road_colour', 'road_fine', 'road_off'), ('asset', 'asset_fine', 'road_fine')):
                a, b = rendered[after], rendered[before]
                delta_parts = {key: a['parts'][key]-b['parts'][key] for key in a['parts']}
                row['effects'][label] = {name: dict(
                    total=stats(a['rgb'], b['rgb'], mask), attribution=component_stats(delta_parts, mask))
                    for name, mask in masks.items()}
            row['boundary']['raw'] = boundary(reference, reference, domain, actors)
            row['attribution_maxabs'] = max(p['parity'] for p in rendered.values())
            if raw in PILOTS:
                direct = rt.n.render(data['cam'], dynamic=not delete)['rgb']
                row['native_reference_maxabs'] = float(np.abs(rendered['native']['rgb']-cpu(direct)).max())
                assert row['native_reference_maxabs'] < 2e-6
                exact = dict(raw_rgb=reference, road_domain=domain, road_seam=seam, actors=actors)
                for name, pkg in rendered.items():
                    exact[name] = pkg['rgb']
                    for part, image in pkg['parts'].items():
                        exact[f'{name}_{part}'] = image
                np.savez_compressed(output/f'{raw:03d}_{state}_exact.npz', **exact)
            if state == 'keep' or raw in PILOTS:
                save(output/'frames'/f'{raw:03d}_{state}_raw.png', reference)
                for name, pkg in rendered.items():
                    save(output/'frames'/f'{raw:03d}_{state}_{name}.png', pkg['rgb'])
                save(output/'frames'/f'{raw:03d}_{state}_native_blue_error.png', heat(blue(rendered['native']['rgb']-reference)))
                save(output/'frames'/f'{raw:03d}_{state}_road_seam.png', seam.astype(np.float32))
            records.append(row)
            print(json.dumps(dict(raw=raw, state=state, elapsed_s=round(time.monotonic()-start, 2),
                                  native_left_blue_error=row['metrics']['native']['left_road_patch']['blue_error'],
                                  attribution_maxabs=row['attribution_maxabs'])), flush=True)
        rt._data.clear()
    assert rt.modelunchanged() and asset.assert_geometry_unchanged()
    report = dict(task='scene008_local_blue_road_seam_20261105_r1',
                  status='FROZEN_DIAGNOSTIC_COMPLETE', training_updates=0, seconds=time.monotonic()-start,
                  raw_ids=list(raw_ids), pilot_delete_ids=list(PILOTS), rows=records,
                  roi_boxes=BOXES, runtime=rt.meta, frozen_scene_unchanged=True,
                  definitions=dict(blue='B-(R+G)/2', excess_blue='Blue(render)-Blue(originalRGB), before display clipping',
                                   components='Exact group RGB contribution under the same joint depth sorting; environment=(1-alpha)*env.',
                                   boundary='Crossing neighbors of saved road alpha*T>0.05, excluding known actors.'),
                  limitations=['Fixed image-space ROIs are diagnostic windows, not persistent semantic labels.',
                               'Deleted actor regions have no same-camera counterfactual RGB truth.',
                               'Additive component attribution is not a physical illumination estimate.',
                               'This task does not test or change vehicle identity, training, or generated geometry.'])
    (output/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], frames=len(records), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
