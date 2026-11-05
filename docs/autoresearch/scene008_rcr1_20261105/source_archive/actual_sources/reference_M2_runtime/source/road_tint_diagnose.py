"""Three-camera causal render controls for road discontinuities and blue tint."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

RUN = Path(__file__).resolve().parent.parent
OLD = RUN.parent / 'scene008_a6_complete_asset_20261105'
sys.path.insert(0, str(OLD / 'source'))
from full_pvd import FixedAsset, sha_file
from scene_runtime import Runtime, GROUPS, select_rows

RAWS = (105, 113, 130)


def cpu(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def as_image(value):
    array = cpu(value)
    if array.ndim == 2:
        array = np.repeat(array[None], 3, axis=0)
    return Image.fromarray(np.uint8(np.clip(array.transpose(1, 2, 0), 0, 1) * 255 + .5))


def blue(image):
    return image[2] - .5 * (image[0] + image[1])


def stats(image, reference, mask):
    n = int(mask.sum())
    if not n:
        return dict(pixels=0)
    value = image[:, mask].astype(np.float64)
    residual = value - reference[:, mask]
    return dict(pixels=n, mean_rgb=value.mean(1).tolist(), signed_rgb_error=residual.mean(1).tolist(),
                mae=float(np.abs(residual).mean()), blue_index=float(blue(value).mean()),
                blue_error=float(blue(residual).mean()), out_of_range_fraction=float(((value < 0) | (value > 1)).mean()))


def delta_stats(value, mask):
    selected = value[:, mask].astype(np.float64)
    return dict(pixels=int(mask.sum()), mae=float(np.abs(selected).mean()),
                rgb_delta=selected.mean(1).tolist(), blue_delta=float(blue(selected).mean()),
                maxabs=float(np.abs(selected).max())) if mask.any() else dict(pixels=0)


def edge_contrast(image, edge, actors):
    values = []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        crossing = (edge[a] != edge[b]) & ~actors[a] & ~actors[b]
        values.append(np.abs(image[(slice(None),) + a] - image[(slice(None),) + b])[:, crossing])
    joined = np.concatenate(values, axis=1)
    return dict(neighbor_pairs=joined.shape[1], rgb_jump=float(joined.mean()) if joined.size else None)


def panel(path, entries):
    canvas = Image.new('RGB', (4 * 480, 3 * 348), 'white')
    for i, (label, value) in enumerate(entries):
        x, y = (i % 4) * 480, (i // 4) * 348
        canvas.paste(as_image(value), (x, y + 28))
        ImageDraw.Draw(canvas).text((x + 8, y + 7), label, fill=(20, 25, 30))
    canvas.save(path)


@torch.no_grad()
def main():
    start = time.monotonic()
    output = RUN / 'tint_diagnostic'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    asset_path = OLD / 'appearance/complete_a6_fitted.npz'
    source_hash = sha_file(asset_path)
    asset = FixedAsset.from_geometry(asset_path).cuda()
    rgb_hash = sha_file(asset_path)
    off, fine = Runtime(fine_road=False), Runtime(fine_road=True)
    assert off.parent_identity == fine.parent_identity
    assert off.repair_state_sha256 == fine.repair_state_sha256
    road_delta = cpu(fine.road_colors - off.road_colors)
    records = []
    for raw in RAWS:
        d = fine.raw(raw)
        reference = cpu(d['rgb'])
        as_image(reference).save(output / f'{raw:03d}_raw.png')
        for delete in (False, True):
            state = 'delete' if delete else 'keep'
            packages = dict(native=off.n.render(off.raw(raw)['cam'], dynamic=not delete))
            scene = off.scene(raw, baseline=True, delete=delete)
            road_rows = scene['rows']['features'][:, 2 + GROUPS.index('road')] > 0
            gated = dict(scene, rows=select_rows(scene['rows'], ~road_rows))
            packages.update(gated_no_road=off.render_rows(raw, gated),
                            road_off=off.render_rows(raw, scene),
                            road_fine=fine.render_scene(raw, baseline=True, delete=delete),
                            asset_off=off.render_scene(raw, asset, delete=delete),
                            asset_fine=fine.render_scene(raw, asset, delete=delete))
            assert torch.allclose(packages['road_off']['rgb'], off.exp.render(raw, keep=not delete)['rgb'], atol=5e-6, rtol=0)
            for a, b in (('road_fine', 'road_off'), ('asset_fine', 'asset_off')):
                assert torch.equal(packages[a]['alpha'], packages[b]['alpha'])
            images = {name: cpu(pkg['rgb']) for name, pkg in packages.items()}
            asset_pkg = packages['asset_fine']
            road_alpha = cpu(asset_pkg['component_alpha_T'][GROUPS.index('road')])
            static_alpha = cpu(asset_pkg['component_alpha_T'][GROUPS.index('legacy_static')])
            actors = cpu(d['actor_labels']) > 0
            road_domain = road_alpha > .05
            seam = binary_dilation(road_domain, iterations=2) & ~binary_erosion(road_domain, iterations=2) & ~actors
            masks = dict(full=np.ones(reference.shape[1:], bool), observed_road=cpu(d['road']).astype(bool),
                         road_interior=(road_alpha > .5) & ~actors,
                         road_static_overlap=(road_alpha > .05) & (static_alpha > .05) & ~actors,
                         road_seam=seam, nonroad=(road_alpha < 1e-6) & ~actors,
                         upper_non_actor=(np.indices(road_alpha.shape)[0] < 160) & ~actors,
                         observed_a6=cpu(d['a6_visible']).astype(bool), observed_a37=cpu(d['a37_visible']).astype(bool))
            metrics = {'raw': {name: stats(reference, reference, mask) for name, mask in masks.items()}}
            environment = {}
            no_environment = {}
            for name, pkg in packages.items():
                env = cpu((1 - pkg['alpha']) * d['env'])
                without = images[name] - env
                environment[name], no_environment[name] = env, without
                metrics[name] = {key: stats(images[name], reference, mask) for key, mask in masks.items()}
                metrics[name]['environment'] = {key: delta_stats(env, mask) for key, mask in masks.items()}
                metrics[name]['without_environment'] = {key: stats(without, reference, mask) for key, mask in masks.items()}
            effects = {}
            for label, after, before in (('static_gate', 'gated_no_road', 'native'),
                                         ('road_layer', 'road_off', 'gated_no_road'),
                                         ('fine_color_native', 'road_fine', 'road_off'),
                                         ('fine_color_asset', 'asset_fine', 'asset_off'),
                                         ('asset_replacement', 'asset_fine', 'road_fine')):
                effects[label] = {key: delta_stats(images[after] - images[before], mask) for key, mask in masks.items()}
            boundary = {name: edge_contrast(image, road_domain, actors) for name, image in dict(raw=reference, **images).items()}
            fine_change = np.abs(images['asset_fine'] - images['asset_off']).max(0)
            alignment = dict(fine_rgb_outside_road_alpha_maxabs=float(fine_change[road_alpha < 1e-6].max(initial=0)),
                             fine_changed_pixels=int((fine_change > 1e-6).sum()),
                             road_domain_pixels=int(road_domain.sum()), road_alpha_mean=float(road_alpha.mean()),
                             max_environment_blue_additivity_error=float(np.abs(blue(images['asset_fine']) - blue(no_environment['asset_fine']) - blue(environment['asset_fine'])).max()))
            record = dict(raw=raw, state=state, metrics=metrics, effects=effects, boundary_contrast=boundary,
                          causal_alignment=alignment, original_RGB_counterfactual_valid=not delete)
            records.append(record)
            for name, image in images.items():
                as_image(image).save(output / f'{raw:03d}_{state}_{name}.png')
            panel(output / f'{raw:03d}_{state}_controls.png', [
                ('Raw RGB / identical camera', reference), ('Native10K', images['native']),
                ('Native + frozen static gates', images['gated_no_road']), ('Road1K / fine off', images['road_off']),
                ('Road1K / fine on', images['road_fine']), ('Complete asset / fine off', images['asset_off']),
                ('Complete asset / fine on', images['asset_fine']), ('Fine color absolute delta x8', np.abs(images['asset_fine'] - images['asset_off']) * 8),
                ('Native environment contribution', environment['native']), ('Asset environment contribution', environment['asset_fine']),
                ('Road alpha*T', road_alpha), ('Legacy-static alpha*T', static_alpha)])
            np.savez_compressed(output / f'{raw:03d}_{state}_exact.npz', raw_RGB=reference,
                                **images, native_environment=environment['native'],
                                asset_environment=environment['asset_fine'], road_alpha_T=road_alpha,
                                legacy_static_alpha_T=static_alpha, road_seam=seam)
            print(json.dumps(dict(raw=raw, state=state, seconds=time.monotonic() - start)), flush=True)
        off._data.clear()
        fine._data.clear()
    assert off.modelunchanged() and fine.modelunchanged() and asset.assert_geometry_unchanged()
    assert sha_file(asset_path) == source_hash == rgb_hash
    report = dict(status='PASS_DIAGNOSTIC', raw_ids=RAWS, training_updates=0, rows=records,
                  seconds=time.monotonic() - start, same_asset_file=str(asset_path), asset_sha256=source_hash,
                  frozen_scene_unchanged=True, geometry_hash=asset.geometry_hash,
                  runtime_off=off.meta, runtime_fine=fine.meta,
                  road_field=dict(rgb_delta_mean=road_delta.mean(0).tolist(),
                                  rgb_delta_min=road_delta.min(0).tolist(), rgb_delta_max=road_delta.max(0).tolist()),
                  source_evidence=dict(runtime=str(OLD / 'source/scene_runtime.py'),
                                       native_interface=str(RUN.parent / 'scene008_native_road1k_20261105_r1/source/native_binding.py'),
                                       static_gates=str(RUN.parent / 'scene008_native_road1k_20261105_r1/source/repair37.py'),
                                       fine_solver=str(RUN.parent / 'scene008_ray_road_colour_fine_20261105/source/refine_colour.py')),
                  definitions=dict(blue_index='B - (R+G)/2 in original renderer RGB units.',
                                   blue_error='Blue index of render minus same-camera original RGB.',
                                   environment='Exact additive renderer term (1-alpha)*env, before display clipping.',
                                   edge_contrast='Mean RGB jump across neighbors crossing fixed road alpha*T=.05, excluding reviewed actors.'),
                  limitations=['Three camera frames only.', 'Original RGB is factual for keep only; delete-region RGB has no counterfactual truth.',
                               'Environment subtraction diagnoses its additive contribution, not physical relighting or white balance.',
                               'No parameters, asset colors, or old checkpoints were changed.'])
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
