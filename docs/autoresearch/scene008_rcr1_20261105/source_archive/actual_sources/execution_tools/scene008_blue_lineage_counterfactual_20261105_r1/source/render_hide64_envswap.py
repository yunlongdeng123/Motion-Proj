"""Zero-training diagnostic: hide64/current-env versus hide64/old-clean-env.

Current native10K + complete M2 + road1K + fine remain enabled. An independent
auxiliary EnvLight samples the same camera directions. Only scene['env'] is
replaced; no Runtime model, Parameter, version expectation or asset changes.
Old env is a counterfactual reference, NOT a new method or a quality repair.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ASSET_SOURCE = CAMPAIGN/'scene008_a6_asset_controls_20261105/source'
ASSET = ASSET_SOURCE.parent/'m2/complete_a6.npz'
OLD_ENV = CAMPAIGN/'scene008_a37_only_B10k_20261103_r1/formal10k/env_light_chkpnt10000.pth'
REFERENCE_ROIS = CAMPAIGN/'scene008_blue_lineage_counterfactual_20261105_r1/result/float_rois'
RAWS = (113, 137)
BOXES = {113: {'user_box': (276, 175, 309, 220), 'white_line': (287, 182, 299, 215)},
         137: {'right_patch': (276, 175, 309, 220), 'environment_band': (210, 213, 290, 233)}}
RESIDUAL8 = ((211, 213), (286, 213), (210, 214), (288, 223),
             (288, 224), (288, 226), (289, 226), (288, 227))
STATES = ('current_env', 'old_clean_env')
LABELS = {'current_env': 'HIDE64 / CURRENT native environment',
          'old_clean_env': 'HIDE64 / OLD CLEAN environment / diagnostic only'}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def execute(args):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(ASSET_SOURCE))
    from scene_runtime import Runtime
    from sh_asset import SHAsset
    from scene import EnvLight
    if args.output.exists():
        raise FileExistsError('Use a fresh output; existing evidence is preserved')
    args.output.mkdir(parents=True)
    (args.output/'float_rois').mkdir()
    (args.output/'crops').mkdir()
    runtime = Runtime(fine_road=True)
    asset = SHAsset.from_geometry(ASSET).cuda().requires_grad_(False)
    lineage = getattr(runtime.n.static, '_sgcm_lineage_ids', None)
    if lineage is None:
        raise ValueError('Persistent static lineage is required')
    ids = torch.nonzero(lineage == 64990, as_tuple=False).flatten()
    assert len(ids) == 64 and len(lineage) == len(runtime.n.static.get_xyz)
    asset_before = {key: value.detach().cpu().clone() for key, value in
                    (*asset.named_parameters(), *asset.named_buffers())}
    assert len(asset_before) == 27
    asset_sha = sha(ASSET)
    current_env_identity = (id(runtime.n.env), id(runtime.n.env.base), runtime.n.env.base._version)
    old_env_identity = identity(args.old_env_checkpoint)
    old_env_sha = sha(args.old_env_checkpoint)
    old_cameras_path = args.old_env_checkpoint.parent/'cameras.json'
    old_cameras = json.loads(old_cameras_path.read_text())
    old_camera_sha = sha(old_cameras_path)

    # Actual EnvLight schema: checkpoint=(capture,step), capture=(base,opt_dict).
    # restore(training_args=None) sets base only; it creates no Adam optimizer.
    payload, step = torch.load(str(args.old_env_checkpoint), map_location='cpu', weights_only=False, mmap=True)
    assert int(step) == 10000 and len(payload) == 2
    base, optimizer_state = payload
    assert base.dtype == torch.float32 and tuple(base.shape) == (6, 1024, 1024, 3)
    assert base.shape == runtime.n.env.base.shape and isinstance(optimizer_state, dict)
    assert bool(torch.isfinite(base).all())
    auxiliary = EnvLight(resolution=int(base.shape[1])).cuda()
    auxiliary.restore((torch.nn.Parameter(base.detach().cuda(), requires_grad=False), optimizer_state), training_args=None)
    auxiliary.requires_grad_(False)
    assert getattr(auxiliary, 'optimizer', None) is None
    assert auxiliary is not runtime.n.env and auxiliary.base.data_ptr() != runtime.n.env.base.data_ptr()
    aux_versions = (id(auxiliary.base), auxiliary.base._version, id(auxiliary.to_opengl), auxiliary.to_opengl._version)
    del payload, base, optimizer_state
    checks = []
    replay_checks = []
    metrics = []
    fixed_pixels = []

    def unchanged(full=False):
        runtime.assert_unchanged()
        assert current_env_identity == (id(runtime.n.env), id(runtime.n.env.base), runtime.n.env.base._version)
        assert aux_versions == (id(auxiliary.base), auxiliary.base._version, id(auxiliary.to_opengl), auxiliary.to_opengl._version)
        assert identity(args.old_env_checkpoint) == old_env_identity
        if full:
            assert runtime.modelunchanged() and sha(ASSET) == asset_sha and asset.assert_geometry_unchanged()
            assert sha(args.old_env_checkpoint) == old_env_sha and sha(old_cameras_path) == old_camera_sha
            current = dict((*asset.named_parameters(), *asset.named_buffers()))
            assert set(current) == set(asset_before)
            for key, before in asset_before.items():
                after = current[key].detach().cpu().contiguous()
                assert before.dtype == after.dtype and before.shape == after.shape
                assert torch.equal(before.contiguous().reshape(-1).view(torch.uint8), after.reshape(-1).view(torch.uint8)), key

    def image(rgb):
        return Image.fromarray(np.uint8(np.clip(rgb.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)*255+.5))

    def blue(rgb):
        return rgb[2]-.5*(rgb[0]+rgb[1])

    def stats(rgb, target, mask):
        error = rgb-target
        b = blue(error)
        count = int(mask.sum())
        return dict(pixels=count, RGB_MAE=float(np.abs(error[:, mask]).mean()) if count else None,
                    signed_blue_error=float(b[mask].mean()) if count else None,
                    absolute_blue_error=float(np.abs(b[mask]).mean()) if count else None,
                    max_positive_blue_error=float(b[mask].max()) if count else None,
                    positive_blue_gt_05_pixels=int((b[mask] > .05).sum()))

    try:
        with torch.no_grad():
            for raw in RAWS:
                data = runtime.raw(raw)
                old_camera = next(row for row in old_cameras if row['img_name'] == f'0008{raw:03d}')
                current_camera = next(row for row in runtime.n.records if row['img_name'] == f'0008{raw:03d}')
                for key in ('rotation', 'position', 'cx', 'cy', 'fx', 'fy'):
                    np.testing.assert_allclose(old_camera[key], current_camera[key], atol=1e-9, rtol=0)
                directions = data['cam'].get_world_directions(False).permute(1, 2, 0)
                current_replay = runtime.n.env(directions).permute(2, 0, 1)
                assert torch.equal(current_replay, data['env']), 'Current camera ray sampling changed'
                old_env = auxiliary(directions).permute(2, 0, 1).detach()
                assert bool(torch.isfinite(old_env).all())
                for delete in (False, True):
                    edit = 'delete' if delete else 'keep'
                    original_scene = runtime.scene(raw, asset, delete=delete)
                    assert original_scene['meta']['native_a6_rows_input'] == 0
                    assert original_scene['meta']['asset_rows'] == len(asset.xyz)
                    rows = dict(original_scene['rows'])
                    rows['mask'] = rows['mask'].clone()
                    rows['mask'][ids] = False
                    unmarked_scene = dict(original_scene, rows=rows)
                    unmarked = runtime.render_rows(raw, unmarked_scene)
                    marked_rows = dict(rows)
                    marked_rows['features'] = rows['features'].clone()
                    assert bool((marked_rows['features'][:, 6:] == 0).all())
                    marked_rows['features'][ids, 6] = 1
                    current_scene = dict(original_scene, rows=marked_rows)
                    old_scene = dict(current_scene, env=old_env)
                    outputs = {'current_env': runtime.render_rows(raw, current_scene),
                               'old_clean_env': runtime.render_rows(raw, old_scene)}
                    parity = {}
                    for key in ('rgb_premultiplied', 'alpha', 'depth_sum'):
                        parity[key] = float((outputs['current_env'][key]-outputs['old_clean_env'][key]).abs().max())
                        assert parity[key] == 0., (raw, edit, key)
                        assert torch.equal(outputs['current_env'][key], unmarked[key]), ('marker', key)
                    parity['four_component_alpha_T'] = float((outputs['current_env']['extra'][:4]-outputs['old_clean_env']['extra'][:4]).abs().max())
                    assert parity['four_component_alpha_T'] == 0.
                    assert torch.equal(outputs['current_env']['extra'][:4], unmarked['extra'][:4])
                    assert float(outputs['current_env']['extra'][4].abs().max()) == 0.
                    assert float(outputs['old_clean_env']['extra'][4].abs().max()) == 0.
                    predicted = (1-outputs['current_env']['alpha'])*(old_env-data['env'])
                    delta_error = outputs['old_clean_env']['rgb']-outputs['current_env']['rgb']-predicted
                    delta_max = float(delta_error.abs().max())
                    assert delta_max < 1e-6, delta_max
                    checks.append(dict(raw=raw, edit=edit, geometry_and_premult_maxabs=parity,
                                       marker_premult_alpha_depth_components_exact=True,
                                       output_delta_minus_transmittance_env_delta_maxabs=delta_max))
                    frames = {}
                    roi_banks = {state: {} for state in STATES}
                    for state in STATES:
                        out = outputs[state]
                        environment = data['env'] if state == 'current_env' else old_env
                        assert bool(torch.isfinite(out['rgb']).all())
                        frames[state] = image(out['rgb'])
                        frames[state].save(args.output/f'raw{raw}_{edit}_hide64_{state}.png')
                        record = dict(raw=raw, edit=edit, state=state, family_hidden_in_both=True, ROIs={})
                        for name, box in BOXES[raw].items():
                            x0, y0, x1, y1 = box
                            cut = lambda value: value[..., y0:y1, x0:x1].detach().cpu().numpy().astype(np.float32)
                            saved = dict(xyxy=np.asarray(box), rgb=cut(out['rgb']), rgb_premultiplied=cut(out['rgb_premultiplied']),
                                         target=cut(data['rgb']), alpha=cut(out['alpha']), depth_sum=cut(out['depth_sum']),
                                         component_alpha_T=cut(out['extra'][:4]), family_alpha_T=cut(out['extra'][4:5]),
                                         family_premultiplied_RGB=cut(out['extra'][5:8]),
                                         environment_RGB=cut(environment),
                                         environment_RGB_contribution=cut((1-out['alpha'])*environment))
                            references = {}
                            for previous in ('normal', 'hide_family', 'clean_SH'):
                                with np.load(args.reference_rois/f'raw{raw}_{edit}_{previous}_{name}.npz', allow_pickle=False) as f:
                                    references[previous] = {key: f[key].copy() for key in
                                            ('xyxy', 'rgb', 'target', 'alpha', 'depth_sum', 'component_alpha_T', 'original_observed_background')}
                            before = references['hide_family']
                            np.testing.assert_array_equal(saved['xyxy'], before['xyxy'])
                            np.testing.assert_array_equal(saved['target'], before['target'])
                            common = before['original_observed_background'].copy()
                            for reference in references.values():
                                np.testing.assert_array_equal(reference['target'], saved['target'])
                                common &= reference['component_alpha_T'][0] <= .01
                            common &= saved['component_alpha_T'][0] <= .01
                            saved['original_observed_background'] = before['original_observed_background']
                            saved['common_observed_background'] = common
                            if state == 'current_env':
                                for key in ('rgb', 'alpha', 'depth_sum', 'component_alpha_T'):
                                    np.testing.assert_array_equal(saved[key], before[key], err_msg=f'Current-env replay {raw}/{edit}/{name}/{key}')
                                replay_checks.append(dict(raw=raw, edit=edit, region=name, exact_previous_hide64_replay=True))
                            np.savez_compressed(args.output/'float_rois'/f'raw{raw}_{edit}_hide64_{state}_{name}.npz', **saved)
                            roi_banks[state][name] = saved
                            record['ROIs'][name] = stats(saved['rgb'], saved['target'], common)
                            if raw == 137 and delete and name == 'environment_band':
                                assert int(common.sum()) == 66
                                for x, y in RESIDUAL8:
                                    j, i = y-y0, x-x0
                                    assert common[j, i]
                                    rgb = saved['rgb'][:, j, i].astype(np.float64)
                                    target = saved['target'][:, j, i].astype(np.float64)
                                    E = saved['environment_RGB_contribution'][:, j, i].astype(np.float64)
                                    G = rgb-E
                                    alpha = float(saved['alpha'][0, j, i])
                                    T = 1-alpha
                                    env_excess = float(blue(E)-T*blue(target))
                                    gaussian_excess = float(blue(G)-alpha*blue(target))
                                    total = float(blue(rgb-target))
                                    fixed_pixels.append(dict(state=state, x=x, y=y, RGB=rgb.tolist(), target_RGB=target.tolist(),
                                         alpha=alpha, transmittance=T, component_alphaT=saved['component_alpha_T'][:, j, i].tolist(),
                                         family_alphaT=float(saved['family_alpha_T'][0, j, i]), environment_RGB_contribution=E.tolist(),
                                         environment_excess=env_excess,Gaussian_excess=gaussian_excess,total_blue_error=total,
                                         decomposition_residual=total-env_excess-gaussian_excess))
                        metrics.append(record)
                        print(f'RENDER raw{raw} {edit} hide64 {state}', flush=True)
                    for name, box in {'full': (0, 0, 480, 320), **BOXES[raw]}.items():
                        x0, y0, x1, y1 = box
                        scale = 1 if name == 'full' else 8
                        w, h = (x1-x0)*scale, (y1-y0)*scale
                        cell = max(540, w)
                        canvas = Image.new('RGB', (cell*2, h+66), 'white')
                        draw = ImageDraw.Draw(canvas)
                        for i, state in enumerate(STATES):
                            draw.text((i*cell+6, 5), LABELS[state], fill='black')
                            draw.text((i*cell+6, 22), f'CURRENT native10K + fixed M2 + road1K + fine / raw{raw} / a37 {edit.upper()}', fill='black')
                            draw.text((i*cell+6, 39), f'{name}: x{x0}:{x1}, y{y0}:{y1} / NOT a new method or quality repair', fill='black')
                            canvas.paste(frames[state].crop(box).resize((w, h), Image.Resampling.NEAREST),
                                         (i*cell+(cell-w)//2, 66))
                        canvas.save(args.output/'crops'/f'raw{raw}_{edit}_{name}_envswap_comparison.png')
                    unchanged()
                    del outputs, unmarked, original_scene, current_scene, old_scene
    finally:
        unchanged(full=True)
    fixed_summary = {}
    for state in STATES:
        points = [point for point in fixed_pixels if point['state'] == state]
        assert len(points) == 8
        fixed_summary[state] = dict(blue_error_sum=sum(point['total_blue_error'] for point in points),
                absolute_blue_error_sum=sum(abs(point['total_blue_error']) for point in points),
                positive_blue_gt_05_pixels=sum(point['total_blue_error'] > .05 for point in points),
                max_positive_blue_error=max(point['total_blue_error'] for point in points))
    write(args.output/'fixed_residual8.json', dict(raw=137, edit='delete', coordinates_fixed_before_env_swap=RESIDUAL8,
          blue_definition='B-(R+G)/2', environment_excess='blue(T*env)-T*blue(target)',
          Gaussian_excess='blue(rgb-T*env)-alpha*blue(target)', group_order=['M2_a6','a37','road','native_static'],
          states=fixed_summary, pixels=fixed_pixels, quality_accepted=False))
    write(args.output/'metrics.json', dict(rows=metrics, exact_geometry_and_premult_checks=checks,
          exact_current_environment_replay_checks=replay_checks, float_before_PNG=True, quality_accepted=False))
    write(args.output/'manifest.json', dict(status='PASS_READ_ONLY_ENV_SWAP_DIAGNOSTIC', native_pipeline=runtime.meta,
          M2_sha256=asset_sha, M2_all_27_tensors_bitwise_unchanged_CPU=True,
          native_and_road_and_fine_modelunchanged=True, Runtime_env_object_Parameter_and_version_unchanged=True,
          auxiliary_EnvLight_separate=True, auxiliary_base_versions_unchanged=True, no_optimizer_for_auxiliary=True,
          old_env_checkpoint=old_env_identity, old_env_sha256=old_env_sha,
          old_cameras_sha256=old_camera_sha, env_class_source_sha256=sha(EnvLight.restore.__code__.co_filename),
          env_schema=dict(iteration=10000, payload=['base','optimizer_state'], base_shape=[6,1024,1024,3], dtype='float32', restore_training_args=None),
          only_swap='scene.env, sampled by independent old-clean EnvLight on SAME current camera.get_world_directions(False)',
          family_hidden_in_both_states=True, family=64990, family_count=64, hide_only_kwargs_mask=True,
          reference_float_rois=str(args.reference_rois), raws=RAWS, edits=['keep','delete'],
          fixed_residual8=RESIDUAL8, boxes_xyxy_exclusive=BOXES,
          physical_raster_unchanged=True, script_sha256=sha(__file__), training_updates=0,
          old_env_is_not_a_new_method_or_quality_repair=True, quality_accepted=False))
    print(json.dumps(dict(output=str(args.output), status='PASS_READ_ONLY_ENV_SWAP_DIAGNOSTIC', training_updates=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-env-checkpoint', type=Path, default=OLD_ENV)
    parser.add_argument('--reference-rois', type=Path, default=REFERENCE_ROIS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute(args)
    else:
        print(json.dumps(dict(states=STATES, family_hidden_in_both=True, raws=RAWS, edits=['keep','delete'],
              only_swap='scene.env using independent auxiliary EnvLight', old_env_checkpoint=str(args.old_env_checkpoint),
              reference_rois=str(args.reference_rois), fixed_residual8=RESIDUAL8,
              output=str(args.output), execute=False, training_updates=0, quality_accepted=False), indent=2))


if __name__ == '__main__':
    main()
