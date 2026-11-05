"""Zero-training render counterfactuals on the CURRENT full native10K/M2/road/fine.

Only joint raster kwargs copies change. Model tensors, Parameter bindings,
Runtime version expectations, complete M2, road, and fine colours never change.
Default is a plan. CUDA rendering requires --execute after source review.
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
LINEAGE = 64990
EXPECTED_FAMILY = 64
RAWS = (113, 137)
BOXES = {113: {'user_box': (276, 175, 309, 220), 'white_line': (287, 182, 299, 215)},
         137: {'right_patch': (276, 175, 309, 220), 'environment_band': (210, 213, 290, 233)}}
LABELS = {'normal': 'NORMAL / current complete pipeline',
          'hide_family': 'HIDE lineage64990 / all64 excluded from raster',
          'clean_SH': 'OLD CLEAN SH / same64 geometry + opacity'}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(args):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(ASSET_SOURCE))
    from scene_runtime import Runtime
    from sh_asset import SHAsset
    from utils.sh_utils import eval_sh
    if args.output.exists():
        raise FileExistsError('Use a fresh output; existing evidence is preserved')
    args.output.mkdir(parents=True)
    (args.output/'float_rois').mkdir()
    (args.output/'crops').mkdir()
    runtime = Runtime(fine_road=True)
    asset = SHAsset.from_geometry(ASSET).cuda().requires_grad_(False)
    static = runtime.n.static
    lineage = getattr(static, '_sgcm_lineage_ids', None)
    if lineage is None or len(lineage) != len(static.get_xyz):
        raise ValueError('Static persistent lineage IDs required; never use a guessed row offset')
    ids = torch.nonzero(lineage == LINEAGE, as_tuple=False).flatten()
    if len(ids) != EXPECTED_FAMILY:
        raise ValueError(f'Expected CURRENT native10K family64, got {len(ids)}')
    nstatic = len(static.get_xyz)
    states = ['normal', 'hide_family']+(['clean_SH'] if args.with_clean_sh else [])
    asset_before = {key: value.detach().cpu().clone() for key, value in
                    (*asset.named_parameters(), *asset.named_buffers())}
    assert len(asset_before) == 27
    family_before = {key: getattr(static, key)[ids].detach().cpu().clone() for key in
                     ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation',
                      '_opacity', '_t', '_scaling_t', '_velocity')}
    asset_sha = sha(ASSET)
    clean = clean_source = None
    if args.with_clean_sh:
        clean_source = json.loads(args.clean_sh_json.read_text())
        assert clean_source['lineage'] == LINEAGE and clean_source['source_row_identity'] == 65911
        assert clean_source['checkpoint_iteration'] == 10000 and len(clean_source['checkpoint_row_ids']) == 1
        source = Path(clean_source['checkpoint'])
        stat = source.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (clean_source['checkpoint_bytes'], clean_source['checkpoint_mtime_ns'])
        dc = np.asarray(clean_source['dc'], np.float32)
        rest = np.asarray(clean_source['rest'], np.float32)
        assert hashlib.sha256(dc.tobytes()+rest.tobytes()).hexdigest() == clean_source['tensor_sha256']
        assert dc.shape == (1, 1, 3) and rest.shape == (1, 15, 3)
        assert np.max(dc*.28209479177387814+.5) < 0 and np.max(np.abs(rest)) == 0
        clean = torch.cat((torch.as_tensor(dc, device='cuda'), torch.as_tensor(rest, device='cuda')), 1)
    rows_summary = []
    marker_checks = []
    appearance_checks = []

    def unchanged(full=False):
        runtime.assert_unchanged()
        if full:
            assert runtime.modelunchanged()
            assert sha(ASSET) == asset_sha and asset.assert_geometry_unchanged()
            current = dict((*asset.named_parameters(), *asset.named_buffers()))
            assert set(current) == set(asset_before)
            for key, before in asset_before.items():
                after = current[key].detach().cpu().contiguous()
                assert before.dtype == after.dtype and before.shape == after.shape
                assert torch.equal(before.contiguous().reshape(-1).view(torch.uint8), after.reshape(-1).view(torch.uint8)), key
            for key, before in family_before.items():
                assert torch.equal(getattr(static, key)[ids].cpu(), before), key

    def counterfactual(scene, state):
        # Runtime.scene concatenates static first, then complete M2, a37, road.
        # merge_rows retains input row order. IDs address only its first N rows.
        original = scene['rows']
        assert len(original['means3D']) >= nstatic
        assert scene['meta']['native_a6_rows_input'] == 0
        assert scene['meta']['asset_rows'] == len(asset.xyz)
        result = dict(original)
        if state == 'hide_family':
            result['mask'] = original['mask'].clone()
            result['mask'][ids] = False
        elif state == 'clean_SH':
            if original['shs'] is not None:
                assert original['shs'].shape[1] == 16 and clean.shape[1] == 16
                result['shs'] = original['shs'].clone()
                result['shs'][ids] = clean.expand(len(ids), -1, -1)
            else:
                result['colors_precomp'] = original['colors_precomp'].clone()
                direction = original['means3D'][ids]-scene['cam'].camera_center
                direction = direction/direction.norm(dim=-1, keepdim=True)
                colour = (eval_sh(3, clean.expand(len(ids), -1, -1).transpose(1, 2), direction)+.5).clamp_min(0)
                result['colors_precomp'][ids] = colour
        return dict(scene, rows=result)

    def marked(scene):
        original = scene['rows']
        rows = dict(original)
        feature = original['features'].clone()
        assert feature.shape[1] == 24 and bool((feature[:, 6:] == 0).all())
        if original['colors_precomp'] is not None:
            colour = original['colors_precomp'][ids]
        else:
            direction = original['means3D'][ids]-scene['cam'].camera_center
            direction = direction/direction.norm(dim=-1, keepdim=True)
            colour = (eval_sh(static.active_sh_degree, original['shs'][ids].transpose(1, 2), direction)+.5).clamp_min(0)
        feature[ids, 6] = 1  # extra[4] = exact family alpha*T in the full scene.
        feature[ids, 7:10] = colour  # extra[5:8] = family premultiplied RGB.
        rows['features'] = feature
        return dict(scene, rows=rows), colour

    def image(rgb):
        return Image.fromarray(np.uint8(np.clip(rgb.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)*255+.5))

    def blue(error):
        return error[2]-.5*(error[0]+error[1])

    def error_stats(rgb, target, mask):
        e = rgb-target
        b = blue(e)
        count = int(mask.sum())
        return dict(pixels=count, RGB_MAE=float(np.abs(e[:, mask]).mean()) if count else None,
                    signed_blue_error=float(b[mask].mean()) if count else None,
                    absolute_blue_error=float(np.abs(b[mask]).mean()) if count else None,
                    max_positive_blue_error=float(b[mask].max()) if count else None,
                    positive_blue_gt_05_pixels=int((b[mask] > .05).sum()))

    try:
        with torch.no_grad():
            for raw in RAWS:
                data = runtime.raw(raw)
                for delete in (False, True):
                    edit = 'delete' if delete else 'keep'
                    base_scene = runtime.scene(raw, asset, delete=delete)
                    # The original complete pipeline and every original model
                    # tensor remain intact while kwargs-only variants render.
                    normal = None
                    panels = {}
                    roi_banks = {}
                    for state in states:
                        scene = counterfactual(base_scene, state)
                        unmarked = runtime.render_rows(raw, scene)
                        instrumented, colour = marked(scene)
                        out = runtime.render_rows(raw, instrumented)
                        parity = {key: float((out[key]-unmarked[key]).abs().max()) for key in ('rgb', 'alpha', 'depth_sum')}
                        parity['group_alpha_T'] = float((out['extra'][:4]-unmarked['extra'][:4]).abs().max())
                        assert max(parity.values()) == 0., (raw, edit, state, parity)
                        marker_checks.append(dict(raw=raw, edit=edit, state=state, maxabs=parity))
                        assert bool(torch.isfinite(out['rgb']).all())
                        if state == 'normal':
                            normal = out
                        if state == 'hide_family':
                            assert float(out['extra'][4].abs().max()) == 0.
                        if state == 'clean_SH':
                            assert float(colour.abs().max()) == 0.
                            for key in ('alpha', 'depth_sum'):
                                assert torch.equal(out[key], normal[key]), key
                            assert torch.equal(out['extra'][:5], normal['extra'][:5])
                            # Same geometry/occlusion: only this family's native
                            # SH RGB is removed. Feature and shader SH arithmetic
                            # may differ by normal FP32 roundoff.
                            residual = normal['rgb']-out['rgb']-normal['extra'][5:8]
                            residual_max = float(residual.abs().max())
                            assert residual_max < 5e-6, residual_max
                            appearance_checks.append(dict(raw=raw, edit=edit,
                                normal_minus_cleanSH_minus_familyRGB_maxabs=residual_max,
                                alpha_depth_group_family_alphaT_exactly_equal=True))
                        frame = image(out['rgb'])
                        frame.save(args.output/f'raw{raw}_{edit}_{state}.png')
                        panels[state] = frame
                        record = dict(raw=raw, edit=edit, state=state, timestamp=float(data['cam'].timestamp),
                                      family_rows=ids.cpu().tolist(), family_shader_RGB_min=colour.min(0).values.cpu().tolist(),
                                      family_shader_RGB_max=colour.max(0).values.cpu().tolist(), ROIs={})
                        roi_banks[state] = {}
                        for name, box in BOXES[raw].items():
                            x0, y0, x1, y1 = box
                            cut = lambda value: value[..., y0:y1, x0:x1].detach().cpu().numpy().astype(np.float32)
                            saved = dict(xyxy=np.asarray(box), rgb=cut(out['rgb']), target=cut(data['rgb']),
                                         alpha=cut(out['alpha']), depth_sum=cut(out['depth_sum']),
                                         component_alpha_T=cut(out['extra'][:4]), family_alpha_T=cut(out['extra'][4:5]),
                                         family_premultiplied_RGB=cut(out['extra'][5:8]),
                                         environment_RGB_contribution=cut((1-out['alpha'])*data['env']))
                            original_actors = (data['obs']['actor_labels'] > 0) | data['obs']['visible_reviewed'].any(0)
                            saved['original_observed_background'] = ~original_actors[y0:y1, x0:x1]
                            np.savez_compressed(args.output/'float_rois'/f'raw{raw}_{edit}_{state}_{name}.npz', **saved)
                            roi_banks[state][name] = saved
                            base = roi_banks['normal'][name]
                            common = saved['original_observed_background'] & (base['component_alpha_T'][0] <= .01) & (saved['component_alpha_T'][0] <= .01)
                            record['ROIs'][name] = dict(xyxy=box,
                                whole_box_family_alpha_T_sum=float(saved['family_alpha_T'].sum()),
                                whole_box_family_alpha_T_max=float(saved['family_alpha_T'].max()),
                                whole_box_family_premultiplied_RGB_sum=saved['family_premultiplied_RGB'].sum((1, 2)).tolist(),
                                whole_box_family_blue_contribution_sum=float(blue(saved['family_premultiplied_RGB']).sum()),
                                observed_background_common_mask='Original actors and M2 alphaT>.01 in EITHER endpoint excluded',
                                normal_error=error_stats(base['rgb'], saved['target'], common),
                                counterfactual_error=error_stats(saved['rgb'], saved['target'], common),
                                whole_box_RGB_change_max=float(np.abs(saved['rgb']-base['rgb']).max()))
                        rows_summary.append(record)
                        unchanged()
                        print(f'RENDER raw{raw} {edit} {state}', flush=True)
                        del unmarked, scene, instrumented
                    # Labelled full views and the exact same renderer-coordinate
                    # crops make residual small strips directly reviewable.
                    for name, box in {'full': (0, 0, 480, 320), **BOXES[raw]}.items():
                        x0, y0, x1, y1 = box
                        scale = 1 if name == 'full' else 8
                        w, h = (x1-x0)*scale, (y1-y0)*scale
                        cell = max(480, w)
                        canvas = Image.new('RGB', (cell*len(states), h+64), 'white')
                        draw = ImageDraw.Draw(canvas)
                        for i, state in enumerate(states):
                            draw.text((i*cell+6, 5), LABELS[state], fill='black')
                            draw.text((i*cell+6, 22), f'CURRENT native10K + fixed M2 + road1K + fine / raw{raw} / a37 {edit.upper()}', fill='black')
                            draw.text((i*cell+6, 39), f'{name}: x{x0}:{x1}, y{y0}:{y1} / kwargs-only counterfactual', fill='black')
                            crop = panels[state].crop(box).resize((w, h), Image.Resampling.NEAREST)
                            canvas.paste(crop, (i*cell+(cell-w)//2, 64))
                        canvas.save(args.output/'crops'/f'raw{raw}_{edit}_{name}_comparison.png')
                    del normal, out, base_scene
    finally:
        # No original tensor was written, rebound or exempted from Runtime's
        # ordinary checks. Verify exact preservation even when a render fails.
        unchanged(full=True)
    write(args.output/'metrics.json', dict(float_before_PNG=True, rows=rows_summary,
          marker_render_parity=marker_checks, clean_SH_appearance_parity=appearance_checks,
          scope='Observed background error only. Full-box radiance/contribution is diagnostic, not unknown-surface RGB truth.',
          counterfactual_is_not_a_physical_repair=True, quality_accepted=False))
    write(args.output/'manifest.json', dict(status='PASS_READ_ONLY_COUNTERFACTUAL', native_pipeline=runtime.meta,
          asset_sha256=asset_sha, M2_all_27_tensors_bitwise_unchanged_CPU=True,
          native_and_road_and_fine_modelunchanged=True, original_family_all_parameters_exactly_unchanged_CPU=True,
          model_Parameter_or_Runtime_version_bindings_changed=False,
          lineage=LINEAGE, source_original_row=65911, family_rows=ids.cpu().tolist(), family_count=len(ids),
          hide_policy='Only cloned joint raster mask: all64 family rows False; no model opacity change',
          clean_SH_policy='Only cloned joint raster SHs/colours: old clean same-source lineage DC/rest; geometry/time/opacity unchanged',
          clean_SH_reference=clean_source, clean_SH_json_sha256=sha(args.clean_sh_json) if clean_source else None,
          group_order=['M2_a6', 'native_a37', 'road', 'native_static'],
          family_alphaT_feature=6, family_premultiplied_RGB_features=[7, 8, 9],
          states=states, raws=RAWS, keep_delete='Only a37 is edited; complete M2/road/fine stay enabled',
          own_raw_camera_verified_by_Runtime=True, raster='Original Runtime.render_rows; one true depth order',
          marker_RGB_alpha_depth_group_exact_parity=True, boxes_xyxy_exclusive=BOXES,
          script_sha256=sha(__file__), training_updates=0, quality_accepted=False))
    print(json.dumps(dict(output=str(args.output), status='PASS_READ_ONLY_COUNTERFACTUAL', training_updates=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-clean-sh', action='store_true')
    parser.add_argument('--clean-sh-json', type=Path, default=Path(__file__).with_name('old_clean_lineage64990_sh.json'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute(args)
    else:
        print(json.dumps(dict(current_full_pipeline=True, lineage=LINEAGE, expected_family=EXPECTED_FAMILY,
              states=['normal', 'hide_family']+(['clean_SH'] if args.with_clean_sh else []), raws=RAWS,
              edits=['keep', 'delete'], output=str(args.output), training_updates=0,
              modify_only_joint_kwargs_copies=True, modify_models_or_Runtime_version_expectations=False,
              execute=False), indent=2))


if __name__ == '__main__':
    main()
