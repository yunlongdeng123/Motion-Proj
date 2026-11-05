"""Read-only, joint-depth RGB attribution of M2/a37/road/static/environment.

Run reference and candidate in SEPARATE processes. Only a copy of the existing
24-feature raster input is instrumented; no model, RGB, gate or method changes.
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
FINALIZER = CAMPAIGN/'scene008_blue_finalizer_package_20261105_r1/finalize'
CANDIDATE_RECEIPT = CAMPAIGN/'scene008_blue_global_M2_final_20261105_r1/receipt.json'
RAWS = (113, 130, 137)
GROUPS = ('M2_a6', 'a37', 'road', 'native_static')
ROIS = {'wide': (208, 187, 286, 239), 'silver_tight': (218, 207, 277, 227)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def execute(args):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    if args.output.exists():
        raise FileExistsError('Preserve existing evidence; use a fresh output')
    args.output.mkdir(parents=True)
    (args.output/'float_rois').mkdir()
    (args.output/'layers').mkdir()
    if args.scene == 'reference':
        sys.path.insert(0, str(ASSET_SOURCE))
        from scene_runtime import Runtime
        from sh_asset import SHAsset
        runtime = Runtime(fine_road=True)
        asset = SHAsset.from_geometry(ASSET).cuda().requires_grad_(False)
        make_scene = lambda raw: runtime.scene(raw, asset, delete=True)
        render_scene = lambda raw, scene: runtime.render_rows(raw, scene)
        raster_rows = lambda scene: scene['rows']
        replace_rows = lambda scene, rows: dict(scene, rows=rows)
        environment = lambda scene: scene['env']
        provenance = dict(reference_current_full_pipeline=True, current_scene=runtime.meta,
                          runtime_source_sha256=sha(ASSET_SOURCE/'scene_runtime.py'))
    else:
        sys.path.insert(0, str(args.finalizer_source))
        from final_runtime import FixedM2RoadRuntime
        import calibration_method
        receipt = json.loads(args.candidate_receipt.read_text())
        binding = receipt['binding']
        assert binding['native_iteration'] == 10000 and binding['road_iteration'] == 1000
        runtime = FixedM2RoadRuntime(binding['native_parent'], 10000, binding['road_run'], purpose='final')
        assert runtime.binding_sha256 == receipt['binding_sha256'], 'Not the exact evaluated candidate'
        calibration_method.GRID = (11, 9)
        runtime.load_fine(receipt['fine_binding']['path'], calibration_method.world_basis)
        assert runtime.fine_binding['coefficients_sha256'] == receipt['fine_binding']['coefficients_sha256']
        asset = runtime.asset
        # scene() defaults to uncorrected road baseline. Pass the already loaded
        # fresh fine colour table explicitly, exactly as render_scene() does.
        make_scene = lambda raw: runtime.scene(raw, delete=True, road_colours=runtime.road_colours)
        render_scene = lambda raw, scene: runtime.render(raw, scene)
        raster_rows = lambda scene: scene[0]
        replace_rows = lambda scene, rows: (rows, scene[1], scene[2])
        environment = lambda scene: scene[2]
        provenance = dict(candidate_receipt=str(args.candidate_receipt), candidate_receipt_sha256=sha(args.candidate_receipt),
                          binding=runtime.binding, binding_sha256=runtime.binding_sha256,
                          fine_binding=runtime.fine_binding, old_row_gate_loaded_into_new_parent=False)
    from utils.sh_utils import eval_sh
    assert float(runtime.n.bg.abs().max()) == 0., 'This RGB reconstruction assumes the actual black raster background'
    asset_before = {key: value.detach().cpu().clone() for key, value in
                    (*asset.named_parameters(), *asset.named_buffers())}
    assert len(asset_before) == 27
    asset_sha = sha(ASSET)
    checks = []
    stats_rows = []

    def unchanged(full=False):
        runtime.assert_unchanged()
        if full:
            if args.scene == 'candidate':
                runtime.receipt()  # Its full CPU/input/repair/colour checks.
            else:
                assert runtime.modelunchanged()
            assert sha(ASSET) == asset_sha and asset.assert_geometry_unchanged()
            after = dict((*asset.named_parameters(), *asset.named_buffers()))
            assert set(after) == set(asset_before)
            for key, before in asset_before.items():
                current = after[key].detach().cpu().contiguous()
                assert before.dtype == current.dtype and before.shape == current.shape
                assert torch.equal(before.contiguous().reshape(-1).view(torch.uint8),
                                   current.reshape(-1).view(torch.uint8)), key

    def picture(rgb):
        return Image.fromarray(np.uint8(np.clip(rgb.transpose(1, 2, 0), 0, 1)*255+.5))

    def rgb_stats(rgb):
        value = rgb.astype(np.float64)
        return dict(sum_RGB=value.sum((1, 2)).tolist(), mean_RGB=value.mean((1, 2)).tolist(),
                    max_RGB=value.max((1, 2)).tolist())

    try:
        with torch.no_grad():
            for raw in RAWS:
                data = runtime.raw(raw)
                scene = make_scene(raw)
                base = render_scene(raw, scene)
                original = raster_rows(scene)
                feature = original['features']
                assert feature.shape[1] == 24 and bool((feature[:, 6:] == 0).all())
                flags = feature[:, 2:6]
                assert bool(((flags == 0) | (flags == 1)).all())
                assert bool((flags.sum(1) == 1).all())
                if original['colors_precomp'] is not None:
                    colours = original['colors_precomp']
                    colour_source = 'Existing precomputed RGB, unchanged'
                else:
                    assert runtime.n.static.active_sh_degree == 3 and original['shs'].shape[1] == 16
                    directions = original['means3D']-data['cam'].camera_center
                    directions = directions/directions.norm(dim=-1, keepdim=True)
                    colours = (eval_sh(3, original['shs'].transpose(1, 2), directions)+.5).clamp_min(0)
                    colour_source = 'Native eval_sh3 +.5 lower clamp, only for feature readout; original SH shader unchanged'
                assert bool(torch.isfinite(colours).all())
                marked_rows = dict(original)
                marked_rows['features'] = feature.clone()
                marked_rows['features'][:, 6:18] = (flags[:, :, None]*colours[:, None, :]).reshape(-1, 12)
                marked_scene = replace_rows(scene, marked_rows)
                out = render_scene(raw, marked_scene)
                parity = {key: float((out[key]-base[key]).abs().max())
                          for key in ('rgb', 'rgb_premultiplied', 'alpha', 'depth_sum')}
                parity['four_group_alphaT'] = float((out['extra'][:4]-base['extra'][:4]).abs().max())
                assert max(parity.values()) == 0., (raw, parity)
                layer_rgb = out['extra'][4:16].reshape(4, 3, *out['rgb'].shape[1:])
                group_alpha = out['extra'][:4]
                env_rgb = (1-out['alpha'])*environment(scene)
                reconstructed = layer_rgb.sum(0)+env_rgb
                premult_error = float((layer_rgb.sum(0)-out['rgb_premultiplied']).abs().max())
                total_error = float((reconstructed-out['rgb']).abs().max())
                assert max(premult_error, total_error) < 2e-5, (raw, premult_error, total_error)
                assert float((group_alpha.sum(0)-out['alpha'][0]).abs().max()) < 2e-5
                assert float(group_alpha[1].abs().max()) == 0., 'a37 must remain deleted'
                checks.append(dict(raw=raw, edit='delete', marker_colour_source=colour_source,
                    all_four_markers_added_together_no_image_or_geometry_change_maxabs=parity,
                    group_RGB_sum_minus_premult_maxabs=premult_error,
                    group_RGB_sum_plus_environment_minus_output_maxabs=total_error))
                for name, box in ROIS.items():
                    x0, y0, x1, y1 = box
                    cut = lambda value: value[..., y0:y1, x0:x1].detach().cpu().numpy().astype(np.float32)
                    saved = dict(xyxy=np.asarray(box), rgb=cut(out['rgb']), target_RGB=cut(data['rgb']),
                        rgb_premultiplied=cut(out['rgb_premultiplied']), group_premultiplied_RGB=cut(layer_rgb),
                        component_alpha_T=cut(group_alpha), environment_RGB_contribution=cut(env_rgb),
                        alpha=cut(out['alpha']), depth_sum=cut(out['depth_sum']))
                    observed = data['obs']
                    saved['original_actor_labels'] = observed['actor_labels'][y0:y1, x0:x1].copy()
                    saved['original_reviewed_actor'] = observed['visible_reviewed'].any(0)[y0:y1, x0:x1].copy()
                    saved['road_edit_mask'] = observed['road_edit'][y0:y1, x0:x1].copy()
                    saved['road_edit_RGB_proxy'] = observed['road_edit_rgb'][y0:y1, x0:x1].transpose(2, 0, 1).copy()
                    np.savez_compressed(args.output/'float_rois'/f'raw{raw}_delete_{name}.npz', **saved)
                    record = dict(raw=raw, edit='delete', region=name, xyxy=box, pixels=(x1-x0)*(y1-y0), groups={})
                    rgb_panels = []
                    alpha_panels = []
                    for i, group in enumerate(GROUPS):
                        component = saved['group_premultiplied_RGB'][i]
                        alpha = saved['component_alpha_T'][i]
                        record['groups'][group] = dict(**rgb_stats(component),
                            alphaT_sum=float(alpha.astype(np.float64).sum()), alphaT_mean=float(alpha.mean()), alphaT_max=float(alpha.max()))
                        rgb_panels.append(picture(component))
                        alpha_panels.append(picture(np.repeat(alpha[None], 3, 0)))
                    record['environment'] = dict(**rgb_stats(saved['environment_RGB_contribution']),
                        transmittance_mean=float((1-saved['alpha']).mean()))
                    record['output'] = rgb_stats(saved['rgb'])
                    stats_rows.append(record)
                    rgb_panels += [picture(saved['environment_RGB_contribution']), picture(saved['rgb'])]
                    alpha_panels += [picture(np.repeat(1-saved['alpha'], 3, 0)), picture(np.repeat(saved['alpha'], 3, 0))]
                    names = [*GROUPS, 'environment', 'composite']
                    w, h = (x1-x0)*4, (y1-y0)*4
                    cell = max(w, 230)
                    canvas = Image.new('RGB', (cell*6, 2*(h+30)+48), 'white')
                    draw = ImageDraw.Draw(canvas)
                    draw.text((6, 5), f'{args.scene.upper()} / raw{raw} DELETE / {name} / x{x0}:{x1}, y{y0}:{y1}', fill='black')
                    draw.text((6, 22), 'True joint-depth premultiplied RGB above; alpha*T / environment T below. Diagnostic only.', fill='black')
                    for i, label in enumerate(names):
                        rgb_panels[i].save(args.output/'layers'/f'raw{raw}_delete_{name}_{label}_RGB.png')
                        alpha_panels[i].save(args.output/'layers'/f'raw{raw}_delete_{name}_{label}_alpha.png')
                        for row, panel in enumerate((rgb_panels[i], alpha_panels[i])):
                            yy = 48+row*(h+30)
                            draw.text((i*cell+6, yy), f'{label} / {"RGB" if row == 0 else "alphaT or T"}', fill='black')
                            canvas.paste(panel.resize((w, h), Image.Resampling.NEAREST), (i*cell+(cell-w)//2, yy+24))
                    canvas.save(args.output/f'raw{raw}_delete_{name}_layers.png')
                unchanged()
                print(f'LAYERS {args.scene} raw{raw} delete', flush=True)
                del base, out, scene, marked_scene, original, marked_rows, feature, flags, colours, layer_rgb
    finally:
        unchanged(full=True)
    write(args.output/'metrics.json', dict(scene=args.scene, group_order=GROUPS, checks=checks, rows=stats_rows,
          all_ROI_pixels_included=True, masks_saved_for_interpretation=True,
          target_role='Observed RGB is not deleted-actor/unknown-surface truth; road_edit_RGB is a proxy.', diagnostic_only=True))
    write(args.output/'manifest.json', dict(status='PASS_READ_ONLY_LAYER_ATTRIBUTION', scene=args.scene,
          provenance=provenance, M2_sha256=asset_sha, M2_all_27_tensors_bitwise_unchanged_CPU=True,
          models_road_fine_unchanged=True, Parameters_or_Runtime_versions_never_modified=True,
          only_modified_input='Clone of joint raster features, slots6:18 = four groups times3 shader RGB',
          original_group_alphaT_slots=[2,3,4,5], RGB_marker_slots=list(range(6,18)),
          raw_frames=RAWS, edit='delete', ROIs=ROIS, group_order=GROUPS,
          true_single_joint_depth_order=True, all_four_RGB_markers_no_RGB_alpha_depth_change_exact=True,
          decomposition_tolerance=2e-5, script_sha256=sha(__file__), training_updates=0, diagnostic_only=True))
    print(json.dumps(dict(output=str(args.output), status='PASS_READ_ONLY_LAYER_ATTRIBUTION', training_updates=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('reference','candidate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--finalizer-source', type=Path, default=FINALIZER)
    parser.add_argument('--candidate-receipt', type=Path, default=CANDIDATE_RECEIPT)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute(args)
    else:
        print(json.dumps(dict(scene=args.scene, raw_frames=RAWS, edit='delete', ROIs=ROIS,
            independent_process_required=True, original_models_unchanged=True, group_RGB_slots=[6,18],
            candidate_receipt=str(args.candidate_receipt) if args.scene=='candidate' else None,
            output=str(args.output), execute=False, training_updates=0), indent=2))


if __name__ == '__main__':
    main()
