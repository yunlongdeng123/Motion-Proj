"""Zero-update replay of B10k and the frozen r3 road repair, including a37.

Import render_baseline(e, raw, include_target, repair_enabled=True) to reuse the
exact static-gate -> dynamic -> road ordering in the C2 implementation.
No existing run files are written.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

R3 = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_only_B10k_road_20261103_r3')
RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')


def render_baseline(e, raw, include_target, repair_enabled=True,
                    apply_gates=None, include_road=None, tag_components=False):
    """One actual-depth raster; include_target=True retains dynamic a37.

    The road repair is active whenever repair_enabled is True; this does not
    route through Native.render or the disabled-repair bypass.
    """
    import torch
    from editor_common import prepare_rasterization, road_kwargs, merge_rows, raster
    camera = e.camera(raw)
    n = e.n
    apply_gates = repair_enabled if apply_gates is None else apply_gates
    include_road = repair_enabled if include_road is None else include_road
    kw = prepare_rasterization(camera, n.static, n.args)
    kw['means2D'] = torch.zeros_like(kw['means3D'])
    def tag(values, component):
        extra = torch.zeros((len(values['features']), 3), device=values['features'].device,
                            dtype=values['features'].dtype)
        extra[:, component] = 1
        values['features'] = torch.cat((values['features'], extra), 1)
    if apply_gates:
        ids = e.repair.ids
        kw['opacities'] = kw['opacities'].index_copy(
            0, ids, kw['opacities'][ids] * e.repair.gate_raw.sigmoid()[:, None])
    if tag_components:
        tag(kw, 0)
    if include_target:
        dynamic = prepare_rasterization(camera, n.dynamic, n.args, dynamic=True)
        dynamic['means2D'] = torch.zeros_like(dynamic['means3D'])
        if tag_components:
            tag(dynamic, 1)
        kw = merge_rows(kw, dynamic)
    if include_road:
        channels = kw['shs'].shape[1] if kw['shs'] is not None else 1
        road = road_kwargs(camera, e.repair.road, n.args, channels,
                           precomputed_color=kw['colors_precomp'] is not None)
        if tag_components:
            tag(road, 2)
        else:
            kw['features'] = torch.cat((kw['features'], torch.zeros_like(kw['features'][:, :1])), 1)
            road['features'] = torch.cat((road['features'], torch.ones_like(road['features'][:, :1])), 1)
        kw = merge_rows(kw, road)
    out = raster(camera, n.args, n.static.active_sh_degree, kw, n.bg)
    env = n.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
    out['rgb'] = out['rgb'] + (1 - out['alpha']) * env
    out['road_visible_alpha'] = out['extra'][2:3] if tag_components else out['extra'] if include_road else torch.zeros_like(out['alpha'])
    if tag_components:
        out['component_alpha_T'] = out['extra']
    return out


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def mechanism_audit(output):
    """Zero-update factorization; never controls the product's no-edit route."""
    started = time.monotonic()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(R3 / 'source'))
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion
    import torch
    import repair37
    from utils.loss_utils import psnr, ssim
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    e = repair37.Experiment()
    checkpoint = torch.load(str(R3 / 'formal/checkpoint001000.pt'), map_location='cpu', weights_only=False)
    assert checkpoint['parent_identity'] == e.n.before and checkpoint['step'] == 1000
    e.repair.load_state_dict(checkpoint['repair'], strict=True)
    e.repair.requires_grad_(False)
    del checkpoint
    gc.collect()
    rows = []
    modes = dict(native=(False, False), gates_only=(True, False), road_only=(False, True), gates_road=(True, True))
    source = Path(e.n.config['source_path']) / 'image/image_0'
    def rgb8(t):
        return np.uint8(np.floor(np.clip(t.detach().cpu().numpy().transpose(1, 2, 0), 0, 1) * 255 + .5))
    with torch.no_grad():
        for raw in (105, 113, 118):
            folder = output / f'{raw:03d}'
            folder.mkdir()
            gt_np = cv2.resize(np.asarray(Image.open(source / f'0008{raw:03d}.png').convert('RGB')) / 255., (480, 320)).astype(np.float32)
            gt = torch.as_tensor(gt_np.transpose(2, 0, 1).copy(), device='cuda')
            d = e.view(raw)
            labels = d['actor_labels']
            target = labels == 5
            masks = dict(all=np.ones((320, 480), bool), target_a37=target,
                target_inner=binary_erosion(target, iterations=2),
                target_edge=binary_dilation(target, iterations=2) & ~binary_erosion(target, iterations=2),
                outside_target=~target, retained_a6=labels == 4,
                road_repair_without_target=d['repair_mask'] & ~target,
                road_boundary=binary_dilation(d['repair_mask'], iterations=2) & ~binary_erosion(d['repair_mask'], iterations=2) & ~d['other_vehicle_guard'])
            packages = {name: render_baseline(e, raw, True, apply_gates=gates, include_road=road, tag_components=True)
                        for name, (gates, road) in modes.items()}
            plain_native = e.n.render(e.camera(raw), dynamic=True)['rgb']
            tag_native_max = float((plain_native - packages['native']['rgb']).abs().max())
            assert tag_native_max <= 1e-6, tag_native_max
            plain_full = render_baseline(e, raw, True)['rgb']
            tag_full_max = float((plain_full - packages['gates_road']['rgb']).abs().max())
            assert tag_full_max <= 1e-6, tag_full_max
            native = packages['native']['rgb'].clamp(0, 1)
            native_u8 = rgb8(native)
            record = dict(raw=raw, tag_native_maxabs=tag_native_max, tag_full_maxabs=tag_full_max, methods={})
            for name, package in packages.items():
                image = package['rgb'].clamp(0, 1)
                alpha = package['component_alpha_T']
                alpha_sum_max = float((alpha.sum(0, keepdim=True) - package['alpha']).abs().max())
                assert alpha_sum_max < 2e-5, alpha_sum_max
                image_u8 = rgb8(image)
                method = dict(psnr=float(psnr(image, gt)), ssim=float(ssim(image[None], gt[None])),
                              component_sum_vs_alpha_maxabs=alpha_sum_max, masks={})
                for mask_name, mask_np in masks.items():
                    mask = torch.as_tensor(mask_np, device='cuda')
                    n = int(mask.sum())
                    dy_delta = alpha[1] - packages['native']['component_alpha_T'][1]
                    diff = (image - native).abs().mean(0)
                    gt_error = (image - gt).abs().mean(0)
                    u8diff = np.abs(image_u8.astype(np.int16) - native_u8.astype(np.int16)).max(2)
                    method['masks'][mask_name] = dict(samples=n,
                        gt_srgb_MAE=float(gt_error[mask].mean()) if n else None,
                        native_srgb_change=float(diff[mask].mean()) if n else None,
                        native_pixels_changed_over1=int((u8diff[mask_np] > 1).sum()),
                        component_alpha_T={component: dict(mean=float(alpha[i, mask].mean()) if n else None,
                            max=float(alpha[i, mask].max()) if n else None) for i, component in enumerate(('static', 'target_dynamic', 'road'))},
                        dynamic_alpha_T_delta_mean=float(dy_delta[mask].mean()) if n else None,
                        dynamic_alpha_T_delta_min=float(dy_delta[mask].min()) if n else None,
                        dynamic_alpha_T_delta_max=float(dy_delta[mask].max()) if n else None,
                        dynamic_contribution_decreased_over01=int(((dy_delta < -.01) & mask).sum()),
                        road_alpha_T_over01=int(((alpha[2] > .01) & mask).sum()),
                        gt_error_sum=float(gt_error[mask].sum()) if n else 0)
                record['methods'][name] = method
                Image.fromarray(image_u8).save(folder / f'{name}_full.png')
                for i, component in enumerate(('static', 'target_dynamic', 'road')):
                    Image.fromarray(rgb8(alpha[i:i+1].expand(3, -1, -1))).save(folder / f'{name}_{component}_alpha_T.png')
                Image.fromarray(rgb8((image - native).abs().mean(0, keepdim=True).expand(3, -1, -1) * 5)).save(folder / f'{name}_native_error_x5.png')
            Image.fromarray(rgb8(gt)).save(folder / 'gt_opencv.png')
            Image.fromarray(np.uint8(target) * 255).save(folder / 'target_mask.png')
            Image.fromarray(np.concatenate([rgb8(packages[name]['rgb']) for name in modes], axis=1)).save(output / f'{raw:03d}_factorization.png')
            rows.append(record)
            write(output / 'progress.json', dict(raw=raw, optimizer_updates=0, seconds=time.monotonic() - started))
            print(json.dumps(dict(stage='noedit_mechanism', raw=raw,
                methods={name: {k: method[k] for k in ('psnr', 'ssim')} for name, method in record['methods'].items()})), flush=True)
    result = dict(optimizer_updates=0, frames=[105, 113, 118], target_dynamic_in_all_modes=True,
        purpose='Read-only causal factorization; no product bypass proposed',
        common_current_camera_and_timestamp=True, original_r3_inputs_unchanged=True,
        component_alpha_T='Feature-tagged within the same native depth-sorted raster; sum verified against total alpha',
        parent_unchanged=e.unchanged(), seconds=time.monotonic() - started, rows=rows)
    assert result['parent_unchanged']
    write(output / 'metrics.json', result)
    print(json.dumps(dict(stage='MECHANISM_COMPLETE', seconds=result['seconds'], optimizer_updates=0)), flush=True)
    del e
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUN / 'evidence/baseline_replay')
    parser.add_argument('--no-lpips', action='store_true')
    parser.add_argument('--mechanism', action='store_true')
    args = parser.parse_args()
    if args.mechanism:
        destination = RUN / 'evidence/noedit_mechanism' if args.output == RUN / 'evidence/baseline_replay' else args.output
        mechanism_audit(destination)
        return
    started = time.monotonic()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(R3 / 'source'))
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion
    import torch
    import repair37
    from utils.loss_utils import psnr, ssim

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / 'frames').mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    e = repair37.Experiment()
    checkpoint_path = R3 / 'formal/checkpoint001000.pt'
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 1000
    assert checkpoint['parent_identity'] == e.n.before
    e.repair.load_state_dict(checkpoint['repair'], strict=True)
    assert all(torch.equal(v.detach().cpu(), checkpoint['repair'][k])
               for k, v in e.repair.state_dict().items())
    e.repair.requires_grad_(False)
    e.repair.eval()
    del checkpoint
    gc.collect()
    lpips_model = None
    lpips_error = None
    if not args.no_lpips:
        try:
            from lpipsPyTorch.modules.lpips import LPIPS
            lpips_model = LPIPS('vgg', '0.1').eval().cuda()
            lpips_model.requires_grad_(False)
        except Exception as exc:
            lpips_error = repr(exc)
            print(json.dumps({'stage': 'lpips_unavailable', 'error': lpips_error}), flush=True)
    contract = dict(
        optimizer_updates=0, baseline='single-a37 B10k plus frozen r3 road1k',
        road_checkpoint=str(checkpoint_path), road_inputs=str(R3 / 'inputs'),
        road_inputs_and_masks_unchanged=True, frame_range=[89, 137], frame_count=49,
        resolution=[480, 320], camera='image_0',
        ground_truth='Original uint8 RGB -> float64 /255 -> cv2.resize INTER_LINEAR -> float32, matching native loadCam',
        full_render='static opacity gate -> retained target dynamic -> fixed road, common raster and environment',
        noedit_repair_enabled=True, target_dynamic_in_repaired_full=True,
        full_metrics='clamped float sRGB, per-frame PSNR/SSIM/LPIPS-vgg then arithmetic mean',
        local_metrics='pixel-weighted sRGB MAE plus maxima; 8bit difference is channel-rounded comparison',
        proxy_reference='Unchanged r3 linear recovery reference; original_shadow uses unchanged older reference_linear',
        input_role='49 observed train / all199 BUILD, no independent clean-road ground truth',
        lpips_enabled=lpips_model is not None, lpips_error=lpips_error,
        parent_identity=e.n.before, torch_version=torch.__version__, cv2_version=cv2.__version__)
    write(outdir / 'contract.json', contract)
    source_path = Path(e.n.config['source_path']) / 'image/image_0'
    methods = ('native_full', 'repaired_full', 'repaired_delete37')
    rows = []
    replay_max = 0
    native_old_max = 0
    helper_legacy_max = 0.0

    def rgb8(t):
        return np.uint8(np.floor(np.clip(t.detach().cpu().numpy().transpose(1, 2, 0), 0, 1) * 255 + .5))

    @torch.no_grad()
    def image_metric(pred, truth):
        p = pred.clamp(0, 1)
        metric = dict(psnr=float(psnr(p, truth)), ssim=float(ssim(p[None], truth[None])))
        metric['lpips_vgg'] = float(lpips_model(p[None], truth[None]).mean()) if lpips_model is not None else None
        return metric

    with torch.no_grad(), (outdir / 'metrics.jsonl').open('x') as log:
        for raw in range(89, 138):
            frame = outdir / 'frames' / f'{raw:03d}'
            frame.mkdir()
            source = np.asarray(Image.open(source_path / f'0008{raw:03d}.png').convert('RGB')) / 255.0
            truth_np = cv2.resize(source, (480, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            truth = torch.as_tensor(truth_np.transpose(2, 0, 1).copy(), device='cuda')
            d = e.view(raw)
            labels = d['actor_labels']
            masks = {
                'target_a37': labels == 5,
                'retained_a6': labels == 4,
                'retained_all': (labels >= 1) & (labels <= 4),
                'other_vehicle_guard': d['other_vehicle_guard'],
                'road_boundary_outer': binary_dilation(d['repair_mask'], iterations=2) & ~d['repair_mask'] & d['valid'] & (labels == 0) & ~d['other_vehicle_guard'],
                'road_boundary_band': binary_dilation(d['repair_mask'], iterations=2) & ~binary_erosion(d['repair_mask'], iterations=2) & ~d['other_vehicle_guard'],
                'outside_repair': d['protect_mask'],
                'marking': d['marking_mask'],
                'left_road': d['valid'] & d['domain_a5'] & ~d['domain_a37'] & (labels == 0),
            }
            for label, alias in enumerate((5, 26, 18, 6), 1):
                masks[f'retained_a{alias}'] = labels == label
            packages = {
                'native_full': render_baseline(e, raw, True, repair_enabled=False),
                'repaired_full': render_baseline(e, raw, True, repair_enabled=True),
                'repaired_delete37': render_baseline(e, raw, False, repair_enabled=True),
            }
            if raw in (89, 101, 105, 113, 137):
                helper_legacy_max = max(helper_legacy_max, float((packages['repaired_delete37']['rgb'] - e.render(raw)['rgb']).abs().max()))
                assert helper_legacy_max <= 1e-6
            arrays = {name: rgb8(pkg['rgb']) for name, pkg in packages.items()}
            reference = np.asarray(Image.open(R3 / 'evaluation/all49' / f'{raw:03d}_repaired.png').convert('RGB'))
            replay = int(np.abs(arrays['repaired_delete37'].astype(np.int16) - reference.astype(np.int16)).max())
            replay_max = max(replay_max, replay)
            assert replay <= 1, (raw, replay)
            old_full = np.asarray(Image.open(R3.parent / 'scene008_a37_only_B10k_20261103_r1/evaluation' / f'{raw:03d}/composite.png').convert('RGB'))
            old_full_difference = int(np.abs(arrays['native_full'].astype(np.int16) - old_full.astype(np.int16)).max())
            native_old_max = max(native_old_max, old_full_difference)
            assert old_full_difference <= 1, (raw, old_full_difference)
            record = dict(raw=raw, methods={}, masks={}, replay_delete_max_uint8=replay,
                          replay_native_full_max_uint8=old_full_difference)
            native = packages['native_full']['rgb'].clamp(0, 1)
            for name in methods:
                record['methods'][name] = image_metric(packages[name]['rgb'], truth)
                Image.fromarray(arrays[name]).save(frame / f'{name}.png')
            Image.fromarray(rgb8(truth)).save(frame / 'gt_opencv.png')
            for name, mask in masks.items():
                count = int(mask.sum())
                values = dict(samples=count, methods={})
                m = torch.as_tensor(mask, device='cuda')
                for method in methods:
                    pred = packages[method]['rgb'].clamp(0, 1)
                    error = (pred - truth).abs()[:, m]
                    change = (pred - native).abs()[:, m]
                    u8error = np.abs(arrays[method].astype(np.int16) - arrays['native_full'].astype(np.int16))[mask]
                    values['methods'][method] = dict(
                        gt_srgb_MAE=float(error.mean()) if count else None,
                        gt_srgb_maxabs=float(error.max()) if count else None,
                        native_full_srgb_change=float(change.mean()) if count else None,
                        native_full_srgb_maxabs=float(change.max()) if count else None,
                        native_full_change_MAE_8bit=float(u8error.mean()) if count else None,
                        native_full_change_max_8bit=int(u8error.max()) if count else None,
                        native_full_pixels_changed_over1=int((u8error.max(1) > 1).sum()) if count else 0)
                record['masks'][name] = values
            proxy_masks = {
                'fit': d['repair_mask'] & ~d['inferred_fine'],
                'new_dark': d['new_shadow_proxy'] & d['repair_mask'] & ~d['inferred_fine'],
                'original_shadow': d['shadow_proxy'] & d['domain_a37'],
                'inferred': d['inferred_fine'],
            }
            record['delete_proxy'] = {}
            linear = repair37.r4.linear(packages['repaired_delete37']['rgb'])
            for name, mask in proxy_masks.items():
                target_key = 'reference_linear' if name == 'original_shadow' else 'recovery_reference_linear'
                target = torch.as_tensor(d[target_key].transpose(2, 0, 1), device='cuda')
                count = int(mask.sum())
                m = torch.as_tensor(mask, device='cuda')
                record['delete_proxy'][name] = dict(samples=count, linear_MAE=float((linear - target).abs()[:, m].mean()) if count else None)
            if raw in (101, 105, 113):
                Image.fromarray(np.uint8(masks['road_boundary_outer']) * 255).save(frame / 'road_boundary_outer.png')
                Image.fromarray(np.uint8(masks['retained_a6']) * 255).save(frame / 'retained_a6.png')
                board = np.concatenate([arrays[k] for k in methods], axis=1)
                Image.fromarray(board).save(outdir / f'{raw:03d}_comparison.png')
            rows.append(record)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            write(outdir / 'progress.json', dict(raw=raw, frames=len(rows), optimizer_updates=0,
                  seconds=time.monotonic() - started, replay_delete_max_uint8=replay_max))
            if raw == 89 or (raw - 88) % 5 == 0 or raw == 137:
                print(json.dumps(dict(stage='baseline_replay', raw=raw, frames=len(rows),
                      seconds=time.monotonic() - started, methods=record['methods'],
                      replay_delete_max_uint8=replay_max)), flush=True)
            del packages, arrays
    summary = dict(contract=contract, frame_count=len(rows), optimizer_updates=0,
                   methods={}, masks={}, delete_proxy={}, replay_delete_max_uint8=replay_max,
                   replay_native_full_max_uint8=native_old_max, helper_legacy_maxabs=helper_legacy_max,
                   parent_unchanged=e.unchanged(), seconds=time.monotonic() - started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    for method in methods:
        summary['methods'][method] = {
            key: float(np.mean([row['methods'][method][key] for row in rows]))
            if rows[0]['methods'][method][key] is not None else None
            for key in ('psnr', 'ssim', 'lpips_vgg')}
    for mask in rows[0]['masks']:
        total = sum(row['masks'][mask]['samples'] for row in rows)
        group = dict(samples=total, methods={})
        for method in methods:
            group['methods'][method] = {}
            for key in rows[0]['masks'][mask]['methods'][method]:
                vals = [(row['masks'][mask]['samples'], row['masks'][mask]['methods'][method][key]) for row in rows if row['masks'][mask]['samples']]
                if key.endswith('pixels_changed_over1'):
                    value = sum(v for n, v in vals)
                elif 'max' in key:
                    value = max(v for n, v in vals) if vals else None
                else:
                    value = sum(n * v for n, v in vals) / total if total else None
                group['methods'][method][key] = value
        summary['masks'][mask] = group
    for name in rows[0]['delete_proxy']:
        total = sum(row['delete_proxy'][name]['samples'] for row in rows)
        summary['delete_proxy'][name] = dict(samples=total, linear_MAE=sum(row['delete_proxy'][name]['samples'] * (row['delete_proxy'][name]['linear_MAE'] or 0) for row in rows) / max(total, 1))
    assert summary['parent_unchanged'] and len(rows) == 49
    write(outdir / 'metrics.json', dict(summary=summary, rows=rows))
    write(outdir / 'summary.json', summary)
    print(json.dumps(dict(stage='COMPLETE', summary=summary), allow_nan=False), flush=True)
    del lpips_model, e
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
