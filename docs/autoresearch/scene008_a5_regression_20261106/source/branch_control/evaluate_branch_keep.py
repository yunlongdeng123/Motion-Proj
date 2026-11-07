"""Native KEEP-only diagnosis. Pilot renders 97/105/109; full49 is explicit.

The complete a5+a37 dynamic pool is retained. No DELETE result is produced.
Common camera/observation validation and checkpoint restoration are reused verbatim.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
OBSERVATIONS = ENGINE.parents[2] / 'scene008_instances_20261105_r1/observations/frames'
PILOT = (97, 105, 109)


def load_common():
    path = Path(__file__).with_name('native_evaluation_common.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'be18a16926f8ebebe345dd7e1ae5cf6ce80378efb2fa6c43ed7ab3271ecaf1dc'
    spec = importlib.util.spec_from_file_location('branch_native_evaluation_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(keep, gt, alpha):
    n = len(keep)
    error = np.abs(np.clip(keep, 0, 1).astype(np.float64) - gt.astype(np.float64)).mean(axis=1)
    raw_error = np.abs(keep.astype(np.float64) - gt.astype(np.float64)).mean(axis=1)
    return dict(pixels=n, rgb_error_sum=float(error.sum()), raw_rgb_error_sum=float(raw_error.sum()),
                alpha_sum=float(alpha.astype(np.float64).sum()),
                keep_rgb_mae=float(error.mean()) if n else None,
                keep_mean_alpha=float(alpha.mean()) if n else None,
                coverage_counts={str(t): int((alpha >= t).sum()) for t in [.1, .5, .9, .99]})


def aggregate(rows):
    n = sum(row['pixels'] for row in rows)
    return dict(pixels=n, weighting='fixed selected pixel-frame count, not unweighted frame mean',
                keep_rgb_mae=sum(row['rgb_error_sum'] for row in rows) / n,
                keep_raw_rgb_mae=sum(row['raw_rgb_error_sum'] for row in rows) / n,
                keep_mean_alpha=sum(row['alpha_sum'] for row in rows) / n,
                keep_coverage={str(t): sum(row['coverage_counts'][str(t)] for row in rows) / n for t in [.1, .5, .9, .99]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--iteration', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--scope', choices=['pilot', 'all49'], default='pilot')
    ap.add_argument('--pilot-reference', type=Path)
    opts = ap.parse_args()
    assert ROOT.name == 'scene008_a5_branch_control_20261107_r1'
    common = load_common()
    opts.model, opts.output = opts.model.resolve(), opts.output.resolve()
    opts.engine, opts.observations = ENGINE, OBSERVATIONS
    opts.expected_mask_pixels, opts.crop, opts.dynamic_aliases = 121128, [0, 160, 175, 320], [5, 37]
    assert opts.iteration > 0 and opts.model.is_relative_to(ROOT)
    assert opts.scope != 'all49' or opts.pilot_reference is not None, 'Full49 requires an already completed matching pilot'
    assert not opts.output.exists() and opts.output.is_relative_to(ROOT) and not opts.output.is_relative_to(opts.model)
    opts.output.mkdir()
    paths = [Path(__file__), Path(__file__).with_name('native_evaluation_common.py')]
    paths += common.source_model_files(opts.model, opts.iteration)
    paths += sorted(ENGINE.rglob('*.py'))
    paths += [OBSERVATIONS / f'{raw:03d}.npz' for raw in range(89, 138)]
    if opts.pilot_reference:
        assert opts.scope == 'all49'
        paths += [opts.pilot_reference / f'frames/{raw:03d}/keep.png' for raw in PILOT]
        pilot = common.read_json(opts.pilot_reference / 'summary.json')
        assert pilot['status'] == 'PASS_NATIVE_KEEP_ONLY_PILOT' and pilot['iteration'] == opts.iteration
        assert Path(pilot['model']).resolve() == opts.model
    before = common.inventory(paths)
    common.write_json(opts.output / 'source_manifest_before.json', dict(files=before))
    start = time.monotonic()
    error = None
    summary = None
    try:
        frames, full_sha, crop = common.observation_headers(opts, common.read_json(opts.model / 'cameras.json'),
                     common.read_json(opts.model / 'actual_scene_binding.json'))
        selected = frames if opts.scope == 'all49' else [f for f in frames if f['raw'] in PILOT]
        assert len(selected) == (49 if opts.scope == 'all49' else 3)
        sys.path.insert(0, str(ENGINE))
        import torch
        from scene.cameras import Camera
        import gaussian_renderer
        from scene import gaussian_model
        assert Path(gaussian_renderer.__file__).resolve() == ENGINE / 'gaussian_renderer/__init__.py'
        assert Path(gaussian_model.__file__).resolve() == ENGINE / 'scene/gaussian_model.py'
        assert torch.cuda.is_available()
        torch.set_num_threads(1)
        args, static, dynamic, env = common.restore_models(opts, common.read_json(opts.model / 'effective_config.json'))
        bg = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
        (opts.output / 'a5_float').mkdir()
        (opts.output / 'anchors').mkdir()
        rows = []
        replay = []
        for frame in selected:
            raw, mask, K, W = [frame[k] for k in ['raw', 'mask', 'K', 'W']]
            camera = Camera(raw, W[:3, :3].T.copy(), W[:3, 3].copy(), cx=K[0, 2], cy=K[1, 2],
                            fx=K[0, 0], fy=K[1, 1], image=torch.zeros((3, 320, 480)),
                            image_name=f'0008{raw:03d}', resolution=(480, 320), timestamp=frame['timestamp'])
            assert np.allclose(camera.world_view_transform.detach().cpu().numpy().T, W, atol=1e-5, rtol=0)
            assert abs(float(camera.timestamp) - frame['timestamp']) <= 2e-7
            with torch.no_grad():
                package = gaussian_renderer.render(camera, static, dynamic, args, bg, env_map=env, is_training=False)
            keep, alpha = common.tensor_hwc(package['render']), common.tensor_alpha(package['alpha'])
            assert np.isfinite(keep).all() and np.isfinite(alpha).all()
            with np.load(frame['path'], allow_pickle=False) as data:
                gt = np.ascontiguousarray(data['rgb'], dtype=np.float32)
                assert np.array_equal(data['actor_labels'] == 1, mask)
            images = dict(gt=common.rgb8(gt), keep=common.rgb8(keep))
            directory = opts.output / f'frames/{raw:03d}'
            directory.mkdir(parents=True)
            for name, array in images.items():
                Image.fromarray(array).save(directory / f'{name}.png')
            if raw in PILOT:
                if opts.pilot_reference:
                    reference = np.asarray(Image.open(opts.pilot_reference / f'frames/{raw:03d}/keep.png').convert('RGB'))
                    mode = 'independent_pilot_to_full49_native_keep_replay'
                else:
                    with torch.no_grad():
                        repeat = gaussian_renderer.render(camera, static, dynamic, args, bg, env_map=env, is_training=False)
                    reference = common.rgb8(common.tensor_hwc(repeat['render']))
                    del repeat
                    mode = 'same_checkpoint_native_forward_repeatability_only'
                maximum = int(np.abs(images['keep'].astype(np.int16) - reference.astype(np.int16)).max())
                assert maximum <= 1
                replay.append(dict(raw=raw, max_uint8_difference=maximum, mode=mode))
            np.savez_compressed(opts.output / f'a5_float/{raw:03d}.npz', raw=np.int16(raw),
                yx=np.column_stack(np.nonzero(mask)).astype(np.int16), keep_rgb=keep[mask], gt_rgb=gt[mask],
                keep_alpha=alpha[mask], mask_sha256=np.asarray(frame['mask_sha256']), model_time=np.float64(frame['timestamp']))
            values = measure(keep[mask], gt[mask], alpha[mask])
            rows.append(dict(raw=raw, fixed_mask_sha256=frame['mask_sha256'], **values))
            if raw in (89, 97, 105, 109):
                common.contact_sheet(opts.output / f'anchors/{raw:03d}_crop.png', [images['gt'], images['keep']],
                                     [f'raw{raw} GT', 'KEEP a5+a37'], crop)
            print(json.dumps(dict(raw=raw, pixels=values['pixels'], keep_rgb_mae=values['keep_rgb_mae'])), flush=True)
            del package, camera
        summary = dict(status='PASS_NATIVE_KEEP_ONLY_PILOT' if opts.scope == 'pilot' else 'PASS_NATIVE49_KEEP_ONLY_EVALUATION',
            scope=opts.scope, frame_count=len(rows), raw_ids=[f['raw'] for f in selected], frames=rows,
            iteration=opts.iteration, model=str(opts.model), dynamic_aliases=[5,37], keep_only=True,
            delete_produced=False, native_delete_removes=[5,37], a37_only_delete_solution=False,
            optimizer_created=False, optimization_updates=0, native_static_rows=len(static.get_xyz),
            native_dynamic_rows=len(dynamic.get_xyz), fixed_denominator=dict(pixels=sum(f['pixels'] for f in selected),
                rule='selected observations actor_labels==1; no rendered-alpha selection'),
            full49_fixed_reference=dict(pixels=121128, sha256=full_sha),
            selected=aggregate(rows), replay=replay, camera_headers_checked=49,
            rendered_frames=len(rows), engine=str(ENGINE), source_integrity='PENDING',
            quality_accepted=False, human_verdict=None,
            caution='Branch targets/priors/representation bundle; KEEP-only diagnosis, not a37-only deletion preservation.')
        if opts.scope == 'all49':
            summary['all49'] = summary['selected']
        common.write_json(opts.output / 'summary.json', summary)
    except BaseException:
        import traceback
        error = traceback.format_exc()
    finally:
        after = common.inventory(paths)
        changed = [path for path in before if before[path] != after.get(path)]
        common.write_json(opts.output / 'source_integrity.json', dict(status='PASS_UNCHANGED' if not changed else 'FAIL',
            checked_files=len(before), changed=changed, before=before, after=after, optimization_updates=0))
    if error or changed:
        common.write_json(opts.output / 'failure.json', dict(traceback=error, changed=changed))
        raise RuntimeError(error or str(changed))
    summary['source_integrity'] = 'PASS_UNCHANGED_BYTES_MTIME_SHA256'
    summary['seconds'] = time.monotonic() - start
    common.write_json(opts.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
