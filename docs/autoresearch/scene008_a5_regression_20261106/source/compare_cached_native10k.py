"""CPU-only comparison of retained native10K PNGs; no checkpoint reload."""
from pathlib import Path
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = CAMPAIGN / 'scene008_a5_single_bug_20261106_r1'
OBS = CAMPAIGN / 'scene008_instances_20261105_r1/observations/frames'
HELPER = CAMPAIGN / 'scene008_a5_old_single30k_20261106_r1/source/evaluate_a5.py'
MODELS = {
    'full5_native10k': CAMPAIGN / 'r15_gn11_stage1_v41_20261023/arms/B/run30k',
    'single_native10k': CAMPAIGN / 'scene008_a37_only_B10k_20261103_r1/formal10k',
}
def main():
    out = ROOT / 'cached_native10k_comparison'
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('native_a5_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    options = SimpleNamespace(observations=OBS, expected_mask_pixels=121128, crop=[0, 160, 175, 320])
    parsed, files = {}, [Path(__file__), HELPER]
    for name, model in MODELS.items():
        event, references, status = helper.event_references(model, 10000)
        assert status == 'EVENT_PRESENT' and len(references) == 49
        frames, denominator, crop = helper.observation_headers(options,
            helper.read_json(model / 'cameras.json'), helper.read_json(model / 'actual_scene_binding.json'))
        parsed[name] = (model, frames, references, denominator)
        files += [event, model / 'cameras.json', model / 'actual_scene_binding.json', model / 'effective_config.json']
        files += [refs['keep'] for refs in references.values()]
    files += [OBS / f'{raw:03d}.npz' for raw in range(89, 138)]
    before = helper.inventory(files)
    summaries, pictures = {}, {}
    for name, (model, frames, references, denominator) in parsed.items():
        rows = []
        for frame in frames:
            raw, mask = frame['raw'], frame['mask']
            with np.load(frame['path'], allow_pickle=False) as z:
                gt = z['rgb'].astype(np.float64)
            with Image.open(references[raw]['keep']) as im:
                rgb8 = np.asarray(im.convert('RGB'))
            assert rgb8.shape == gt.shape == (320, 480, 3)
            rgb = rgb8.astype(np.float64) / 255.
            per_pixel = np.abs(rgb - gt).mean(2)
            edge = helper.border_inside(mask)
            rows.append(dict(raw=raw, pixels=int(mask.sum()), boundary_pixels=int(edge.sum()),
                error_sum=float(per_pixel[mask].sum()), boundary_error_sum=float(per_pixel[edge].sum()),
                keep_rgb_mae=float(per_pixel[mask].mean()) if mask.any() else None,
                source_png=str(references[raw]['keep'])))
            if raw in (89, 97, 105, 109):
                dest = out / name / 'frames' / f'{raw:03d}'
                dest.mkdir(parents=True)
                Image.fromarray(rgb8).save(dest / 'keep.png')
                pictures[(name, raw)] = rgb8
                pictures[('gt', raw)] = np.uint8(np.clip(gt, 0, 1) * 255 + .5)
        pixels = sum(r['pixels'] for r in rows)
        border = sum(r['boundary_pixels'] for r in rows)
        assert pixels == 121128
        summaries[name] = dict(model=str(model), native_event_iteration=10000, frame_count=49, frames=rows,
            fixed_denominator_sha256=denominator, pixels=pixels,
            keep_rgb_mae=sum(r['error_sum'] for r in rows) / pixels,
            boundary_pixels=border, boundary_keep_rgb_mae=sum(r['boundary_error_sum'] for r in rows) / border,
            cached_PNG_sha_bound=True, native_checkpoint_reloaded=False,
            checkpoint_components_still_available=all((model / f'{p}10000.pth').exists() for p in ('chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state')))
    assert summaries['full5_native10k']['fixed_denominator_sha256'] == summaries['single_native10k']['fixed_denominator_sha256']
    (out / 'anchors').mkdir()
    for raw in (89, 97, 105, 109):
        helper.contact_sheet(out / 'anchors' / f'{raw:03d}_crop.png',
            [pictures[(name, raw)] for name in ('gt', 'full5_native10k', 'single_native10k')],
            [f'raw{raw} observed', 'Full5 cached native10K', 'Single cached native10K'], [0, 160, 175, 320])
    after = helper.inventory(files)
    assert before == after
    assert 'torch' not in sys.modules
    result = dict(status='PASS_CACHED_NATIVE10K_PNG_SAME_BUDGET_COMPARISON', GPU_used=False,
        torch_imported=False, optimization_updates=0, native_GPU_renders=0,
        input='Existing original native composite uint8 PNG files referenced by iteration_010000 events; not reconstructed weights.',
        metric='PNG/255 compared against the same original float observed RGB on actor_labels==1; fixed pixel-frame weighted channel MAE',
        fixed_denominator=dict(pixels=121128, sha256=summaries['full5_native10k']['fixed_denominator_sha256']),
        models=summaries,
        delta_single_minus_full5_keep_rgb_mae=summaries['single_native10k']['keep_rgb_mae']-summaries['full5_native10k']['keep_rgb_mae'],
        limits=['PNG quantization differs from fresh floating-point evaluation; this table stays separate.',
                'Full5 native10K weights are absent, so no reload/GPU replay is claimed.',
                'Equal10K budget alone does not isolate changed input support, branch losses, or RNG trajectories.'],
        source_integrity='PASS_UNCHANGED_BYTES_MTIME_SHA256')
    helper.write_json(out / 'summary.json', result)
    helper.write_json(out / 'source_integrity.json', dict(status='PASS_UNCHANGED', before=before, after=after))
    print(json.dumps({name: {k:v for k,v in row.items() if k not in ('frames',)} for name,row in summaries.items()}))
if __name__ == '__main__':
    main()
