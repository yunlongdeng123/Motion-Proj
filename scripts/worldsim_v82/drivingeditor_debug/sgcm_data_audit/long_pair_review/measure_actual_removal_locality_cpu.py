"""Predefined removal-locality supplement from saved NPZ; actual V required.

No Torch/model/renderer/GPU imports. Reads three completed runs and original masks,
following the pinned loader's sky-linear/actor-nearest decode recipe. It verifies
all six region counts against saved evals; this is not a runtime mask snapshot.
Writes only this reviewer directory and never edits canonical primary metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V = ROOT / 'continuation_diagnostic/opacity_outside30k_to32000'
ARMS = {'B1': V.parent / 'original30k_to32000', 'B2': V.parent / 'original30k_to32000_noise_reference', 'V': V}
PROTOCOL_SHA = 'af9726931762fde14c00684ed1004ce2a6464a46299e832f969cdce3c916a75d'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed', action='store_true')
    args = parser.parse_args()
    assert sha(HERE / 'interpretation_preregistered.json') == PROTOCOL_SHA
    if not args.completed:
        print(json.dumps({'status': 'CPU_LOCALITY_RECIPE_PREPARED_WAIT_REAL_V_COMPLETION', 'V_endpoint_result_or_placeholder_written': False, 'GPU_used': False}))
        return
    status = read(V / 'status.json')
    assert status['status'] == 'complete' and status['actual_exit_code'] == 0 and status['gpu_lock_held'] is False
    plan = read(V / 'diagnostic_plan.json')
    source = Path(plan['training_config']['source_path'])
    masks_source = Path(plan['repository']) / 'utils/paper_masks.py'
    assert sha(masks_source) == plan['runtime_source_sha256'][str(masks_source)]
    common = read(V / 'receipts/common49_inputs_after_V_cpu.json')
    assert common['actual467_input_SHA_match_original'] is True and common['new_shadow_mask_absent'] is True
    assert not (source / 'dynamic_masks/shadow_mask_0').exists()
    finals = {arm: read(root / 'extra_eval/32000/diagnostic_eval.json') for arm, root in ARMS.items()}
    tables = {arm: {f['frame_id']: f for f in value['frames']} for arm, value in finals.items()}
    summaries = {arm: read(root / 'receipts/full49_summary.json') for arm, root in ARMS.items()}
    rows = []
    mask_receipts = []
    for raw in range(49):
        name = f'0017{raw:03}'
        frame_id = 'image_0/' + name
        actor_relative = 'dynamic_masks/dynamic_mask_0/' + name + '.png'
        sky_relative = 'sky/sky_0/' + name + '.png'
        for relative in (actor_relative, sky_relative):
            assert sha(source / relative) == common['actual_input_SHA'][relative]
        with Image.open(source / actor_relative) as image:
            actor = np.asarray(image.convert('L').resize((480, 320), Image.Resampling.NEAREST)) > 0
        with Image.open(source / sky_relative) as image:
            sky = cv2.resize((np.asarray(image) > 0).astype(np.float32), (480, 320)) > 0
        masks = {'full': np.ones((320, 480), dtype=bool), 'sky': sky, 'non_sky': ~sky, 'actor': actor, 'outside_actor': ~actor, 'non_sky_outside_actor': ~sky & ~actor}
        counts = {key: int(value.sum()) for key, value in masks.items()}
        assert all(counts == {key: r['pixel_count'] for key, r in tables[arm][frame_id]['regions'].items()} for arm in ARMS)
        mask_receipts.append({'frame_id': frame_id, 'actual_evaluation_region_counts_exact': counts, 'reconstructed_sky_bool_SHA': hashlib.sha256(sky.tobytes()).hexdigest(), 'reconstructed_actor_bool_SHA': hashlib.sha256(actor.tobytes()).hexdigest()})
        row = {'frame_id': frame_id, 'region_pixel_counts': counts, 'arms': {}}
        for arm, root in ARMS.items():
            npz_path = Path(tables[arm][frame_id]['float32_arrays'])
            assert npz_path.parent == root / 'extra_eval/32000/float32'
            pin = next(p for p in summaries[arm]['artifact_pins'] if p.get('artifact') == 'raw_float32_arrays' and p['frame_id'] == frame_id)
            assert pin['path'] == str(npz_path) and npz_path.stat().st_size == pin['bytes']
            with np.load(npz_path, allow_pickle=False) as arrays:
                composite = np.clip(arrays['composite'], 0, 1)
                static = np.clip(arrays['static'], 0, 1)
            assert composite.shape == static.shape == (3, 320, 480) and np.isfinite(composite).all() and np.isfinite(static).all()
            error = np.abs(composite.astype(np.float64) - static.astype(np.float64))
            quantized_difference = np.abs(np.floor(composite * 255. + .5) - np.floor(static * 255. + .5))
            edited = np.any(quantized_difference > 2, axis=0)
            row['arms'][arm] = {key: {'mean_absolute_composite_static_difference': float(error[:, mask].mean()) if counts[key] else None, 'edited_support_gt2_uint8_pixel_fraction': float(edited[mask].mean()) if counts[key] else None} for key, mask in masks.items()}
        rows.append(row)
    aggregate = {}
    for region in masks:
        selected = [r for r in rows if r['region_pixel_counts'][region]]
        aggregate[region] = {'nonempty_frame_count': len(selected), 'empty_frame_ids': [r['frame_id'] for r in rows if not r['region_pixel_counts'][region]], 'pixel_count_summed': sum(r['region_pixel_counts'][region] for r in selected), 'arms': {}}
        for arm in ARMS:
            values = {key: [r['arms'][arm][region][key] for r in selected] for key in ('mean_absolute_composite_static_difference', 'edited_support_gt2_uint8_pixel_fraction')}
            aggregate[region]['arms'][arm] = {'equal_frame_mean': {key: float(np.mean(v)) for key, v in values.items()}, 'pixel_weighted_mean': {key: float(np.average(v, weights=[r['region_pixel_counts'][region] for r in selected])) for key, v in values.items()}}
    result = {'status': 'ACTUAL_COMPLETED_THREE_ARM_PREDEFINED_REMOVAL_LOCALITY_CPU_SUPPLEMENT', 'parent': 30000, 'endpoint': 32000, 'frame_count': 49, 'original_interpretation_sha256': PROTOCOL_SHA, 'reviewer_source_sha256': sha(Path(__file__)), 'canonical_primary_metrics_untouched': True, 'recipe': {'difference': 'mean over RGB and region pixels of abs(clip(composite,0,1)-clip(static,0,1))', 'edited_support': 'any channel abs(floor(clip(composite)*255+.5)-floor(clip(static)*255+.5))>2', 'mask_decode': 'same frozen actor L nearest resize; sky positive float32 OpenCV linear resize then>0', 'source_mask_SHA_checked': True, 'all_six_saved_region_counts_exact': True}, 'frames': rows, 'domains': aggregate, 'mask_recipe_receipts': mask_receipts, 'limitations': ['Difference measures locality relative to the actual composite, not edited-region accuracy or dense ground truth.', 'Actor-mask exterior may include the original cast shadow. Shadow removal can be legitimate change; counts alone are not an editing acceptance test.', 'Masks reconstructed with unchanged pinned recipe and actual source inputs; no independent runtime mask/Scene-pose pixel snapshots are available.', 'Existing NPZ file hashes are bound by the canonical completed CPU export. The supplement verifies actual path/bytes and reads arrays but does not repeat large NPZ/checkpoint hashes.'], 'GPU_used': False, 'normal_algorithm_or_paper_reproduction_claimed': False}
    destination = HERE / 'removal_locality_actual_cpu.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'path': str(destination), 'sha256': sha(destination), 'observed_outside_summary': aggregate['non_sky_outside_actor']}))


if __name__ == '__main__':
    main()
