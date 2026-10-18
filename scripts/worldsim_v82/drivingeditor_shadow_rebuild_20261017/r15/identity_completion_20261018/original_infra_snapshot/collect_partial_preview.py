"""CPU-only audit and packaging of existing partial real 30k evaluation.

No render, training, inference, inferred camera matrices or fabricated final gate.
"""
import json
import os
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone

os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
EVAL = BASE / 'post30k_r2/evaluation'
EXPORT = BASE / 'preview_export_r1'
PREPARED = BASE / 'prepared/scene0008_two_actor_v1'
REFERENCE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013/reports/gmc_gn11_30k_saved_cpu_20261013_0910/manifest.json')
sys.path.insert(0, str(BASE / 'infra'))
from evaluate_scene import validate_arrays, observed_metrics, read_mask_labels
import numpy as np
from PIL import Image

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def pixels(array):
    display = array[0] if array.shape[0] == 1 else array.transpose(1, 2, 0)
    return (display.clip(0, 1) * 255).round().astype(np.uint8)

def verify_png(path, array):
    with Image.open(path) as source:
        if source.size != (480, 320):
            raise ValueError('PNG native size mismatch: ' + str(path))
        actual = np.asarray(source)
    if not np.array_equal(actual, pixels(array)):
        raise ValueError('Saved PNG differs from actual FP32 display: ' + str(path))

if EXPORT.exists():
    raise FileExistsError('Preserve previous partial audit/export: ' + str(EXPORT))
if (EVAL / 'scene_evaluation_manifest.json').exists():
    raise ValueError('This tool only audits the retained partial evaluation')
yield_receipt = read(EVAL / 'priority_yield.json')
if yield_receipt.get('status') != 'YIELDED_TO_PRIORITY_WORK':
    raise ValueError('Expected actual priority yield receipt')
source_cameras = {row['img_name']: row for row in read(EVAL / 'cameras.json')}
reference = {Path(row['id']).name: row for row in read(REFERENCE)['frames']}
with np.load(PREPARED / 'instances.npz', allow_pickle=False) as bundle:
    ids = bundle['instance_ids'].tolist()
    frames = bundle['frame_ids'].tolist()
    times = bundle['timestamps'].astype(float).tolist()
    known = bundle['mask_known'].copy()
    valid = bundle['valid'].copy()
if ids != ['NoQttdyd4C16Y_zkGhiFKg', 'fXVhuN-RuJDvZwnkQ-0YSQ'] or not known.all():
    raise ValueError('Real two actor identity/mask readiness mismatch')
files = sorted((EVAL / 'observed').glob('*.npz'))
count = yield_receipt['completed_frames']['observed']
if len(files) != count or not 0 < count < 49 or yield_receipt['completed_frames']['fixed_camera'] != 0:
    raise ValueError('Actual retained FP32 count differs from priority receipt')
rows, png_paths = [], []
fields = ('composite', 'static_rgb', 'static_raw_rgb', 'all_vehicles_raw_rgb', 'scene_alpha',
          'dynamic_visible_alpha', 'static_visible_alpha', 'all_vehicles_isolated_raw_alpha', 'gt')
for index, path in enumerate(files):
    frame = frames[index]
    prefix = '%03d_%s' % (index, frame)
    if path.stem != prefix:
        raise ValueError('Partial frames are not a genuine contiguous observed prefix')
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name] for name in bundle.files}
    validate_arrays(arrays, ids, 320, 480, observed=True)
    labels = read_mask_labels(PREPARED / 'instance_masks' / (frame + '.png'), 480, 320, 2)
    expected_masks = np.stack([(labels == actor + 1).astype(np.float32) for actor in range(2)])
    if not np.array_equal(arrays['observed_masks'], expected_masks):
        raise ValueError('Actual saved masks differ from real DEVA identity labels: ' + frame)
    if any(expected_masks[actor].any() and not valid[index, actor] for actor in range(2)):
        raise ValueError('Mask/trajectory validity mismatch')
    metrics = observed_metrics(arrays, ids)
    metrics['RGB_SSIM_clipped'] = None
    metrics['SSIM_available_from_retained_partial_assets'] = False
    images, instances = {}, []
    for field in fields:
        png = EVAL / 'observed' / (prefix + '_' + field + '.png')
        verify_png(png, arrays[field])
        images[field] = png.relative_to(EVAL).as_posix()
        png_paths.append(png)
    reference_gt = REFERENCE.parent / reference[frame]['images']['gt']
    with Image.open(reference_gt) as source:
        gt_pixels = np.asarray(source.convert('RGB'))
    if not np.array_equal(gt_pixels, pixels(arrays['gt'])):
        raise ValueError('Actual observed GT differs from independently saved GMC GT: ' + frame)
    for actor, track in enumerate(ids):
        item_images = {}
        for label, value in (
            ('visible_rgb', arrays['instance_visible_rgb'][actor]),
            ('visible_alpha', arrays['instance_visible_contribution'][actor:actor + 1]),
            ('body_mask', arrays['observed_masks'][actor:actor + 1])):
            png = EVAL / 'observed' / (prefix + '_instance%02d_' % actor + label + '.png')
            verify_png(png, value)
            item_images[label] = png.relative_to(EVAL).as_posix()
            png_paths.append(png)
        instances.append({'index': actor, 'track_id': track, 'images': item_images,
                          'metrics': metrics['instances'][actor]})
    if frame not in source_cameras:
        raise ValueError('Missing actual source camera JSON row')
    rows.append({'frame_index': index, 'frame_id': frame, 'model_time': times[index],
                 'camera_frame_id': frame, 'camera': {'width': 480, 'height': 320},
                 'source_camera_JSON': source_cameras[frame],
                 'source_camera_JSON_scope': 'source calibration provenance; GPU matrices not independently serialized',
                 'source_FP32_remote_path': str(path), 'images': images, 'instances': instances,
                 'all_FP32_finite': True, 'metrics': metrics})
proof = read(BASE / 'post30k_r1/restore_030000/receipt.json')
if proof.get('step') != 30000 or proof.get('runtime_restoration_verified') is not True:
    raise ValueError('Genuine 30k restoration receipt missing')
if 'torch' in sys.modules:
    raise AssertionError('CPU audit must not import Torch')
manifest = {
    'schema': 'drivingeditor_rigid_scene_partial_preview_v1',
    'status': 'PARTIAL_ACTUAL30K_OBSERVED_PREVIEW', 'scene_id': 'scene0008_023_071',
    'checkpoint': {'path': str(BASE / 'runs/gmc_two_actor_fixed30k_r1/checkpoint_030000.pt'), 'step': 30000},
    'instance_ids': ids, 'streams': {'observed': {'GT_available': True, 'frames': rows},
                                  'fixed_camera': {'GT_available': False, 'frames': []}},
    'contracts': {'all_FP32_finite': True, 'identity_and_native_size_verified': True,
                  'GT_reference_PNG_verified': True, 'real_body_mask_PNG_verified': True,
                  'observed_frames_verified': count, 'fixed_camera_frames': 0, 'edit_operations': 0,
                  'model_and_optimizer_unmodified_after_evaluation': None},
    'partial_reason': 'Actual StreetCrafter preparation priority detected; evaluation yielded with assets preserved',
    'priority_yield_source': str(EVAL / 'priority_yield.json'),
    'limits': ['Observed prefix only, not complete49 aggregate or quality verdict',
               'Fixed camera and six edit operations not exported',
               'End-of-evaluation readonly model/Adam verification not reached',
               'No independently serialized GPU camera matrices in retained partial assets',
               'Per-frame CPU metrics recomputed from actual FP32; SSIM unavailable in partial receipt'],
    'human_quality_verdict': None, 'Torch_imported_for_CPU_audit': False,
    'generated_UTC': datetime.now(timezone.utc).isoformat(), 'verified_PNG_count': len(png_paths)}
EXPORT.mkdir()
manifest_path = EXPORT / 'partial_preview_manifest.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
with tarfile.open(EXPORT / 'partial_preview_assets.tar.gz', 'w:gz') as archive:
    archive.add(manifest_path, arcname=manifest_path.name)
    for path in png_paths:
        archive.add(path, arcname=path.relative_to(EVAL).as_posix())
print(json.dumps({'status': 'ACTUAL_PARTIAL_CPU_ASSET_AUDIT_PASS', 'frames': count,
                  'PNG_count': len(png_paths), 'Torch_imported': False,
                  'package': str(EXPORT / 'partial_preview_assets.tar.gz')}))
