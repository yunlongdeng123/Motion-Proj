"""Verify completed background files without launching inference or training."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

cv2.setNumThreads(1)
root = Path('/root/autodl-tmp/data/v76_vadgs')
expected = {f'{frame:03d}_{camera}' for frame in range(61) for camera in range(6)}
result = {'task_id': 'VADGS-BACKGROUND-COMPLETE-20260926',
          'recorded_at': datetime.datetime.now().astimezone().isoformat(),
          'failure_ledger_refs': ['V76-F02'], 'failure_ledger_delta': 'none',
          'matched_scene_training_admitted': False, 'scenes': {}}
for name in ['scene_0230', 'scene_0255']:
    scene = root / name
    report_path = scene / 'sam_prior_evidence/background_report.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' and report['background_only']
    assert len(report['views']) == 366
    assert {r['name'] for r in report['views']} == expected
    counts = {}
    for folder in ['depth_v2', 'normal_img', 'sam_bkgd_masks']:
        actual = {p.stem for p in (scene / folder).glob('*.png')}
        assert actual == expected, (name, folder, expected - actual, actual - expected)
        counts[folder] = len(actual)
    guard = scene / 'sam_prior_evidence/DYNAMIC_IDENTITY_BLOCKED.json'
    assert guard.exists()
    assert not list((scene / 'sam_masks').glob('*.png'))
    files = []
    for key in sorted(expected):
        path = scene / 'sam_bkgd_masks' / (key + '.png')
        encoded = path.read_bytes()
        arr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert arr is not None and arr.dtype == np.uint8 and arr.shape == (900, 1600), (name, key)
        hist = np.bincount(arr.reshape(-1), minlength=256)
        labels = np.flatnonzero(hist)
        assert hist[1] == 0 and hist[255] == 0 and np.any(labels >= 2), (name, key)
        files.append({'name': key, 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(),
                      'region_labels': int((labels >= 2).sum()), 'unassigned_fraction': float(hist[0] / arr.size)})
    result['scenes'][name] = {
        'expected_views': 366, 'counts': counts, 'background_files_decoded_and_validated': 366,
        'background_shape': [900, 1600], 'background_dtype': 'uint8',
        'report_status': report['status'], 'report_mtime': datetime.datetime.fromtimestamp(report_path.stat().st_mtime).astimezone().isoformat(),
        'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'invocation_device': report.get('invocation_device'), 'invocation_seconds': report.get('seconds'),
        'reused_existing_views': sum(bool(r.get('reused_existing_files')) for r in report['views']),
        'new_view_devices': dict(Counter(r.get('device', 'unspecified') for r in report['views'] if not r.get('reused_existing_files'))),
        'dynamic_identity_blocked': json.loads(guard.read_text()),
        'files': files,
    }
run = Path('/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000')
log = (run / 'train.log').read_text(errors='replace')
result['main_pipeline_state'] = json.loads((run / 'pipeline_state.json').read_text())
result['last_training_iteration'] = int(re.findall(r'(\d+)/30000', log)[-1])
result['training_error_lines'] = [s for s in log.splitlines() if re.search(r'Traceback|RuntimeError|CUDA out of memory|Loss=[+-]?(?:nan|inf)', s, re.I)]
result['training_runtime_warnings'] = sorted({s for s in log.splitlines() if 'RuntimeWarning' in s})
result['process_snapshot'] = subprocess.check_output(['ps', '-p', '22094,27160,28434', '-o', 'pid,stat,etime,%cpu,rss,args'], text=True)
result['shutdown_ready'] = False
result['scope_note'] = 'Background encoding/completeness only, not semantic accuracy or dynamic identity approval. No new model inference. Continue existing P0R1 30k controller.'
target = root / 'background_completion_20260926.json'
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'result': str(target), 'last_training_iteration': result['last_training_iteration'],
                  'scenes': {k: {a: b for a, b in v.items() if a not in ['files', 'dynamic_identity_blocked']} for k, v in result['scenes'].items()},
                  'training_error_lines': result['training_error_lines']}, ensure_ascii=False))
