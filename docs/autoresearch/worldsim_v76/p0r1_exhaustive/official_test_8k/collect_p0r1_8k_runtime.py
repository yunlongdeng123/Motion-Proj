"""Read-only runtime snapshot after the explicit 8k evaluation."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

run = Path('/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000')
code = Path('/root/autodl-tmp/external/worldsim_v75/VAD-GS')
log = (run / 'train.log').read_text(errors='replace')
lines = log.splitlines()
checkpoints = {}
for p in sorted((run / 'trained_model').glob('*.pth')):
    checkpoints[p.name] = {'bytes': p.stat().st_size,
                           'mtime': datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat()}
p = run / 'trained_model/iteration_8000.pth'
digest = hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
checkpoints[p.name]['sha256'] = digest.hexdigest()
progress = [s for s in lines if re.search(r'\d+/30000', s)]
warnings = [s for s in lines if 'RuntimeWarning' in s]
errors = [s for s in lines if re.search(r'Traceback|RuntimeError|CUDA out of memory|Loss=[+-]?(?:nan|inf)', s, re.I)]
priors = {}
for scene in ['scene_0230', 'scene_0255']:
    root = Path('/root/autodl-tmp/data/v76_vadgs') / scene
    priors[scene] = {folder: len(list((root / folder).glob('*.png')))
                     for folder in ['depth_v2', 'normal_img', 'sam_bkgd_masks']}
sources = {}
for name, lo, hi in [('train.py', 238, 248), ('lib/models/trellis.py', 252, 291)]:
    src = (code / name).read_text().splitlines()
    sources[name] = {'first_line': lo, 'last_line': hi, 'lines': src[lo-1:hi]}
result = {
    'recorded_at': datetime.datetime.now().astimezone().isoformat(),
    'task_id': 'VADGS-P0R1-000-TEST-8000',
    'evaluation_exit_code': int((run / 'official_test_8k/exit_code.txt').read_text()),
    'pipeline_state': json.loads((run / 'pipeline_state.json').read_text()),
    'last_training_iteration': int(re.findall(r'(\d+)/30000', log)[-1]),
    'recent_training_progress': progress[-3:],
    'error_matches': errors,
    'runtime_warning_count': len(warnings),
    'unique_runtime_warnings': sorted(set(warnings)),
    'need_extension_count': log.count('need extension'),
    'checkpoints': checkpoints,
    'prior_png_counts': priors,
    'processes': subprocess.check_output(['ps', '-p', '22094,27160,28434', '-o', 'pid,stat,etime,%cpu,rss,args'], text=True),
    'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True),
    'source_context': sources,
    'source_interpretation': 'need extension is a semantic voxel vacancy branch. Propagation is enabled every fifth view-stack epoch before 24000, subject to SSIM and coverage checks. NumPy empty-slice/division warnings are retained; their effect on propagation has not been proven harmless. No algorithm change or training restart in this audit.',
    'failure_ledger_delta': 'none',
    'shutdown_ready': False,
}
target = run / 'official_test_8k/runtime_snapshot.json'
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ['recorded_at', 'last_training_iteration', 'evaluation_exit_code', 'runtime_warning_count', 'unique_runtime_warnings', 'prior_png_counts', 'error_matches']}, ensure_ascii=False))
