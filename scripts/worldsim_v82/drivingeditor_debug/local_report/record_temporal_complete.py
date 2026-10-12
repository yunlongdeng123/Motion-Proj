import datetime, hashlib, json
from pathlib import Path

D = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012')
P = D / 'component_diagnostic/fixed_pose_temporal32k'
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
s, m = load(P / 'results/status.json'), load(P / 'results/diagnostic_manifest.json')
assert s['status'] == 'complete' and s['actual_exit_code'] == 0
assert s['completed_render_count'] == 110 and not s['gpu_lock_held']
assert (P / 'launcher.exit').read_text().strip() == '0'
assert m['original_file_SHA_unchanged'] and not m['training_or_backward_started']
t = load(D / 'task_manifest.json')
assert t['current_experiment'] in ['opacity_outside30k_to32000', 'fixed_pose_temporal32k']
t.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         current_experiment='fixed_pose_temporal32k',
         current_experiment_status='complete_readonly_110_renders_exit0',
         gpu_owner=None, gpu_owner_scope='Temporal read-only task finished. S observed-static RGB training remains in CPU preparation; no training approved yet.',
         gpu_status_path='component_diagnostic/fixed_pose_temporal32k/results/status.json',
         next_experiment='Isolated observed-static RGB S source/config/harness/guard review, then exact root approval if valid. No formal50k or other scenes.',
         acceptance_not_yet_met=True, paper_complete_reproduction_claimed=False)
t['fixed_pose_temporal'] = dict(status='complete_exit0', actual_render_count=98, counterfactual_render_count=12,
    training_or_backward=False, original_files_unchanged=True,
    manifest_sha256=sha(P / 'results/diagnostic_manifest.json'),
    root_receipt_sha256=sha(P / 'root_coordination_receipt.json'),
    interpretation='At fixed raw10 pose, static output changes with time. Endpoint controls identify temporal opacity/support as the larger source of change than SHM position; freezing both reproduces the native anchor exactly. This does not establish novel-time quality or a training repair. Model-coordinate units require a separate scale receipt.',
    GPU_finished_utc=s['finished_utc'])
tmp = D / 'task_manifest.temporal_update.tmp'
tmp.write_text(json.dumps(t, ensure_ascii=False, indent=2) + '\n')
tmp.replace(D / 'task_manifest.json')
print(json.dumps({'current_experiment':t['current_experiment'], 'status':t['current_experiment_status'], 'manifest_sha256':t['fixed_pose_temporal']['manifest_sha256']}))
