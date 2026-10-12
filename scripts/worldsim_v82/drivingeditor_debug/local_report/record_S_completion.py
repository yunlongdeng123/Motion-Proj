"""Record only the actual completed S experiment; never changes experiment evidence."""
import datetime
import hashlib
import json
from pathlib import Path

D = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012')
O = D / 'continuation_diagnostic/observed_static_rgb30k_to32000'
P = D / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S/materialized_plan'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
s = read(O / 'receipts/full49_summary.json')
m = read(O / 'receipts/S_vs_originals_locked_metrics.json')
r = read(O / 'receipts/GPU_release_after_S.json')
assert s['harness_exit_code'] == s['wrapper_exit_code'] == 0
assert s['step_count'] == 2000 and s['all4_node_eval_RNG_preserved']
assert s['all_protected_SHA_unchanged'] and len(s['checkpoint_validation']) == 4
assert all(v['all2000_camera_time_shift_LR_exact'] for v in s['paired_originals'].values())
assert m['endpoint_iteration'] == 32000 and len(m['primary_metrics']) == 11
assert r['actual_worker_exit_code'] == r['actual_launcher_exit_code'] == 0
t = read(D / 'task_manifest.json')
assert t['current_experiment'] == 'observed_static_rgb30k_to32000', 'Refuse to replace a future task state'
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
t.update(updated_utc=stamp, current_experiment_status='complete_exit0_full49_fourstate_paired_verified_effect_review_in_progress',
         root_receipt=str(P / 'static_anchor_root_coordination_receipt.json'),
         gpu_owner=None, gpu_owner_scope='S completed. No subsequent GPU experiment approved yet.',
         next_experiment='Interpret S all49 results and independently audit initialization/preprocessing before selecting the next isolated from-zero validation.',
         acceptance_not_yet_met=True, paper_complete_reproduction_claimed=False)
t['observed_static_rgb_S'].update(status='complete_exit0_full49_fourstate_paired_verified', checked_utc=stamp,
    last_complete_iteration_at_check=32000, elapsed_seconds=s['elapsed_seconds'],
    completion_receipt_sha256=sha(O / 'receipts/full49_summary.json'),
    locked_metric_receipt_sha256=sha(O / 'receipts/S_vs_originals_locked_metrics.json'),
    GPU_release_receipt_sha256=sha(O / 'receipts/GPU_release_after_S.json'),
    terminal_manifest_sha256=sha(O / 'terminal_32000/diagnostic_snapshot_manifest.json'),
    actual_exit_code=0, whole_debug_task_complete=False,
    metrics=[{k:v[k] for k in ['region','array','metric','equal_frame_mean_B1','equal_frame_mean_B2','equal_frame_mean_S']} for v in m['primary_metrics']])
a = D / 'loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/static_initialisation_actor_audit.json'
if a.exists():
    q = read(a)
    t['static_initialization_audit'] = dict(receipt=str(a), sha256=sha(a), counts=q['counts'],
        interpretation='Selected actor returns are excluded exactly, but native point-instance witnesses identify additional moving-target returns retained in static. Sparse annotation coverage does not establish the full contamination rate or the cause of confetti.',
        source_inputs_unchanged=q['all_audited_input_and_S_source_SHA_unchanged_after'])
t['parallel_CPU_work'] = {'M_detector':'preprocessing_gap_audit/mdetector_adapter_candidate; isolated contract/source checks, no native inference or changed training inputs',
    'static_initialization':'loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu; original return lineage and independent native labels',
    'shadow_weights':'preprocessing_gap_audit/shadow_weight_access_20261012.json; bounded public search, no weights downloaded'}
tmp = D / 'task_manifest.S_completed.tmp'
tmp.write_text(json.dumps(t,ensure_ascii=False,indent=2)+'\n')
tmp.replace(D / 'task_manifest.json')
print(json.dumps({'status':t['current_experiment_status'],'checked_utc':stamp,'acceptance_not_yet_met':True}))
