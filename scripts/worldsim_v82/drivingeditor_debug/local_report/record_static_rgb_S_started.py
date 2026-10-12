"""Record a live, verified S startup without changing any experiment files."""
import datetime, hashlib, json, subprocess
from pathlib import Path
D = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012')
S = D / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S'
P = S / 'materialized_plan'
O = D / 'continuation_diagnostic/observed_static_rgb30k_to32000'
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
status = load(O / 'status.json')
compose = load(O / 'first_original_compose_reference_comparison.json')
backward = load(O / 'first_backward_reference_comparison.json')
launcher = load(P / 'static_anchor_launcher_status.json')
assert status['status'] == 'controlled_training' and status['gpu_lock_held']
assert status['restore']['status'] == 'actual_restore_pass'
assert all(x['exact_equal'] for x in compose['comparisons'].values())
assert not compose['added_static_RGB_executed']
assert all(x['exact_equal'] for x in backward['comparisons'].values())
assert backward['total_decomposition_exact'] and backward['original_backward_returned']
assert not (P / 'static_anchor_launcher.exit').exists()
pid = status['owner_pid']
cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
assert str(P / 'continuation_static_anchor_harness.py') in cmdline
rows = [json.loads(line) for line in (O / 'step_losses.jsonl').read_text().splitlines() if line.endswith('}')]
last = rows[-1]['iteration']
assert [x['iteration'] for x in rows] == list(range(30001, last + 1))
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
t = load(D / 'task_manifest.json')
assert t['current_experiment'] in ['fixed_pose_temporal32k', 'observed_static_rgb30k_to32000']
t.update(updated_utc=stamp, current_experiment='observed_static_rgb30k_to32000',
    current_experiment_status='controlled_training_first_restore_and_preS_original10_and_backward_PASS',
    gpu_owner='vgpu_component_diagnostic_solmax',
    gpu_owner_scope='Exactly this S diagnostic only; no other GPU job approved.',
    gpu_status_path='continuation_diagnostic/observed_static_rgb30k_to32000/status.json',
    next_experiment='Finish S and compare all49 locked metrics/actor preservation/locality against B1 and B2; inspect visual results before deciding any further exact reviewed training.',
    acceptance_not_yet_met=True, paper_complete_reproduction_claimed=False)
t['observed_static_rgb_S'] = dict(status='running',
    output=str(O), plan_directory=str(P), historical_worker_pid=pid,
    historical_launcher_pid=launcher['launcher_pid'], started_utc=launcher['started_utc'],
    checked_utc=stamp, last_complete_iteration_at_check=last,
    root_receipt_sha256=sha(P / 'static_anchor_root_coordination_receipt.json'),
    prepared_plan_sha256=sha(P / 'static_anchor_prepared_plan.json'),
    independent_review_sha256=sha(D / 'sgcm_data_audit/observed_static_rgb_S_review/final_review_S.json'),
    original30k_full_restore=status['restore']['status'],
    original10_exact_before_computing_S=True, first_backward_returned=True,
    total_FP32_decomposition_exact=True, first_original_loss=backward['original_objective_loss'],
    first_S_unweighted=backward['observed_static_rgb_unweighted'], first_total_loss=backward['actual_new_total_loss'],
    coefficient=1.0, first_iteration=30001, stop_iteration=32000, horizon=50000,
    scientific_status='Single restored2024 observed-static RGB term added to paper objective, absent from displayed paper Eq10. No V term. Recovery diagnostic, not from-zero success or author-certified paper formula.',
    progress_policy='Read last complete step_losses.jsonl row, current /proc PID/cmd/apps/lock and actual launcher exit; node status updates only at30001/30300/31000/32000. Never relaunch from an old PID or stale node status.')
t['fixed_pose_temporal']['unit_verification'] = 'XYZ displacement computed in model coordinates then divided by pinned scene scale0.1 to original Waymo metric length; no extra Scene geometry scale. Pixel changes are unit-independent.'
t['fixed_pose_temporal']['interpretation'] = 'At fixed raw10 pose, static output changes over49 times. Endpoint temporal-opacity/support control reduces output changes much more than position-only control; freezing both exactly reproduces the anchor. Raw0 is one SHM cycle from raw10, a special case. No novel-time GT-quality or training-repair claim.'
tmp = D / 'task_manifest.S_started.tmp'
tmp.write_text(json.dumps(t, ensure_ascii=False, indent=2) + '\n')
tmp.replace(D / 'task_manifest.json')
print(json.dumps({'status':t['current_experiment_status'], 'iteration':last, 'worker_pid':pid, 'checked_utc':stamp}))
