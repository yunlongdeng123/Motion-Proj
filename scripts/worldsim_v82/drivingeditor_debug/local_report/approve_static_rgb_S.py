"""One-shot root receipt for the independently reviewed, isolated S diagnostic."""
import datetime, fcntl, hashlib, json, shutil, subprocess, sys
from pathlib import Path

B = Path('/root/autodl-tmp/drivingeditor_repro')
D = B / 'campaign/vgpu_bugfix_20261012'
S = D / 'loss_audit/next_diagnostics_cpu/observed_static_rgb_S'
P = S / 'materialized_plan'
review = D / 'sgcm_data_audit/observed_static_rgb_S_review/final_review_S.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
assert len(sys.argv) == 2 and sha(review) == sys.argv[1], 'Exact root-read review SHA required'
pins = {
    P / 'static_anchor_prepared_plan.json': '3e738e462f0ef1c68b9f5e23d2a6ad97a9503ce6e54c895b4d55e3e9382df273',
    P / 'continuation_static_anchor_harness.py': '57f06969017b31220f2afbffcc41999108a10404b5ad3519cd7a142587584d65',
    P / 'launch_static_anchor_continuation.py': '5dadc75cae9a515ae3cae09bc033ecde5db03385dd5b9243d5e6c093f531cc9b',
    P / 'experiment_binding.json': '8b98f2f46a9a3a345b2dfac666b9cc32c6090288a55de6870645ddc6978ed5b6',
    S / 'static_anchor_guard.py': '0a879a7a2574c877eb6464a29f7c91948e40e09b29310e34a02a86a5c7c23764',
    S / 'frozen_repository/train.py': 'c4ca7ee8de35d748aac404394a52f65f4f12ff3a1b4d069e31d0c5bb2b8dd3b2',
    S / 'evaluation_protocol_S_locked.json': '13a7bf75015fb90326d9659c49f1947ed63e45e67d03f219e3a23894e44235e2',
}
for path, expected in pins.items():
    assert sha(path) == expected, str(path)
plan, audit = load(P / 'static_anchor_prepared_plan.json'), load(review)
assert audit['status'] == 'PASS'
for key in ['harness_sha256', 'experiment_binding_sha256']:
    assert audit[key] == plan[key]
assert audit['prepared_plan_sha256'] == pins[P / 'static_anchor_prepared_plan.json']
assert plan['first_iteration'] == 30001 and plan['stop_iteration'] == 32000
assert plan['horizon'] == 50000 and plan['seed'] == 0 and plan['algorithm_changes'] == ['train.py']
assert plan['all467_original_inputs_hash_verified']
temporal = load(D / 'component_diagnostic/fixed_pose_temporal32k/results/status.json')
assert temporal['status'] == 'complete' and temporal['actual_exit_code'] == 0 and not temporal['gpu_lock_held']
output = D / 'continuation_diagnostic/observed_static_rgb30k_to32000'
assert not output.exists(), 'Never repeat/overwrite an existing S output'
assert not (P / 'static_anchor_launcher_status.json').exists(), 'S launcher already used'
free_gib = shutil.disk_usage(B).free / 2**30
assert free_gib >= 30
with (B / 'campaign/gpu_slot.lock').open('r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    assert not apps, apps
    receipt = dict(
        decision='APPROVED_SINGLE_VARIABLE_OBSERVED_STATIC_RGB_30K_TO32000',
        approved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), approved_by='root',
        existing_user_authorization='Isolated debugging experiments and reversible fixes; no additional human approval required.',
        prepared_plan_sha256=pins[P / 'static_anchor_prepared_plan.json'],
        harness_sha256=plan['harness_sha256'], observer_sha256=plan['observer_sha256'],
        launcher_sha256=pins[P / 'launch_static_anchor_continuation.py'],
        experiment_binding_sha256=plan['experiment_binding_sha256'], guard_sha256=plan['guard_sha256'],
        independent_review_path=str(review), independent_review_sha256=sys.argv[1],
        first_iteration=30001, stop_iteration=32000, horizon=50000, seed=0, output=str(output),
        execution_owner='vgpu_component_diagnostic_solmax', GPU_apps_before_approval=[],
        exclusive_lock_available_before_approval=True, free_disk_GiB=free_gib,
        exact_addition='Coefficient1 released2024 full-HW observed-static .8L1+.2DSSIM over 1-Md; no V outside-opacity.',
        root_review='Read actual9-line source, original-objective capture before S, exact first10/FP32total/backward gates, strict same-original30k restore and paired protocol. Read all110 temporal renders summary and actual endpoint contacts. Frozen source/canonical evidence preserved.',
        temporal_manifest_sha256=sha(D / 'component_diagnostic/fixed_pose_temporal32k/results/diagnostic_manifest.json'),
        restrictions='Exactly this isolated2000-step recovery diagnostic; no formal50k, from-zero, other scenes or follow-up GPU launch. Original kernel/FP32/data/masks/resolution/seed/horizon/full original loss retained. Paper-unlisted diagnostic, not certified paper formula or complete reproduction.',
    )
    target = P / 'static_anchor_root_coordination_receipt.json'
    with target.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'decision':receipt['decision'], 'receipt':str(target), 'sha256':sha(target), 'free_disk_GiB':free_gib}))
