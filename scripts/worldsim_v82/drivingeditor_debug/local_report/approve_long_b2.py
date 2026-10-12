"""Authorize exact reviewed B2 after actual B1 completion; never launch training."""
import datetime, hashlib, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / 'loss_audit/long_pair_30k_32000/materialized_plan'
B2 = P / 'baseline_noise_reference'
R = HERE / 'sgcm_data_audit/long_pair_review'
B1 = HERE / 'continuation_diagnostic/original30k_to32000'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
review = read(R / 'final_review_B2.json')
plan = read(B2 / 'prepared_plan.json')
assert review['status'] == 'PASS'
assert sha(R/'final_review_B2.json') == '98060871efbd303b495a6376d97b5728320c5be0a0a2dbeabfebe2b82be36b00'
for name, key in [('prepared_plan.json','prepared_plan_sha256'),('continuation_harness.py','harness_sha256'),('launch_controlled_continuation.py','launcher_sha256')]:
    assert sha(B2/name) == review[key]
assert (plan['first_iteration'],plan['stop_iteration'],plan['horizon'],plan['steps']) == (30001,32000,50000,2000)
assert set(plan['training_config_changed_fields']) == {'model_path','start_checkpoint'}
assert plan['algorithm_changes'] == [] and plan['terminal_eval_camera_count'] == 49
assert sha(B1/'receipts/full49_summary.json') == '36879999ab39e208b0cb031e7557c83a3dabc11f32e361687796b8e641eced31'
release = read(B1/'receipts/gpu_release_handoff.json')
assert release['actual_step_count'] == 2000 and release['actual_worker_exit_code'] == release['actual_launcher_exit_code'] == 0
assert read(B1/'receipts/full49_summary.json')['status'] == 'all_long_original_receipts_verified_CPU'
assert sha(B1/'terminal_32000/diagnostic_snapshot_manifest.json') == '6d30f1583186570d0850c60ff9195a9048110cd538a05289edc0b6a4d499e340'
remote_code = '''import json,pathlib,subprocess,fcntl,hashlib,shutil,datetime
b=pathlib.Path('/root/autodl-tmp/drivingeditor_repro'); d=b/'campaign/vgpu_bugfix_20261012'; p=d/'loss_audit/long_pair_30k_32000/materialized_plan/baseline_noise_reference'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not apps, apps
assert not (d/'continuation_diagnostic/original30k_to32000_noise_reference').exists()
assert not (p/'root_coordination_receipt.json').exists()
assert shutil.disk_usage(b).free>30*2**30
with (b/'campaign/gpu_slot.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_compute_apps':[],'exclusive_lock_available':True,'free_disk_GiB':shutil.disk_usage(b).free/2**30,'B2_output_exists':False,'remote_hashes':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['prepared_plan.json','continuation_harness.py','launch_controlled_continuation.py']}}))
'''
result = subprocess.run(['ssh','-o','ConnectTimeout=15','wm-vgpu-1011','/root/autodl-tmp/envs/drivingeditor/bin/python','-'], input=remote_code, text=True, capture_output=True, check=True)
live = json.loads(result.stdout)
assert all(v == sha(B2/k) for k,v in live['remote_hashes'].items())
receipt = {
    'decision':'APPROVED_CONTROLLED_30K_TO_32000', 'arm':'B2_original_repeat',
    'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'prepared_plan_sha256':sha(B2/'prepared_plan.json'), 'harness_sha256':plan['harness_sha256'],
    'observer_sha256':plan['observer_sha256'], 'launcher_sha256':sha(B2/'launch_controlled_continuation.py'),
    'first_iteration':30001,'stop_iteration':32000,'lr_horizon':50000,
    'GPU_owner':'vgpu_component_diagnostic_solmax','independent_review_sha256':sha(R/'final_review_B2.json'),
    'evaluation_protocol_sha256':sha(P/'evaluation_protocol_locked.json'),
    'B1_completion_receipt_sha256':sha(B1/'receipts/full49_summary.json'),
    'B1_terminal_manifest_sha256':sha(B1/'terminal_32000/diagnostic_snapshot_manifest.json'),
    'B1_GPU_release_receipt_sha256':sha(B1/'receipts/gpu_release_handoff.json'),
    'root_live_precheck':live,
    'scope':'Only exact reviewed original-repeat B2 30001..32000 from original30k four states, original FP32 math/kernel/config/50k schedule. No variant or other GPU authorization. Recheck GPU/lock on launch.',
    'scientific_reason':'B1 composite recovered but separation remains poor. Paired original repeat measures observed numerical variation before long candidate interpretation; not a universal noise bound.',
    'success_conditions':['actual full restore and first10 scalar comparison','complete finite2000 steps','all2000 camera/time_shift/LR exact paired to B1','all49 identical GT/mask domains','all4 node RNG preserved','complete four terminal states','protected originals unchanged','actual exit0'],
    'paper_reproduction_complete':False,
}
dest=B2/'root_coordination_receipt.json'
with dest.open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2)
remote='/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan/baseline_noise_reference/root_coordination_receipt.json'
subprocess.run(['scp',str(dest),'wm-vgpu-1011:'+remote],check=True)
print(json.dumps({'path':str(dest),'sha256':sha(dest),'decision':receipt['decision'],'live':live}))
