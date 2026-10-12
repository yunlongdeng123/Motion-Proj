"""Bind root authorization to the actual reviewed long V and completed originals."""
import datetime,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=HERE/'loss_audit/long_pair_30k_32000/materialized_plan'
R=HERE/'sgcm_data_audit/long_pair_review'
B2=HERE/'continuation_diagnostic/original30k_to32000_noise_reference'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
review=read(R/'final_review_V.json');plan=read(P/'opacity_outside_prepared_plan.json');binding=read(P/'experiment_binding.json')
assert review['status']=='PASS'
assert sha(R/'final_review_V.json')=='f9b25211ed63de56ff7a5cf7bf865648a8cfc59be0d3b4bb8c5c33e12a1b5a07'
pins={
 'continuation_variant_harness.py':review['harness_sha256'],
 'opacity_outside_prepared_plan.json':review['prepared_plan_sha256'],
 'experiment_binding.json':review['experiment_binding_sha256'],
 'outside_experiment_guard.py':review['guard_sha256'],
 'launch_opacity_outside_continuation.py':review['launcher_sha256'],
 'evaluation_protocol_locked.json':review['locked_protocol_sha256'],
}
assert all(sha(P/k)==v for k,v in pins.items())
assert (plan['first_iteration'],plan['stop_iteration'],plan['horizon'],plan['steps'])==(30001,32000,50000,2000)
assert plan['terminal_eval_camera_count']==49
assert plan['training_config_added_fields']==review['extra_config']
assert set(plan['training_config_changed_fields'])=={'model_path','start_checkpoint'}
release=read(B2/'receipts/gpu_release_handoff.json')
for k in ['all2000_steps_finite_logged','all2000_camera_time_shift_LR_exact_to_B1','first10_actual_values_exact','actual_full_saved30k_restore_PASS','all4_eval_nodes_RNG_preserved','terminal_all49_GT_FP32_dimension_regioncount_exact_to_B1','terminal_four_states_hash_bytes_CPU_schema_validated','all_protected_original_SHA_unchanged']:
 assert release[k] is True,k
assert release['worker_exit_code']==release['launcher_exit_code']==0
assert sha(B2/'receipts/B1_B2_observed_noise.json')==review['actual_original_repeat_noise_sha256']
assert sha(B2/'receipts/common49_inputs_after_B2_cpu.json')==review['actual_common49_inputs_after_B2_sha256']
assert sha(HERE/'loss_audit/long_pair_30k_32000/v_binding_cpu/pair_reference_validation.json')==review['actual_B1_B2_pair_sha256']
remote_code='''import json,pathlib,subprocess,fcntl,hashlib,shutil,datetime
b=pathlib.Path('/root/autodl-tmp/drivingeditor_repro');d=b/'campaign/vgpu_bugfix_20261012';p=d/'loss_audit/long_pair_30k_32000/materialized_plan'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not (d/'continuation_diagnostic/opacity_outside30k_to32000').exists()
assert not (p/'opacity_outside_root_coordination_receipt.json').exists()
assert shutil.disk_usage(b).free>30*2**30
names=['continuation_variant_harness.py','opacity_outside_prepared_plan.json','experiment_binding.json','outside_experiment_guard.py','launch_opacity_outside_continuation.py','evaluation_protocol_locked.json']
with (b/'campaign/gpu_slot.lock').open('r+') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_compute_apps':[],'exclusive_lock_available':True,'free_disk_GiB':shutil.disk_usage(b).free/2**30,'V_output_exists':False,'remote_hashes':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names},'review_sha256':hashlib.sha256((d/'sgcm_data_audit/long_pair_review/final_review_V.json').read_bytes()).hexdigest()}))
'''
live=json.loads(subprocess.run(['ssh','-o','ConnectTimeout=15','wm-vgpu-1011','/root/autodl-tmp/envs/drivingeditor/bin/python','-'],input=remote_code,text=True,capture_output=True,check=True).stdout)
assert live['remote_hashes']==pins and live['review_sha256']==sha(R/'final_review_V.json')
receipt={
 'decision':'APPROVED_SINGLE_VARIABLE_OPACITY_OUTSIDE_30K_TO32000','arm':'V_long_outside_opacity',
 'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'prepared_plan_sha256':sha(P/'opacity_outside_prepared_plan.json'),'harness_sha256':plan['harness_sha256'],
 'observer_sha256':plan['observer_sha256'],'experiment_binding_sha256':sha(P/'experiment_binding.json'),
 'guard_sha256':plan['guard_sha256'],'launcher_sha256':sha(P/'launch_opacity_outside_continuation.py'),
 'independent_review_path':'/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/sgcm_data_audit/long_pair_review/final_review_V.json',
 'independent_review_sha256':sha(R/'final_review_V.json'),
 'first_iteration':30001,'stop_iteration':32000,'lr_horizon':50000,'seed':0,
 'GPU_owner':'vgpu_component_diagnostic_solmax','output':plan['output'],
 'algorithm_changed_files':review['algorithm_changed_files'],'extra_config':review['extra_config'],
 'B1_B2_pair_sha256':review['actual_B1_B2_pair_sha256'],
 'B1_B2_actual_noise_sha256':review['actual_original_repeat_noise_sha256'],
 'common_input_receipt_sha256':review['actual_common49_inputs_after_B2_sha256'],
 'B2_release_receipt_sha256':sha(B2/'receipts/gpu_release_handoff.json'),
 'evaluation_protocol_sha256':review['locked_protocol_sha256'],
 'root_live_precheck':live,
 'scope':'Only exact V from original30k complete states to32000, original FP32/kernel/data/seed/50k schedule and loss terms, plus reviewed .05 outside opacity hypothesis. No joint depth/reset/SGCM/static-time changes; originals untouched. Recheck GPU/lock before launch.',
 'required_runtime_evidence':['exact actual full restore','original10 scalars exact BEFORE first backward','all2000 camera/time_shift/LR paired','all4 eval RNG preserved','final49 GT/domain pairing and finite metrics','complete four terminal states','unchanged protected originals','actual worker and launcher exit0'],
 'effect_interpretation':'Compare V separately to both B1/B2 using locked metrics, all49 per-frame errors, actor preservation and real static/dynamic/edit-locality images; one repeated original difference is not a universal CUDA noise bound. Report tradeoffs.',
 'scientific_limitations':'Hypothesis, not author-certified Eq13. Recovery from degraded30k is not from-zero validation or complete paper reproduction.',
 'paper_reproduction_complete':False,'shutdown_authorized_by_this_receipt':False,
}
dest=P/'opacity_outside_root_coordination_receipt.json'
with dest.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
remote='/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan/opacity_outside_root_coordination_receipt.json'
subprocess.run(['scp',str(dest),'wm-vgpu-1011:'+remote],check=True)
print(json.dumps({'receipt':str(dest),'sha256':sha(dest),'decision':receipt['decision'],'live':live}))
