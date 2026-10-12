"""Root authorization for only the reviewed 2,000-step original-objective arm."""
import datetime, hashlib, json, subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLAN=HERE/'loss_audit/long_pair_30k_32000/materialized_plan'
REVIEW=HERE/'sgcm_data_audit/long_pair_review'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=PLAN/'prepared_plan.json';plan=json.loads(plan_path.read_text())
review_path=REVIEW/'materialized_and_prepared_review_cpu.json';review=json.loads(review_path.read_text())
remote_path=REVIEW/'remote_source_plan_receipt_cpu.json';remote=json.loads(remote_path.read_text())
assert review['status']=='PASS_CPU_SOURCE_CONFIG_REVIEW_NO_GPU_AUTHORIZATION'
assert all(c['PASS'] for c in review['checks']) and remote['status']=='PASS'
assert remote['independent_source_config_review_sha256']==sha(review_path)
assert sha(plan_path)==review['actual_prepared_original_arms']['B1']['sha256']=='359b9d62ee9aec28951532cedbaed45401838dd6ea2b961d5704bec2d303a2d5'
assert sha(PLAN/'continuation_harness.py')==plan['harness_sha256']=='88a2df38e5ac4df5905f1de030701189763d469f9eb3daf9e3339f1931aba829'
assert sha(PLAN/'launch_controlled_continuation.py')=='6337b0c660d1eedb95b822d87d0feaa6d5691ba22da4de8d8d54fb750a7e68b0'
assert (plan['first_iteration'],plan['stop_iteration'],plan['horizon'],plan['steps'])==(30001,32000,50000,2000)
assert set(plan['training_config_changed_fields'])=={'model_path','start_checkpoint'}
assert plan['algorithm_changes']==[] and plan['terminal_eval_camera_count']==49
assert sha(PLAN/'evaluation_protocol_locked.json')==review['loss_agent_locked_protocol_sha256']=='c725cd2153d1104ddaf81078ac753f4e7f474e0be325ca9fbb5b1fccb342f658'
receipt={
 'decision':'APPROVED_CONTROLLED_30K_TO_32000',
 'arm':'B1_original_objective',
 'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'prepared_plan_sha256':sha(plan_path),'harness_sha256':plan['harness_sha256'],'observer_sha256':plan['observer_sha256'],
 'launcher_sha256':sha(PLAN/'launch_controlled_continuation.py'),
 'first_iteration':30001,'stop_iteration':32000,'lr_horizon':50000,
 'GPU_owner':'vgpu_component_diagnostic_solmax',
 'independent_review_sha256':sha(review_path),'remote_source_plan_verification_sha256':sha(remote_path),
 'evaluation_protocol_sha256':sha(PLAN/'evaluation_protocol_locked.json'),
 'root_live_precheck':{'free_disk_GiB':91.73259353637695,'GPU_compute_apps':[],'B1_output_exists':False,'check_before_launch_again':True},
 'scope':'Exactly one original-objective30k-to32000 arm. Same full30k state, data, FP32 original kernel and50k schedule. Preserve original files. No B2 or variant GPU authorization in this receipt.',
 'scientific_reason':'300-step hypothesis substantially improved observed static background with small composite cost but persistent leakage. Longer paired reference and original repeat are needed before judging a fix.',
 'first_step_requirement':'Verify actual full restoration and report all10 original first-step scalar comparisons; do not claim trajectory equivalence from a single step.',
 'success_conditions':['exact approved source/config/runtime','complete2000 original-camera/time-shift/LR records','all RNG-preserving eval nodes and final49','finite results','four terminal state files and unchanged protected original files','actual exit0'],
 'paper_reproduction_complete':False,
}
dest=PLAN/'root_coordination_receipt.json'
with dest.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
subprocess.run(['scp',str(dest),'wm-vgpu-1011:/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/materialized_plan/root_coordination_receipt.json'],check=True)
print(json.dumps({'path':str(dest),'sha256':sha(dest),'decision':receipt['decision']}))
