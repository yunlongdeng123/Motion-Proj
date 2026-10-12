"""Root's reviewed, bounded experiment receipt; does not launch training."""
import datetime,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PLAN=HERE/'component_diagnostic/continuation_plan'
review_path=HERE/'sgcm_data_audit/outside_opacity_review/final_review.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=PLAN/'opacity_outside_prepared_plan.json'
plan=json.loads(plan_path.read_text())
review=json.loads(review_path.read_text())
assert sha(plan_path)=='ef4b873994b8a8d21fcbc96efbce1c6d764249f5b8fdf43be19c60bedbc7153e'
assert review['status']=='PASS' and sha(review_path)=='3c35972dbfaad03c5d212ae848b2be4ea7168f549731a92790edd15bacc98f05'
assert sha(PLAN/'continuation_variant_harness.py')==plan['harness_sha256']==review['harness_sha256']
assert plan['experiment_binding_sha256']==review['experiment_binding_sha256']
assert sha(PLAN/'runtime_safety_parameterized.py')==plan['observer_sha256']
launcher_sha=sha(PLAN/'launch_opacity_outside_continuation.py')
assert launcher_sha=='426f99b3b1a9c75ffcf8e225933065c5fa1ca898dfa566aa7b5e42b6715a94dc'
assert plan['first_iteration']==30001 and plan['stop_iteration']==30300 and plan['horizon']==50000
receipt={
    'decision':'APPROVED_SINGLE_VARIABLE_OPACITY_OUTSIDE_30K_TO30300',
    'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'prepared_plan_sha256':sha(plan_path),
    'harness_sha256':plan['harness_sha256'],
    'observer_sha256':plan['observer_sha256'],
    'experiment_binding_sha256':plan['experiment_binding_sha256'],
    'guard_sha256':plan['guard_sha256'],
    'launcher_sha256':launcher_sha,
    'first_iteration':30001,'stop_iteration':30300,
    'independent_review_path':'/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/sgcm_data_audit/outside_opacity_review/final_review.json',
    'independent_review_sha256':sha(review_path),
    'GPU_owner':'vgpu_component_diagnostic_solmax',
    'scope':'One isolated paired300-step experiment, same original30k state and50k horizon, only outside-opacity hypothesis coefficient.05; no formal resume or other-scene work.',
    'root_review':'Reviewed actual two-file loss delta, isolated imported sources and strict guard, true first-backward comparison, paired sampler/time_shift/LR, RNG-safe same49 evaluation, terminal four-state recording, and original files preservation.',
    'not_author_formula_certification':True,
}
path=PLAN/'opacity_outside_root_coordination_receipt.json'
path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps({'path':str(path),'sha256':sha(path),'decision':receipt['decision']}))
