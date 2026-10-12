"""Authorize the same bounded experiment after a pre-training import-check fix."""
import datetime,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PLAN=HERE/'component_diagnostic/continuation_plan'
review_path=HERE/'sgcm_data_audit/outside_opacity_review/final_review_v2.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=PLAN/'opacity_outside_prepared_plan.json'
plan=json.loads(plan_path.read_text())
review=json.loads(review_path.read_text())
assert sha(plan_path)=='be9e385b37805457d2d9986985c9b7bd65af3fcde66445d3f15d156db5c90a93'
assert review['status']=='PASS'
assert sha(PLAN/'continuation_variant_harness.py')==plan['harness_sha256']==review['harness_sha256']=='3a2f9add3cab0763c4b0ef612fd5b735afd7f2d337b151306da1a4ceffcb6154'
assert plan['experiment_binding_sha256']==review['experiment_binding_sha256']=='b96ec92c0694a251e39da85a3ae4abab131366a75ad52cd47098ec0df953b214'
assert review['prepared_plan_sha256']==sha(plan_path)
assert sha(PLAN/'runtime_safety_parameterized.py')==plan['observer_sha256']
launcher_sha=sha(PLAN/'launch_opacity_outside_continuation.py')
assert launcher_sha=='426f99b3b1a9c75ffcf8e225933065c5fa1ca898dfa566aa7b5e42b6715a94dc'
assert plan['first_iteration']==30001 and plan['stop_iteration']==30300 and plan['horizon']==50000
receipt={
    'decision':'APPROVED_SINGLE_VARIABLE_OPACITY_OUTSIDE_30K_TO30300',
    'attempt':2,'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'prepared_plan_sha256':sha(plan_path),'harness_sha256':plan['harness_sha256'],
    'observer_sha256':plan['observer_sha256'],'experiment_binding_sha256':plan['experiment_binding_sha256'],
    'guard_sha256':plan['guard_sha256'],'launcher_sha256':launcher_sha,
    'first_iteration':30001,'stop_iteration':30300,
    'independent_review_path':'/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/sgcm_data_audit/outside_opacity_review/final_review_v2.json',
    'independent_review_sha256':sha(review_path),'GPU_owner':'vgpu_component_diagnostic_solmax',
    'scope':'Same single-variable300-step hypothesis. Prior failed startup never entered training. Only observer import check corrected to actual utils.loss_utils.psnr; algorithm/config/guard/weights unchanged.',
    'prior_failure_evidence':'/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/continuation_diagnostic/attempts/opacity_outside_01_runtime_image_utils_import',
    'not_author_formula_certification':True,
}
path=PLAN/'opacity_outside_root_coordination_receipt.json'
path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps({'path':str(path),'sha256':sha(path),'decision':receipt['decision'],'review_sha256':sha(review_path)}))
