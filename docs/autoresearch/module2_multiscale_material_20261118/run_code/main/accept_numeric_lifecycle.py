from pathlib import Path
import json,time
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
n=json.loads((P/'cpu_numeric_r1/receipt.json').read_text())
x=json.loads((P/'qa/chart_external_locator_crosscheck.json').read_text())
assert n['numeric_gate_pass'] is True
for t in n['tests']:
    assert t['mass_ge_point9']['RGB_error_p99']<=1/255 and t['mass_ge_point9']['RGB_error_max']<=4/255
assert max(t['absolute_error'] for t in n['frequency'])<1e-4
lease=json.loads((P/'gpu_lease.json').read_text());assert lease['active'] and lease['owner']=='road_asset'
lease['scope']='registered shared-field lifecycle only: <=12 engineering optimizer updates; formal still locked'
(P/'gpu_lease.json').write_text(json.dumps(lease,indent=2)+'\n')
(P/'qa/root_preflight_acceptance.json').write_text(json.dumps(dict(
 schema='module2_root_numeric_and_chart_review.v1',UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 numeric_gate_pass=True,chart_source_review='root inspected actual exact-node/structured inverse and external AABB crosscheck',
 reference='CPU numerical engineering reference, not strict global representation certificate',
 engineering_optimizer_updates_authorized=12,formal_optimizer_updates_authorized=0,
 native_scope_updated=True,GPU_keep_on=True),indent=2)+'\n')
(P/'loss_spec.json').write_text(json.dumps(dict(schema='module2_binary_validity_premult_loss.v1',
 per_pixel_q=dict(A=1.,C=1.,S=1.),
 q_meaning='binary frozen validity mask, not calibrated teacher correctness; old zero photo-q is not reused on new909',
 target='fixed_current_alpha * target_sRGB',render='actual native premult road_sRGB',
 normalization='each A/C/S sumRGB_abs/(3*fixed mask pixels)',weights=dict(A=1.,C=.5,S=1.),
 regularizers='execution_contract optimizer.regularizers; train-world-support normalized',
 fullS=3187,photoS=3131,old_photoS=2222,new_supported_diagnostic=909,low56='kept in full-domain reporting, no opacity optimization'),indent=2)+'\n')
print('CPU gates accepted; actual shared-field lifecycle permitted, formal locked')
