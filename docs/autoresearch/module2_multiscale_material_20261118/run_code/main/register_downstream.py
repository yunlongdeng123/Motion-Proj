from pathlib import Path
import json,hashlib

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
parent='/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
I=P/'matched_I0/execution_contract.json'
if I.exists():raise FileExistsError('保留已有下游合同')
I.write_text(json.dumps(dict(schema='matched_current_material_RGB_execution.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 coverage_parent=parent,gpu_authorized=True,formal_updates=800,engineering_updates=4,
 method_contract=str(P/'matched_rgb_control_contract.json'),preparation=str(P/'matched_I0/preparation_spec.json'),
 initial='same repaired coverage, authoritative zero-field filteredA0 ci0 and final84266 scope',
 queue='only after shared formal pair ends and root lease becomes stage_qa; no concurrent GPU',
 expected_owner='stage_qa',GPU_keep_on=True,production_promoted=False,human_verdict=None),indent=2)+'\n')
M=P/'trellis2/native_preparation_r1/manifest.json'
expected='8d7c740bc56238031210ad6f889983dc80d9779b37f83ce19f22c38ea30f8a95'
actual=hashlib.sha256(M.read_bytes()).hexdigest();assert actual==expected
T=P/'trellis2/native_execution_contract.json'
if T.exists():raise FileExistsError('保留已有T2 native合同')
T.write_text(json.dumps(dict(schema='trellis2_fixed_prior_native_columns_execution.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 gpu_authorized=True,optimizer_updates=0,frames=[86,110,134,179,198],variants=['T2_mid','A0_mid'],
 prepared_manifest_sha256=actual,coverage_parent=parent,
 shader='same original native Gaussian renderer; current common84266 write mask, all other rows readonly legacy',
 prior='one previously executed fixedmesh TRELLIS2 inference; literalPNG colors, midpoint physical Sigma source-normalized read',
 numerical_reference='same-integrator A0_mid isolates integration effects; exteriorsamefilteredA0',
 expected_native_forward_calls=30,expected_native_backward_calls=0,world_geometry_sigma_opacity_active_frozen=True,
 prior_observed_ledger_unchanged=True,effective_T2_prior_separate=True,
 public_reload='fiveframe road RGB/alpha/RGB8 from actual public loader vs saved float outputs; not claiming separate public joint replay',
 queue='after shared pair and matchedI0; root grants exclusive static_handoff lease then execute',
 expected_owner='static_handoff',GPU_keep_on=True,production_promoted=False,human_verdict=None),indent=2)+'\n')
print('Downstream contracts ready; existing road_asset lease unchanged')
