from pathlib import Path
import json,time
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
p=P/'numeric_filter_contract.json'
if p.exists():raise FileExistsError('保留数值门')
d=dict(schema='world_material_filter_numeric_gate.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 registered_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),before_formal_or_smoke_optimizer=True,
 candidate='physical world tangent GH15 + exact fixed chart inverse; bilinear world materials, mH*H inside filter',
 reference='dense GH41 actual-world queries; numerical reference only, not an exact global error certificate',
 kernel_failure_preserved=dict(GH7_maxsigma_m=.01955,wavelength_m=.04,analytic_response=.0089577666,GH7_response=.0500198248),
 chart_worldXY_max_error_m=1e-7,
 kernel_unit_Nyquist_response_abs_error_max=1e-4,
 material_kernel_valid_mass_primary_min=.9,
 actual_RGB_primary_abs_error_p99_max=1/255,actual_RGB_primary_abs_error_max=4/255,
 sample_policy='seed101 stratified fixed rows including source-view S, new892, true observed paint/detail, actual largest-sigma rows; counts/source/overlap explicit, do not select by low error',
 high_test='use actual allowed H±.04 and source1 mH, include near-Nyquist/oriented structure and real A0 white-paint edges',
 low_valid_mass='report rows/errors/source and training-pixel contribution separately; never silently remove fullS or call them accurate',
 scope='engineering accuracy gate only; no hidden-road GT or3DGS upper-limit certificate',
 on_failure='fix integration/query engineering before optimization; preserve old attempt, no visually selected resolution/bounds/extra training')
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(str(p))
