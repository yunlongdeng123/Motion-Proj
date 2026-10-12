import json
from pathlib import Path
from statistics import mean
HERE=Path(__file__).resolve().parent
rows=[]
for path in sorted((HERE/'component_diagnostic/cases').glob('*/case_manifest.json')):
    case=json.loads(path.read_text())
    frames=[]
    for f in case['frames']:
        full=f['regions']['full']['metrics']
        outside=f['regions']['non_sky_outside_actor']['metrics']
        frames.append({'frame':f['frame'],
            'composite_psnr_full':full['composite']['clamped_rgb_psnr_db'],
            'static_psnr_observed_non_sky_outside_actor':outside['static']['clamped_rgb_psnr_db'],
            'alpha_dynamic_observed_non_sky_outside_actor':outside['alpha_d']['mean'],
            'alpha_static_observed_non_sky_outside_actor':outside['alpha_s']['mean'],
            'static_environment_contribution_mean_observed_non_sky_outside_actor':outside['static_env_contribution']['mean'],
            'dynamic_rgb_support_observed_non_sky_outside_actor':outside['dynamic']['rgb_support_gt2_uint8_pixel_fraction']})
    rows.append({'case_id':case['case_id'],'frames':frames,'means':{k:mean(f[k] for f in frames) for k in frames[0] if k!='frame'}})
report={'status':'completed_component_diagnostic_only','selected_frame_count':5,'horizon':50000,
        'scope':'Means over same5 selected cameras; no edit-region GT claimed. Mixed components are inference diagnostics only.',
        'cases':rows}
(HERE/'component_diagnostic/compact_case_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
for row in rows:print(row['case_id'],{k:round(v,5) for k,v in row['means'].items()})
