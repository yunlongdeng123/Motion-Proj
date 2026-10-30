from pathlib import Path
import json,numpy as np
r=Path('work/v43_sim_mechanism_20261030/source_control_r1/evidence');h=json.loads((r/'observed16_source_trace.json').read_text());print('TRACE',h['max_category_conservation_relative_error'],h['closure_category_vs_total_max_abs'])
for f in ['000000','000008','000015']:
 p=r/'run_r2_color_maps'/('source_'+f);s=json.loads((p/'source_receipt.json').read_text());print('SOURCE',f,'roi',s['roi_pixels'],'preSta/dyn/rem/ret',*[s[k] for k in ['original_static_body_alphaT','original_dynamic_body_alphaT','removed_static_original_body_alphaT','retained_static_original_body_alphaT']],'post',s['postdelete_body_alphaT'],'gain>5frac',s['postdelete_newly_revealed_body_alphaT']/s['postdelete_body_alphaT'],'mixedfrac',s['postdelete_mixed_tail_body_alphaT']/s['postdelete_body_alphaT'])
 for x in s['cases']:
  if x['case'] in ['baseline_delete','remove_top_error_32','remove_top_error_128','top128_color_only_reference_moment','baseline_delete_env_white','baseline_delete_env_reference']:
   z=x['regions'];print(x['case'],z['original_body_revealed_road']['raw_mae'],z['original_body_revealed_road']['clamped_mae'],z['road_outside_original_body']['clamped_mae'])
 if (p/'per_row_sources.npz').exists():
  a=np.load(p/'per_row_sources.npz');w=a['after_body'];print('recalc_mid',a['before_body'].sum(),a['before_dynamic_body'].sum(),a['before_body'][~a['selected']].sum(),w.sum());print('top8',[(int(i),float(w[i]),float(a['depth_world_z'][i]),float(a['post_depth_rms_reference_m'][i])) for i in s['groups']['top_error_128'][:8]])
print('HISTORY_TOP8')
for x in h['top8']:print(x['row_id'],x['sum_vehicle'],x['sum_road'],x['sum_other'],x['sum_total'],x['frames_total_gt1e4'],x['hdr_view_count'])
