"""Read existing first-wave metrics; do not re-evaluate or change any student."""
from pathlib import Path
import json,hashlib
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_adaptive_20261121_r1');out=P/'diagnostics/residual_firstwave';out.mkdir(parents=True,exist_ok=True)
metrics=P/'bench/metrics.json';q=json.loads(metrics.read_text());result=[]
def row(stage,raw,source,sigma):return next(x for x in q['rows']if x['id']==stage and x['raw']==raw and x['source']==source and x['scale']==sigma)
for raw in [86,110]:
 for source in ['C','S']:
  parent=row('parent',raw,source,1);final=row('appearance',raw,source,1);g=next(x for x in q['blackgap_rows']if x['id']=='appearance'and x['raw']==raw and x['source']==source);new=[x for x in q['newly_covered_rows']if x['raw']==raw and x['source']==source]
  h=[]
  for sigma in [1,2,4]:
   a=row('parent',raw,source,sigma);b=row('appearance',raw,source,sigma)
   if a['stable_score_n']!=b['stable_score_n']or a['core_n']!=b['core_n']:raise ValueError('common stable denominators differ')
   h.append(dict(scale=sigma,core_n=a['core_n'],parent_HF_cosine=a['HF_cosine'],final_HF_cosine=b['HF_cosine'],parent_HF_energy_ratio=a['HF_energy_ratio'],final_HF_energy_ratio=b['HF_energy_ratio'],parent_HF_RMSE=a['HF_RMSE'],final_HF_RMSE=b['HF_RMSE']))
  result.append(dict(raw=raw,source=source,full2D_domain_n=final['full_source_n'],certified_road_n=g['certified_road_n'],uncertified_editing_n=g['uncertified_editing_n'],parent_full_premult_RGB_MAE=parent['full_premult_RGB_MAE'],final_full_premult_RGB_MAE=final['full_premult_RGB_MAE'],common_stable_n=final['stable_score_n'],common_stable_HF=h,newly_covered_n=final['newly_covered_n'],newly_covered_final_HF=[{k:x[k]for k in ['scale','newly_covered_n','newly_covered_certified_n','newly_covered_uncertified_editing_n','core_n','RGB_MAE','RGB_RMSE','HF_cosine','HF_energy_ratio','HF_RMSE']}for x in new],certified_final_no_support_lt005_n=g['certified_road_no_support_n'],certified_final_thin_005_to095_n=g['certified_road_low_alpha_n'],certified_final_low_alpha_lt095_n=g['certified_road_low_alpha_total_n']))
d=dict(schema='adaptive_firstwave_four_rows.v1',rows=result,source_file=str(metrics),source_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),source_keys='rows[id,parent|appearance,raw,source,scale]; newly_covered_rows[raw,source,scale]; blackgap_rows[id=appearance,raw,source]',normalization='full P vs opaqueT on identical complete2D C/S editing denominators; parent/final HF on identical all-stage common-stable domain; newly-covered HF final-only',new_GPU_calls=0,new_training_updates=0,metrics_recomputed=False)
(out/'four_rows.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
