import pathlib,json,numpy as np,collections,tarfile
from geometry_contract import R
rows=[];summaries=[]
for p in sorted(list((R/'evaluation_bgscale').glob('*/*/*/summary.json'))+list((R/'evaluation_common6').glob('*/*/*/summary.json'))):
 s=json.loads(p.read_text());summaries.append(s)
 for r in s['rows']:
  if r['status']!='EVALUATED':continue
  rows.append({k:s[k] for k in ['method','scene','log','variant']}|r)
groups=collections.defaultdict(list)
for r in rows:groups[(r['method'],r['variant'],r['protocol'])].append(r)
table=[]
for (method,variant,protocol),rs in groups.items():
 total={k:sum(r['counts'][k] for r in rs) for k in rs[0]['counts']};logrates=[]
 for log in sorted(set(r['log'] for r in rs)):
  rr=[r for r in rs if r['log']==log];n=sum(r['counts']['rays'] for r in rr)
  logrates.append({'log':log,'rays':n,**{k:sum(r['counts'][k] for r in rr)/n for k in ['hit','early','near_vertices','intrusion_sum_m','early_with_later_correct','miss']}})
 static=[r for r in rs if r['static'] is True];ss={k:sum(r['strata']['build_supported'][k] for r in static) for k in ['rays','early','hit','miss','near_vertices']}
 rec={'method':method,'variant':variant,'protocol':protocol,'actors':len(rs),'counts':total,'static_supported':ss,'logs':logrates,'equal_log_mean':{k:float(np.mean([x[k] for x in logrates])) for k in ['hit','early','near_vertices','intrusion_sum_m','early_with_later_correct','miss']}}
 table.append(rec)
 if protocol=='cal_build_1.0':print(json.dumps(rec),flush=True)
completed=len(list((R/'predictions').glob('*/*/*/result.json')))
out={'status':'FOUR_MODELS_COMPLETE' if completed==48 and len(summaries)==72 else 'PARTIAL_OMEGA_OR_CONTROLS_PENDING','summaries':len(summaries),'model_results':completed,'table':table,'per_actor':rows}
(R/'aggregate.json').write_text(json.dumps(out,indent=2))
with tarfile.open(R/'summary_bundle.tar','w') as t:
 for name in ['registration.json','cohort.json','evaluation_amendment.json','common6_registration.json','aggregate.json','queue_status.json','omega_status.json','omega512_registration.json','readout_qa.json']:
  if (R/name).exists():t.add(R/name,arcname='evidence/'+name)
 for s in (R/'evaluation_bgscale').glob('*/*/*/summary.json'):t.add(s,arcname='evaluation_summaries/'+str(s.relative_to(R/'evaluation_bgscale')))
 for s in (R/'evaluation_common6').glob('*/*/*/summary.json'):t.add(s,arcname='evaluation_summaries/'+str(s.relative_to(R/'evaluation_common6')))
 for s in (R/'predictions').glob('*/*/*/result.json'):t.add(s,arcname='inference_results/'+str(s.relative_to(R/'predictions')))
print('bundle',R/'summary_bundle.tar')
