from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,read,dump
O=T/'r10'
def mean(x):return float(np.mean(x)) if x else None
def aggregate(rows,arms,key):
 out={'cases':len(rows),'scenes':len({r['scene'] for r in rows}),'macro_case':{},'macro_scene':{}}
 for a in arms:
  by=defaultdict(list)
  for r in rows:
   v=r['arms'][a][key]
   if v is not None:by[r['scene']].append(v)
  out['macro_case'][a]=mean([v for ar in by.values() for v in ar]);out['macro_scene'][a]=mean([mean(v) for v in by.values()])
 for basis in ['base','r7','r8']:
  valid=[r for r in rows if r['arms'][basis][key] is not None and r['arms']['r10'][key] is not None]
  out['r10_vs_'+basis]={'better_cases':sum(r['arms']['r10'][key]<r['arms'][basis][key] for r in valid),'total_cases':len(valid),'scene_relative_change':out['macro_scene']['r10']/out['macro_scene'][basis]-1 if out['macro_scene'][basis] else None}
 return out
def main():
 plan=read(O/'evaluation_plan.json');state=read(O/'evaluation/state.json');assert state['all_four_arms_complete'];rows=[]
 for c in plan['cases']:
  if c['kind']!='synthetic':continue
  r={'eval_id':c['eval_id'],'scene':c['receiver_scene'],'suite':c['suite'],'type':c['type'],'process_family':c.get('process_family','r8_validation_control'),'arms':{}}
  for a in plan['arms']:
   scores=read(O/'evaluation'/c['eval_id']/(a+'_metrics.json'))['scores'];r['arms'][a]={}
   for key in ['hole','protected_inside_hole','native_outside_hole']:r['arms'][a][key]=mean([s[key]['MAE'] for s in scores if s.get(key) is not None])
  rows.append(r)
 groups={}
 for suite in sorted({r['suite'] for r in rows}):
  rr=[r for r in rows if r['suite']==suite];groups[suite]={key:aggregate([r for r in rr if key!='protected_inside_hole' or r['type']!='background'],plan['arms'],key) for key in ['hole','protected_inside_hole']}
 process_groups={}
 for family in sorted({r['process_family'] for r in rows}):
  rr=[r for r in rows if r['process_family']==family];process_groups[family]={key:aggregate([r for r in rr if key!='protected_inside_hole' or r['type']!='background'],plan['arms'],key) for key in ['hole','protected_inside_hole']}
 result={'task':'WS-V77-TARGET-PROTECTED-20260929','run':'r10','data':read(O/'dataset_catalog.json')['summary'],'training':read(O/'training/state.json'),'same_recipe':read(O/'training/same_recipe_as_r7.json'),'groups':groups,'process_groups':process_groups,'synthetic_cases':rows,'all_inference_complete':True,'reused_windows':sum(r.get('reused',False) for r in state['completed']),'fresh_windows':sum(not r.get('reused',False) for r in state['completed']),'real_case_count':sum(c['kind']=='real_development' for c in plan['cases']),'real_actor_free_GT':None,'human_verdict':None,'final_used':False,'scope':'limited1-second process pilot; new sweep val only one independent world. Old/new group not pooled to hide type results; pixel MAE not identity/temporal/video verdict'}
 dump(O/'results_summary.json',result);print(groups);print(process_groups)
if __name__=='__main__':main()
