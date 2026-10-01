"""r9覆盖缺额不训练；单独登记r10有限过程最小证据pilot。"""
from pathlib import Path
import sys,copy,shutil
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,O,read,dump
R9=O;O=T/'r10';R8=T/'r8'

def main():
 if (O/'dataset_catalog.json').exists():print('FROZEN: no reselection');return
 O.mkdir(exist_ok=True);manifest=read(R9/'data_review/synthetic_manifest.json')['clips'];qa=read(R9/'independent_data_reviews.json');checks=read(R9/'technical_checks.json');by={r['case_id']:r for r in qa['cases']};tech={r['case_id']:r for r in checks['cases']}
 assert qa['model']=='gpt-6-sol' and qa['reasoning_effort']=='xhigh' and qa['fast'] is False and len(by)==len(manifest)==22
 assert checks['stage']=='complete' and checks['static_scan_check_complete']
 eligible=[c for c in manifest if by[c['case_id']]['decision']=='pass' and by[c['case_id']]['assistant_score']==2 and tech[c['case_id']]['technical_pass']]
 old=read(R8/'dataset_catalog.json');oldtrain=set(read(R8/'source_split.json')['old_r7_training_scenes'])|{c['receiver_scene'] for c in old['cases'] if c['split']=='train'};val_scenes=set(read(R8/'evaluation_source_split.json')['synthetic_validation_scenes'])
 # 全部未进入任何旧训练的新增世界都只做验证，按过程优先固定最多4例。
 priorities={'sweep_B':0,'visibility_transition':1,'static_A_moving_ego':2};newval=[];perfamily=Counter();per_scene=Counter()
 for c in sorted(eligible,key=lambda c:(priorities[c['process_family']],c['case_id'])):
  if c['scene'] in oldtrain or per_scene[c['scene']]>=2:continue
  if c['process_family']=='visibility_transition' and per_scene[c['scene']]:continue
  newval.append(c);per_scene[c['scene']]+=1
  if len(newval)==4:break
 val_scenes|={c['scene'] for c in newval};newtrain=[c for c in eligible if c['scene'] not in val_scenes]
 chosen=[];scenes=Counter()
 for c in sorted(newtrain,key=lambda c:(priorities[c['process_family']],c['case_id'])):
  if scenes[c['scene']]>=2:continue
  chosen.append({'dataset_id':'r10/'+c['case_id'],'case_id':c['case_id'],'folder':c['folder'],'frame_count':10,'type':c['type'],'receiver_scene':c['scene'],'source_id':c['source_id'],'split':'train','assistant_score':2,'technical_pass':True,'independent_QA':by[c['case_id']],'process_family':c['process_family'],'process':c['temporal_process'],'human_verdict':None});scenes[c['scene']]+=1
 # 先每个旧world一例，保留覆盖，然后固定类型优先填满；不按输出筛样。
 candidates=[c for c in old['cases'] if c['split']=='train' and c['receiver_scene'] not in val_scenes]
 for rank in range(3):
  for scene in sorted({c['receiver_scene'] for c in candidates}):
   if scenes[scene]>rank or len(chosen)>=50:continue
   options=[c for c in candidates if c['receiver_scene']==scene and not any(r['folder']==c['folder'] for r in chosen)]
   if not options:continue
   options.sort(key=lambda c:({'dense_actors':0,'single_actor':1,'background':2}[c['type']],c['case_id']))
   c=copy.deepcopy(options[0]);c['dataset_id']='r10_control/'+c['case_id'];c.update(process_family='r8_stable_control',admission_provenance=str(R8/'dataset_catalog.json'));chosen.append(c);scenes[scene]+=1
 sweep=[c for c in chosen if c['process_family']=='sweep_B'];sv=[c for c in newval if c['process_family']=='sweep_B'];summary={'training_cases':len(chosen),'training_scene_count':len(scenes),'max_training_cases_per_scene':max(scenes.values(),default=0),'training_type_counts':dict(Counter(c['type'] for c in chosen)),'training_process_counts':dict(Counter(c['process_family'] for c in chosen)),'sweep_training_worlds':len({c['receiver_scene'] for c in sweep}),'new_process_validation_cases':len(newval),'new_sweep_validation_worlds':len({c['scene'] for c in sv}),'new_sweep_validation_cases':len(sv),'human_verdict':None}
 ready=len(chosen)==50 and len(scenes)>=20 and max(scenes.values())<=3 and len(sweep)>=3 and len({c['receiver_scene'] for c in sweep})>=3 and len({c['scene'] for c in sv})>=1 and all(summary['training_type_counts'].get(t,0)>0 for t in ['background','single_actor','dense_actors'])
 dump(O/'admission_result.json',dict(summary,ready=ready,scope='bounded exploratory pilot, sweep validation only1 world possible; not sufficient generalization'))
 assert ready,summary
 vals=[copy.deepcopy(c) for c in old['cases'] if c['split']=='validation']
 for c in vals:c.update(dataset_id='r10_oldval/'+c['case_id'],process_family='r8_validation_control')
 for c in newval:
  vals.append({'dataset_id':'r10_val/'+c['case_id'],'case_id':c['case_id'],'folder':c['folder'],'frame_count':10,'type':c['type'],'receiver_scene':c['scene'],'source_id':c['source_id'],'split':'validation','assistant_score':2,'technical_pass':True,'independent_QA':by[c['case_id']],'process_family':c['process_family'],'process':c['temporal_process'],'human_verdict':None})
 assert not set(scenes)&{c['receiver_scene'] for c in vals};assert not oldtrain&{c['receiver_scene'] for c in vals}
 summary.update(validation_cases=len(vals),validation_scene_count=len({c['receiver_scene'] for c in vals}),validation_type_counts=dict(Counter(c['type'] for c in vals)))
 dump(O/'dataset_catalog.json',{'cases':chosen+vals,'summary':summary,'architecture_unchanged':True,'GT':'true nuScenes Y, A fully erased before encoder','process_scope':'actual one-second10 exposures; GT cuboid geometric evidence proxy, not exact texture','human_verdict':None})
 dump(R9/'closeout.json',{'stage':'closed_bounded_factory_coverage_shortage','training_steps':0,'coverage_goal_met':False,'sweep_candidates':7,'sweep_candidate_worlds':5,'followup':'separately registeredr10 limited admitted-process pilot; no further r9 grid sweep','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'data process coverage shortage; not model negation','human_verdict':None})
 cases=copy.deepcopy(read(R8/'evaluation_plan.json')['cases'])
 for c in cases:c['suite']='r8_frozen_synthetic' if c['kind']=='synthetic' else 'real_DEVELOPMENT'
 for c in vals:
  if c['process_family']!='r8_validation_control':cases.append(dict(c,eval_id='temporal_'+c['case_id'],kind='synthetic',frames=list(range(10)),GT_available=True,suite='new_temporal_validation'))
 dump(O/'evaluation_plan.json',{'cases':cases,'arms':['base','r7','r8','r10'],'seed':42,'steps':25,'frames':10,'inference_resolution':[576,1024],'same_input_for_every_arm':True,'real_actor_free_GT':None,'final_used':False,'human_verdict':None,'selection_before_new_model_outputs':True,'sweep_validation_world_count':summary['new_sweep_validation_worlds']})
 dump(O/'run.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r10','stage':'frozen_quality_admitted_pilot','host':'wm-3090-1001','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','seed_training':6201,'seed_inference':42,'steps':160,'module_scope':'same80 spatial attention tensors','power_authorization':None,'human_verdict':None})
 print(summary)
if __name__=='__main__':main()
