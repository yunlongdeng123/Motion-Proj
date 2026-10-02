"""实测模型可见任务去重与过程覆盖；不按合成RGB风格计多样性。"""
from pathlib import Path
import sys,os
from collections import defaultdict,Counter
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
S=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(S/'iteration9'));sys.path.insert(0,str(S/'iteration11'))
from temporal_factory import T,read,dump
from temporal_metrics import process
from summarize_scope import aggregate
from PIL import Image
import numpy as np
O=T/'r19'

def main():
 catalog=read(T/'r14/dataset_catalog.json')['cases'];source={c['source_id']:c for c in read(T/'r8/factory/source_manifest.json')['clips']};groups=defaultdict(list);rows=[]
 for c in catalog:
  folder=Path(c['folder']);p=read(folder/'pair_manifest.json');src=source[p['source_id']];h=np.stack([np.asarray(Image.open(folder/'model_hole'/f'{i:03}.png'))>0 for i in range(10)])
  groups[tuple(f['filename'] for f in src['frames'])].append((c,h,p))
  pm={tok:[np.asarray(Image.open(folder/'protected'/f'{i:03}_{tok}.png'))>0 for i in range(10)] for tok in p['protected_instances']}
  ba={tok:[next(a for a in f['actors'] if a['instance_token']==tok) for f in src['frames']] for tok in pm}
  proc=process(src['frames'],[f['actor'] for f in p['frames']],h,pm,ba)
  rows.append({'dataset_id':c['dataset_id'],'scene':c['receiver_scene'],'split':c['split'],'type':c['type'],'process':proc})
 dup=[];drop=set();roots={}
 for rgb_files,vv in groups.items():
  reps=[]
  for c,h,p in sorted(vv,key=lambda x:x[0]['dataset_id']):
   match=next((a for a,ah in reps if np.array_equal(h,ah)),None)
   if match:
    dup.append({'keep':match['dataset_id'],'duplicate':c['dataset_id'],'split_keep':match['split'],'split_duplicate':c['split'],'same_real_RGB_files':list(rgb_files),'same_H_all_10':True,'same_masked_condition_all_10':True,'same_Y_all_10':True,'scope':'X差异若完全被H擦除，不构成新训练任务；保护标签差异另看，不能静默合并监督'})
    drop.add(c['dataset_id']);roots[c['dataset_id']]=match['dataset_id']
   else:reps.append((c,h))
 assert not any(r['split_keep']!=r['split_duplicate'] for r in dup),'同任务跨train/val泄漏'
 counts={}
 for split in ['train','validation']:
  rr=[r for r in rows if r['split']==split];pp=[r['process'] for r in rr];pb=[b for r in pp for b in r['protected'].values()]
  counts[split]={'cases':len(rr),'unique_model_visible_tasks':sum(r['dataset_id'] not in drop for r in rr),'worlds':len(set(r['scene'] for r in rr)),'sweep_cases':sum(p['sweep_over_any_B'] for p in pp),'sweep_worlds':len({r['scene'] for r in rr if r['process']['sweep_over_any_B']}),'static_A_moving_ego_cases':sum(p['static_A_moving_ego_image_change'] for p in pp),'visibility_transition_cases':sum(p['visibility_transition_any_B'] for p in pp),'protected_instance_cases':len(pb),'B_other_frame_cuboid_support_mean':float(np.mean([b['approx_other_frame_support_mean'] for b in pb if b['approx_other_frame_support_mean'] is not None])) if pb else None,'B_no_other_frame_proxy_support_cases':sum(b['no_geometric_other_frame_support'] for b in pb),'duration_min_max_s':[min(p['duration_s'] for p in pp),max(p['duration_s'] for p in pp)]}
 # 旧验证计划保留r8 ID，当前catalog有r10_oldval别名；按真实目录关联，不按别名前缀关联。
 results=read(T/'r14/results_summary.json');plan=read(T/'r14/evaluation_plan.json');byfolder={str(Path(c['folder']).resolve()):c['dataset_id'] for c in catalog};byid={c['eval_id']:byfolder[str(Path(c['folder']).resolve())] for c in plan['cases'] if c['kind']=='synthetic'};unique_rows=[r for r in results['synthetic_cases'] if byid[r['eval_id']] not in drop];corrected={}
 assert len(unique_rows)==len(results['synthetic_cases'])-sum(r['split_duplicate']=='validation' for r in dup)
 for suite in sorted({r['suite'] for r in unique_rows}):
  rr=[r for r in unique_rows if r['suite']==suite]
  corrected[suite]={key:aggregate([r for r in rr if key!='protected_inside_hole' or r['type']!='background'],plan['arms'],key) for key in ['hole','protected_inside_hole']}
 result={'summary':counts,'duplicate_tasks':dup,'cross_train_val_duplicates':0,'unique_GT_eval_cases':len(unique_rows),'original_GT_eval_cases':len(results['synthetic_cases']),'original_results_unchanged':True,'deduplicated_groups':corrected,'process_source':'当前实际磁盘H和完整GT A/B位姿；cuboid对应只是近似几何证据，不是纹理GT','cases':rows,'training_steps':0,'GPU_forwards':0}
 dump(O/'process_and_duplicates.json',result);print('PROCESS_DUPLICATES',counts,'duplicate_count',len(dup),'unique_GT_eval',len(unique_rows),flush=True)
if __name__=='__main__':main()
