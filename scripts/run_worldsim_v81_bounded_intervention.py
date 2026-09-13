"""单轮三日志真实有效观测干预；两模型独立进程；恢复使用同一真实观测。"""
import os,json,sys,argparse,datetime
from pathlib import Path
O=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01');A=O/'av2_atlas'
def freeze():
 src=json.loads((A/'interventions_before_model.json').read_text());chosen=[]
 # 双侧设计在RGB审核时失败：前一帧出画或被车挡；不根据任何模型输出改动。
 for r in src['selected']:
  g=r['geometry'][2]
  if g['visible_frustum_and_sparse_occlusion_fraction']<.7 or (g['median_parallax_deg'] or 0)<1:continue
  m=json.loads(Path(r['manifest']).read_text());future=max(m['context_views'],key=lambda v:v['timestamp_us']);variants={}
  for v in ['rich','removed','restored']:
   dest=A/'input_manifests'/f'{r["roi_id"]}_{v}.json';dest.write_text(json.dumps({**m,'views':m['views']+([future] if v!='removed' else []),'context_views':[],'role':'VIEW_DIAGNOSTIC','variant':v},indent=2));variants[v]=str(dest)
  chosen.append({**r,'variants':variants,'effective_evidence_rgb_review':'future warp shows same visible wall/column; previous warp rejected before models','scale_fit':'identical current INPUT LiDAR target-camera projection IDs outside ROI+16px for every variant'})
 out={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_outputs_accessed':False,'amendment':'two-sided context is not usable; retain registered logs/targets, freeze one genuinely visible +0.5s observation before all new forwards; previous frame not used','comparison':'rich=[target,future]; removed=[target]; restored=[target,same future]; identical seed and preprocessing','main_effect':'removed minus average(rich,restored) under same-input-LiDAR calibration algorithm; normal and shape also separate; recovery equality is deterministic restoration not independent replication','practical_effect':'depth delta >0.25m with sparse residual above frozen threshold, or normal delta >5deg with sparse normal>10deg; same axis on all 3 discovery logs is repeatable discovery trend; no texture causal claim','selected':chosen,'jobs':len(chosen)*3*2,'human_verdict':None}
 assert not (A/'final_intervention_protocol.json').exists();(A/'final_intervention_protocol.json').write_text(json.dumps(out,indent=2));print(json.dumps({'frozen_logs':len(chosen),'jobs':out['jobs']}))
def run(method):
 from run_worldsim_v81_inference import main
 protocol=json.loads((A/'final_intervention_protocol.json').read_text());events=[]
 for r in protocol['selected']:
  for variant,path in r['variants'].items():
   dest=O/'new_predictions'/method/r['roi_id']/variant
   if (dest/'result.json').exists():raise FileExistsError(dest)
   main(['--method',method,'--manifest',path,'--variant','full6','--out',str(dest),'--allow-relative-single-view','--execute'])
   record=json.loads((dest/'result.json').read_text());record.update(variant=variant,role='VIEW_DIAGNOSTIC',dataset='AV2',roi_id=r['roi_id'],input_protocol=protocol['comparison']);(dest/'result.json').write_text(json.dumps(record,indent=2));events.append({'roi_id':r['roi_id'],'variant':variant,'result':str(dest/'result.json')});(O/f'new_{method}_progress.json').write_text(json.dumps(events,indent=2))
 (O/f'new_{method}_done.json').write_text(json.dumps({'status':'DONE','jobs':len(events),'method':method}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--method',choices=['dvgt','vggt']);a=p.parse_args()
 if a.freeze:freeze()
 else:run(a.method)
