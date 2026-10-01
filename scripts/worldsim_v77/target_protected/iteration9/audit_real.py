"""原70例实际世界运动；邻车遮挡仅GT包络代理，隐藏纹理证据保持unknown。"""
from pathlib import Path
import sys
from collections import defaultdict
import ijson,numpy as np,cv2
from PIL import Image
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77');sys.path.insert(0,str(P/'delete_audit'));sys.path.insert(0,str(P/'target_protected'))
from prepare_clips import target_at,matrix,polygon
from geometry_factory import read,dump
from temporal_metrics import process
from audit_temporal import summary,T,O

def filtered(p,key,allowed):
 out={}
 with p.open('rb') as f:
  for r in ijson.items(f,'item',use_float=True):
   if r[key] in allowed:out[r['token']]=r
 return out

def main():
 root=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1');meta=root/'metadata/v1.0-trainval'
 expected=read(root/'expected_sources.json')['clips'];sel={c['clip_id']:c for c in read(root/'selection.json')['clips']}
 samples={r['token']:r for r in read(meta/'sample.json')};ks={k for c in expected for k in c['keyframe_sample_tokens']}
 anns=filtered(meta/'sample_annotation.json','sample_token',ks);by=defaultdict(list)
 for r in anns.values():by[r['instance_token']].append(dict(r,timestamp=samples[r['sample_token']]['timestamp']))
 for rs in by.values():rs.sort(key=lambda r:r['timestamp'])
 sd=filtered(meta/'sample_data.json','token',{f['sample_data_token'] for c in expected for f in c['frames']})
 ego=filtered(meta/'ego_pose.json','token',{r['ego_pose_token'] for r in sd.values()});cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
 rows=[];cache={}
 for c in expected:
  cid=c['clip_id'];spec=sel[cid];target=spec['instance_token'];frames=[];poses=[];hp=[];neighbors=defaultdict(list)
  tokens={r['instance_token'] for r in anns.values() if r['sample_token'] in c['keyframe_sample_tokens'] and r['instance_token']!=target}
  for f in c['frames']:
   d=sd[f['sample_data_token']];e=ego[d['ego_pose_token']];ca=cal[d['calibrated_sensor_token']];C=matrix(e['translation'],e['rotation'])@matrix(ca['translation'],ca['rotation']);K=np.array(ca['camera_intrinsic']);K[0]*=1024/d['width'];K[1]*=576/d['height'];t=f['source_timestamp_us']
   frame={'timestamp':t,'camera_to_world':C.tolist(),'intrinsics_1024':K.tolist()};a,kind=target_at(t,by[target])
   if a is None:break
   frames.append(frame);poses.append(a);hp.append(np.asarray(Image.open(root/'clips'/cid/'model_mask'/f"{f['frame']:05}.png"))>0)
   for tok in tokens:
    b,_=target_at(t,by[tok]);mask=np.zeros((576,1024),'uint8')
    if b is not None:
     shape=polygon(b,{'w2c':np.linalg.inv(C),'k':K})
     # 只有完整B区间比A更远才作为behind包络；不是像素实例认证。
     ar=np.array(a['translation']);br=np.array(b['translation']);ad=(np.linalg.inv(C)@np.r_[ar,1])[2];bd=(np.linalg.inv(C)@np.r_[br,1])[2]
     if shape and bd-ad>1:cv2.fillConvexPoly(mask,np.array(shape['hull'],int),1)
    neighbors[tok].append(mask>0)
  if len(frames)!=len(c['frames']):rows.append({'case_id':cid,'geometry_complete':False,'reason':'target track unavailable'});continue
  # 不知道实际B像素，禁止把GT包络对应当真实输入可见证据。
  pm={t:m for t,m in neighbors.items() if len(m)==len(frames) and any((mm&hh).sum()>20 for mm,hh in zip(m,hp))}
  r={'case_id':cid,'scene':spec['scene'],'camera':spec['camera'],'geometry_complete':True,'input_difficulty_proxy':spec['input_difficulty_proxy'],**process(frames,poses,hp,pm,None)}
  r['protected_semantics']='behind GT envelope proxy only; original pixel visibility/identity and temporal texture evidence UNKNOWN';r['evidence_semantics']='UNKNOWN real hidden RGB; no actor-free GT, no SAM2 per-neighbor identity';r['GT_envelope_sweep_proxy']=r.pop('sweep_over_any_B');r['GT_visibility_transition_proxy']=r.pop('visibility_transition_any_B');rows.append(r);cache[cid]=(frames,poses,hp,pm)
 named=[]
 for c in read(T/'r8/real_evaluation_plan.json')['cases']:
  fs,aa,hh,pm=cache[c['clip_id']];ids=c['frames'];r=process([fs[i] for i in ids],[aa[i] for i in ids],[hh[i] for i in ids],{t:[m[i] for i in ids] for t,m in pm.items()},None)
  r['GT_envelope_sweep_proxy']=r.pop('sweep_over_any_B');r['GT_visibility_transition_proxy']=r.pop('visibility_transition_any_B');named.append({'eval_id':c['eval_id'],'scene':c['scene'],'camera':c['camera'],'frames_selected':ids,**r,'evidence_semantics':'UNKNOWN real hidden appearance; GT envelope temporal proxy only'})
 out={'stage':'complete','clips':rows,'named_fixed_eval_windows':named,'summary':{'clips':len(rows),'static_A_moving_ego':sum(r.get('static_A_moving_ego',False) for r in rows),'static_A_moving_ego_image_change':sum(r.get('static_A_moving_ego_image_change',False) for r in rows),'GT_envelope_sweep_proxy':sum(r.get('GT_envelope_sweep_proxy',False) for r in rows)},'human_verdict':None,'limits':'GT track interpolation + actual camera poses/model masks; no GT hull means observed pixel segmentation; evidence and hidden actor identity UNKNOWN'}
 dump(O/'real_temporal_audit.json',out);print(out['summary']);print('NAMED',[(r['eval_id'],round(r['ego_camera_motion']['path_m'],2),round(r['A_world_motion']['diameter_m'],2),r['GT_envelope_sweep_proxy']) for r in named])
if __name__=='__main__':main()
