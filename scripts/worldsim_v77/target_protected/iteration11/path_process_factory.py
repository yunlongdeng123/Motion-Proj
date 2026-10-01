"""只在既有合法虚拟A路径上改变相对速度，所有当前世界约束重查。"""
from pathlib import Path
import os,sys,math,copy,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P/'iteration9'));sys.path.insert(0,str(P/'iteration11'))
from temporal_factory import T,R8,read,dump,geometry,protections,exact,old
from geometry_factory import footprint,projection,wrap
from temporal_metrics import process
from pyquaternion import Quaternion
from collections import Counter,defaultdict
import numpy as np,cv2
O=T/'r15'

def pose(p,mid,delta):
 a=copy.deepcopy(p['actor']);R=Quaternion(a['rotation']).rotation_matrix
 a['translation']=(np.asarray(a['translation'])+R[:,0]*delta*((p['timestamp']-mid)/1e6)).tolist()
 return a

def endpoint_span(c,anchor,t,delta):
 vals=[];mid=anchor['frames'][5]['timestamp']
 for i in [0,9]:
  p=anchor['frames'][i];pr=projection(pose(p,mid,delta),c['frames'][i]);b=next(a for a in c['frames'][i]['actors'] if a['instance_token']==t);bb=np.array(b['projection']['box_xyxy'])
  if pr is None:return None
  vals.append((np.mean(pr['box'][[0,2]])-bb[0])/(bb[2]-bb[0]))
 return vals[-1]-vals[0]

def fitted(c,anchor,t):
 z=endpoint_span(c,anchor,t,0);one=endpoint_span(c,anchor,t,1)
 if z is None or one is None or abs(one-z)<.015:return []
 return sorted(set(round((goal-z)/(one-z),3) for goal in [-.4,.4] if .25<=abs((goal-z)/(one-z))<=8))

def trajectory(geo,c,anchor,delta):
 sid=anchor['source_id'];mid=anchor['frames'][5]['timestamp'];plane=np.array(geo.ground[sid]['plane']);rows=[];prev=None;min_gap=1e9;max_support=0
 for p,f,obs in zip(anchor['frames'],c['frames'],geo.obstacles[sid]):
  a=pose(p,mid,delta);R=Quaternion(a['rotation']).rotation_matrix
  # 原路线yaw保持；重新贴原平面，不能把沿坡轴的位移当高度跳变。
  xy=np.asarray(a['translation'])[:2];a['translation']=(np.r_[xy,np.dot(np.r_[xy,1],plane)]+R[:,2]*a['size'][2]/2).tolist()
  foot=footprint(a)
  if not geo.road.covers(foot):return None,'outside_mapped_drivable_or_parking'
  support=float(geo.ground[sid]['_tree'].query(np.array(foot.exterior.coords)[:4])[0].max());max_support=max(max_support,support)
  if support>2.5:return None,'ground_support_gap'
  pr=projection(a,f)
  if pr is None:return None,'behind_camera'
  bb=pr['box'];wh=bb[2:]-bb[:2]
  if min(bb[0],bb[1],1024-bb[2],576-bb[3])<8 or wh[0]<72 or wh[1]<40 or not .006<=np.prod(wh)/(1024*576)<=.18:return None,'size_border'
  if bb[3]+6>=512:return None,'ego_band'
  for ob in obs:
   gap=foot.distance(ob['_foot']);min_gap=min(min_gap,gap)
   if gap<.3:return None,'collision_clearance'
  center=np.asarray(a['translation']);yaw=math.atan2(R[1,0],R[0,0])
  if prev:
   dt=(f['timestamp']-prev['timestamp'])/1e6;velocity=(center-prev['center'])/dt;vxy=velocity[:2];speed=np.linalg.norm(vxy)
   if np.linalg.norm(velocity)*.1>2 or abs(wrap(math.degrees(yaw-prev['yaw'])))*.1/dt>5:return None,'trajectory_continuity'
   ratio=(wh/prev['wh'])**(.1/dt)
   if ratio.min()<.85 or ratio.max()>1.18:return None,'scale_continuity'
   # 新增方向检查：不把侧滑或高速倒车当合理移动A。
   along=float(vxy@R[:2,0]);cross=abs(float(vxy@R[:2,1]))
   if along<-.5 or (speed>.5 and math.degrees(math.atan2(cross,max(along,1e-6)))>25):return None,'velocity_yaw_inconsistent'
  rows.append({'frame':f['frame'],'timestamp':f['timestamp'],'actor':a,'box':bb.tolist(),'depth_interval':[pr['near_depth'],pr['far_depth']]});prev={'timestamp':f['timestamp'],'center':center,'yaw':yaw,'wh':wh}
 return dict(anchor,frames=rows,min_GT_clearance_m=min_gap,max_ground_support_distance_m=max_support,relative_speed_mps=delta,trajectory_policy='existing empty-space virtual path plus fitted longitudinal relative velocity'),None

def main(shard,count):
 cv2.setNumThreads(1);geo=geometry();frozen=read(O/'source_selection.json');pool=read(R8/'all_asset_candidates.json')['candidates'];by=defaultdict(list)
 for p in pool:by[p['source_id']].append(p)
 assets={name:dict(np.load(R8/'assets'/f'{name}.npz')) for name in ['sedan','suv']};out=O/'planned';out.mkdir(exist_ok=True)
 for row in frozen['sources'][shard::count]:
  sid=row['source_id'];dest=out/(sid+'.json')
  if dest.exists():continue
  start=time.time();g=geo.prepare(sid);pm=protections(geo,sid);c=geo.sources[sid];rej=Counter();chosen=defaultdict(list)
  anchors=sorted(by[sid],key=lambda p:(p['type']=='background',p['asset'],p['offset_longitudinal_m'],p['offset_lateral_m']))[:4]
  if g['pass'] and pm:
   for anchor in anchors:
    targets=anchor['protected_instances'] or [c['actors'][0]['instance_token']]
    for tok in targets[:1]:
     if tok not in pm:rej['protected_mask_unavailable']+=1;continue
     for delta in fitted(c,anchor,tok):
      tr,why=trajectory(geo,c,anchor,delta)
      if tr is None:rej[why]+=1;continue
      mesh=assets[anchor['asset']];aa=[old.silhouette(mesh['vertices'],mesh['faces'],p['actor'],f) for p,f in zip(tr['frames'],c['frames'])];q,why=exact(geo,tr,aa,pm)
      if q is None:rej[why]+=1;continue
      active={t:pm[t] for t in q['protected_instances']};ba={t:[next(a for a in f['actors'] if a['instance_token']==t) for f in c['frames']] for t in active};hs=[cv2.dilate(a.astype('uint8'),np.ones((7,7),'uint8'))>0 for a in aa];proc=process(c['frames'],[p['actor'] for p in tr['frames']],hs,active,ba)
      family='sweep_B' if proc['sweep_over_any_B'] else 'visibility_transition' if proc['visibility_transition_any_B'] else 'static_A_moving_ego' if proc['static_A_moving_ego_image_change'] else None
      if family is None:rej['no_actual_temporal_process']+=1;continue
      if len(chosen[family])>=2:continue
      chosen[family].append(tr|q|{'process_family':family,'temporal_process':proc,'edge_mode':'near_hard','mask_dilation_px':3,'human_verdict':None,'source_split':row['split'],'endpoint_targets':[-.4,.4],'fit_token':tok,'fit_uses_model_output':False})
  rows=[p for vv in chosen.values() for p in vv];dump(dest,{'source':row,'candidates':rows,'counts':dict(Counter(p['process_family'] for p in rows)),'rejects':dict(rej),'seconds':time.time()-start});print(sid,row['split'],len(rows),dict(Counter(p['process_family'] for p in rows)),flush=True)
 dump(O/f'shard_{shard}.json',{'stage':'complete','shard':shard,'sources':len(frozen['sources'][shard::count]),'human_verdict':None})

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--count',type=int,default=1);a=p.parse_args();main(a.shard,a.count)
