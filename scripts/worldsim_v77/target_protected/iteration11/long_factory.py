"""同空间锚点扩至三秒；有限相对速度及静止对照，全窗口原质量门槛。"""
from pathlib import Path
import os,sys,copy,math,time
from collections import Counter,defaultdict
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(P/'iteration9'));sys.path.insert(0,str(P/'iteration11'))
from temporal_factory import T,R8,F,old,read,dump,exact
from geometry_factory import footprint,ground_orientation,projection,wrap,source_masks,transform
from iteration5.planning import normalized_masks
from temporal_metrics import process
from path_process_factory import pose
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
import numpy as np,cv2
O=T/'r16';ROOT=O/'factory'

def geometry():
 previous=old.O;old.O=O
 try:g=old.ParkingGeometry(ROOT)
 finally:old.O=previous
 return g

def protections(g,sid):
 c=g.sources[sid];out={}
 for j,a in enumerate(c['actors']):
  tok=a['instance_token'];jid=sid if j==0 else sid+'_'+tok[:8];dest=ROOT/('segmented' if j==0 else 'segmented_secondary')/jid
  if not (dest/'mask_manifest.json').exists():continue
  rec=read(dest/'mask_manifest.json')
  if not rec['technical_temporal_mask_pass']:continue
  mm=source_masks(ROOT,sid,None if j==0 else jid);assert len(mm)==30
  out[tok]=mm
 return out

def support(g,sid):
 # r12明确修正：原拟合地面不变，原始近地返回补支持，不能证明无障碍。
 cache=O/'ground_support_recovery';cache.mkdir(exist_ok=True);dest=cache/(sid+'.npz');gg=g.ground[sid]
 if dest.exists():points=np.load(dest)['points']
 else:
  plane=np.asarray(gg['plane']);loc=np.array([f['actors'][0]['translation'] for f in g.sources[sid]['frames']]);lo=loc[:,:2].min(0)-16;hi=loc[:,:2].max(0)+16;cloud=[]
  for i in [0,3,6]:
   d=g.context[sid]['frames'][i]['sensors']['LIDAR_TOP'];p=np.fromfile(ROOT/'rgb'/d['filename'],np.float32).reshape(-1,5)[:,:3];ca=d['calibrated_sensor'];e=d['ego_pose'];m=transform(e['translation'],e['rotation'])@transform(ca['translation'],ca['rotation']);p=p@m[:3,:3].T+m[:3,3];height=p[:,2]-np.c_[p[:,:2],np.ones(len(p))]@plane;cloud.append(p[(abs(height)<=.08)&np.all((p[:,:2]>=lo)&(p[:,:2]<=hi),axis=1)])
  points=np.concatenate(cloud);np.savez_compressed(dest,points=points)
 assert len(points)>0;gg['original_ground_support_points']=gg['_tree'].n;gg['_tree']=cKDTree(points[:,:2]);gg['support_recovery']={'near_plane_band_m':.08,'support_limit_unchanged_m':2.5,'plane_unchanged':True,'raw_support_points':len(points)}

def extended(g,c,anchor,original):
 mid=anchor['frames'][5];b=original['frames'][5]['actors'][0];offset=np.array(mid['actor']['translation'])[:2]-np.array(b['translation'])[:2];plane=np.asarray(g.ground[c['source_id']]['plane']);rows=[];size=mid['actor']['size']
 for f in c['frames']:
  base=f['actors'][0];r=Quaternion(base['rotation']).rotation_matrix;R=ground_orientation(math.atan2(r[1,0],r[0,0]),plane);xy=np.array(base['translation'])[:2]+offset;a={'translation':(np.r_[xy,np.dot(np.r_[xy,1],plane)]+R[:,2]*size[2]/2).tolist(),'rotation':Quaternion(matrix=R).elements.tolist(),'size':size};rows.append({'frame':f['frame'],'timestamp':f['timestamp'],'actor':a})
 return dict(anchor,frames=rows,scene=c['scene'])

def fitted(c,anchor,tok):
 def span(delta):
  v=[];mid=anchor['frames'][15]['timestamp']
  for i in [0,29]:
   pr=projection(pose(anchor['frames'][i],mid,delta),c['frames'][i]);b=next(a for a in c['frames'][i]['actors'] if a['instance_token']==tok);bb=np.array(b['projection']['box_xyxy'])
   if pr is None:return None
   v.append((np.mean(pr['box'][[0,2]])-bb[0])/(bb[2]-bb[0]))
  return v[-1]-v[0]
 z=span(0);one=span(1)
 if z is None or one is None or abs(one-z)<.015:return []
 return sorted(set(round((goal-z)/(one-z),3) for goal in [-.4,.4] if .25<=abs((goal-z)/(one-z))<=8))

def trajectory(g,c,anchor,delta,static=False):
 sid=c['source_id'];mid=anchor['frames'][15]['timestamp'];plane=np.array(g.ground[sid]['plane']);rows=[];prev=None;min_gap=1e9;max_support=0
 for p,f,obs in zip(anchor['frames'],c['frames'],g.obstacles[sid]):
  a=copy.deepcopy(anchor['frames'][15]['actor']) if static else pose(p,mid,delta);R=Quaternion(a['rotation']).rotation_matrix;xy=np.asarray(a['translation'])[:2];a['translation']=(np.r_[xy,np.dot(np.r_[xy,1],plane)]+R[:,2]*a['size'][2]/2).tolist();foot=footprint(a)
  if not g.road_for(sid).covers(foot):return None,'outside_mapped_drivable_or_parking'
  sup=float(g.ground[sid]['_tree'].query(np.array(foot.exterior.coords)[:4])[0].max());max_support=max(max_support,sup)
  if sup>2.5:return None,'ground_support_gap'
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
   dt=(f['timestamp']-prev['timestamp'])/1e6;v=(center-prev['center'])/dt;speed=np.linalg.norm(v[:2]);ratio=(wh/prev['wh'])**(.1/dt)
   if np.linalg.norm(v)*.1>2 or abs(wrap(math.degrees(yaw-prev['yaw'])))*.1/dt>5 or ratio.min()<.85 or ratio.max()>1.18:return None,'trajectory_continuity'
   along=float(v[:2]@R[:2,0]);cross=abs(float(v[:2]@R[:2,1]))
   if along<-.5 or (speed>.5 and math.degrees(math.atan2(cross,max(along,1e-6)))>25):return None,'velocity_yaw_inconsistent'
  rows.append({'frame':f['frame'],'timestamp':f['timestamp'],'actor':a,'box':bb.tolist(),'depth_interval':[pr['near_depth'],pr['far_depth']]});prev={'timestamp':f['timestamp'],'center':center,'yaw':yaw,'wh':wh}
 return dict(anchor,frames=rows,min_GT_clearance_m=min_gap,max_ground_support_distance_m=max_support,relative_speed_mps=delta,trajectory_policy='world_static' if static else 'extended_old_virtual_path_plus_relative_speed'),None

def main(shard,count):
 cv2.setNumThreads(1);g=geometry();original={c['source_id']:c for c in read(F/'source_manifest.json')['clips']};pool=read(R8/'all_asset_candidates.json')['candidates'];by=defaultdict(list)
 for p in pool:by[p['source_id']].append(p)
 out=O/'planned';out.mkdir(exist_ok=True);assets={n:dict(np.load(R8/'assets'/f'{n}.npz')) for n in ['sedan','suv']}
 for sid in sorted(g.sources)[shard::count]:
  dest=out/(sid+'.json')
  if dest.exists():continue
  start=time.time();ground=g.prepare(sid);pm=protections(g,sid);rej=Counter();chosen=defaultdict(list);c=g.sources[sid]
  if ground['pass'] and pm:
   support(g,sid)
   for old_anchor in sorted(by[sid],key=lambda a:(a['type']=='background',a['asset'],a['offset_longitudinal_m'],a['offset_lateral_m']))[:4]:
    anchor=extended(g,c,old_anchor,original[sid]);targets=old_anchor['protected_instances'] or [c['actors'][0]['instance_token']];tok=targets[0]
    if tok not in pm:rej['protected_mask_unavailable']+=1;continue
    for delta,static in [(d,False) for d in fitted(c,anchor,tok)]+[(0.,True)]:
     tr,why=trajectory(g,c,anchor,delta,static)
     if tr is None:rej[why]+=1;continue
     mesh=assets[anchor['asset']];alphas=[old.silhouette(mesh['vertices'],mesh['faces'],p['actor'],f) for p,f in zip(tr['frames'],c['frames'])];q,why=exact(g,tr,alphas,pm)
     if q is None:rej[why]+=1;continue
     active={t:pm[t] for t in q['protected_instances']};ba={t:[next(a for a in f['actors'] if a['instance_token']==t) for f in c['frames']] for t in active};hs=[cv2.dilate(a.astype('uint8'),np.ones((7,7),'uint8'))>0 for a in alphas];proc=process(c['frames'],[p['actor'] for p in tr['frames']],hs,active,ba);family='sweep_B' if proc['sweep_over_any_B'] else 'visibility_transition' if proc['visibility_transition_any_B'] else 'static_A_moving_ego' if proc['static_A_moving_ego_image_change'] else None
     if family is None:rej['no_actual_process']+=1;continue
     if len(chosen[family])>=2:continue
     chosen[family].append(tr|q|{'process_family':family,'temporal_process':proc,'source_split':c['source_split'],'human_verdict':None,'ground':{k:v for k,v in g.ground[sid].items() if not k.startswith('_')}})
  else:rej['ground_or_stable_masks_unavailable']+=1
  rows=[c for vv in chosen.values() for c in vv];dump(dest,{'source_id':sid,'scene':c['scene'],'source_split':c['source_split'],'candidates':rows,'counts':dict(Counter(r['process_family'] for r in rows)),'rejects':dict(rej),'seconds':time.time()-start,'ground_pass':ground['pass']});print('LONG_FACTORY',sid,len(rows),dict(rej),flush=True)
 dump(O/f'factory_shard_{shard}.json',{'stage':'complete','shard':shard,'sources':len(sorted(g.sources)[shard::count]),'human_verdict':None})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--count',type=int,default=1);a=p.parse_args();main(a.shard,a.count)
