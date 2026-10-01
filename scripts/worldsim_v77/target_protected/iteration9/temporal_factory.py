"""固定世界轨迹工厂：不逐帧跟随B，扫过能力以实测过程认定。"""
from pathlib import Path
import sys,os,math,time,copy
from collections import Counter,defaultdict
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(P));sys.path.insert(0,str(P/'iteration8'))
import asset_factory as old
from geometry_factory import read,dump,footprint,ground_orientation,projection,wrap,source_masks
from iteration5.planning import normalized_masks
from temporal_metrics import process
import numpy as np,cv2
from pyquaternion import Quaternion
T=old.T;R8=T/'r8';O=T/'r9';F=R8/'factory'

def geometry():
 # 只复用旧地面/相机缓存，所有新位姿与逐帧障碍重新检查。
 return old.ParkingGeometry(F)

def protections(geo,sid):
 c=geo.sources[sid];out={}
 for j,a in enumerate(c['actors']):
  tok=a['instance_token'];jid=sid if j==0 else sid+'_'+tok[:8];folder=F/('segmented' if j==0 else 'segmented_secondary')/jid/'sam2_raw'
  if not folder.exists():continue
  mm=source_masks(F,sid,None if j==0 else jid)
  st=normalized_masks(mm,[next(b for b in f['actors'] if b['instance_token']==tok)['projection']['box_xyxy'] for f in c['frames']])
  if len(mm)==10 and all(s['pixels'] and s['normalized_iou']>=.8 and .85<=s['normalized_area_ratio']<=1.15 for s in st):out[tok]=mm
 return out

def trajectory(geo,anchor,velocity):
 sid=anchor['source_id'];c=geo.sources[sid];plane=np.array(geo.ground[sid]['plane']);mid=c['frames'][5]
 ref=anchor['frames'][5]['actor'];r=Quaternion(ref['rotation']).rotation_matrix;yaw=math.atan2(r[1,0],r[0,0]);R=ground_orientation(yaw,plane)
 v=np.array([math.cos(yaw),math.sin(yaw)])*velocity;xy0=np.array(ref['translation'])[:2];size=ref['size'];rows=[];gap_min=1e9;support_max=0;prev=None
 for f,obs in zip(c['frames'],geo.obstacles[sid]):
  dt=(f['timestamp']-mid['timestamp'])/1e6;xy=xy0+v*dt;bottom=np.r_[xy,np.dot(np.r_[xy,1],plane)];center=bottom+R[:,2]*size[2]/2
  actor={'translation':center.tolist(),'rotation':Quaternion(matrix=R).elements.tolist(),'size':size};foot=footprint(actor)
  if not geo.road.covers(foot):return None,'outside_mapped_drivable_or_parking'
  support=geo.ground[sid]['_tree'].query(np.array(foot.exterior.coords)[:4])[0].max();support_max=max(support_max,float(support))
  if support>2.5:return None,'ground_support_gap'
  pr=projection(actor,f)
  if pr is None:return None,'behind_camera'
  bb=pr['box'];wh=bb[2:]-bb[:2]
  if min(bb[0],bb[1],1024-bb[2],576-bb[3])<8 or wh[0]<72 or wh[1]<40 or not .006<=np.prod(wh)/(1024*576)<=.18:return None,'size_border'
  if bb[3]+6>=512:return None,'ego_band'
  for ob in obs:
   gap=foot.distance(ob['_foot']);gap_min=min(gap_min,gap)
   if gap<.3:return None,'collision_clearance'
  if prev:
   fac=100000/(f['timestamp']-prev['timestamp']);ratio=(wh/prev['wh'])**fac
   if np.linalg.norm(center-prev['center'])*fac>2 or ratio.min()<.85 or ratio.max()>1.18:return None,'trajectory_continuity'
  rows.append({'frame':f['frame'],'timestamp':f['timestamp'],'actor':actor,'box':bb.tolist(),'depth_interval':[pr['near_depth'],pr['far_depth']]});prev={'timestamp':f['timestamp'],'center':center,'wh':wh}
 return {'source_id':sid,'scene':c['scene'],'frames':rows,'asset':anchor['asset'],'min_GT_clearance_m':float(gap_min),'max_ground_support_distance_m':support_max,
  'ground':{k:v for k,v in geo.ground[sid].items() if not k.startswith('_')},'trajectory_policy':'world_static' if velocity==0 else 'constant_world_velocity',
  'world_velocity_xy_mps':v.tolist(),'speed_signed_mps':velocity,'anchor_offsets_m':[anchor['offset_longitudinal_m'],anchor['offset_lateral_m']]},None

def exact(geo,tr,alphas,pm):
 sid=tr['source_id'];ratios={t:[] for t in pm};h_ratios={t:[] for t in pm};static=set()
 for i,(aa,p,obs) in enumerate(zip(alphas,tr['frames'],geo.obstacles[sid])):
  H=cv2.dilate(aa.astype('uint8'),np.ones((13,13),'uint8'))>0;yy,xx=np.where(aa)
  if len(yy)<200 or np.ptp(xx)+1<72 or np.ptp(yy)+1<40:return None,'actual_size'
  if H[512:].any():return None,'ego_band'
  for tok,mm in pm.items():ratios[tok].append(float((aa&mm[i]).sum()/mm[i].sum()));h_ratios[tok].append(float((H&mm[i]).sum()/mm[i].sum()))
  for ob in obs:
   pr=ob['_projection'];tok=ob['instance_token']
   if pr is None:continue
   bb=pr['box'];x0,y0=np.maximum(np.floor(bb[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(bb[2:]).astype(int),[1024,576])
   if x1<=x0 or y1<=y0 or H[y0:y1,x0:x1].sum()<=max(12,.01*H.sum()):continue
   if tok in pm:
    if h_ratios[tok][-1]>.01 and p['depth_interval'][1]>.3+pr['near_depth']:return None,'protected_depth_order'
   elif ob['category'] in {'movable_object.barrier','movable_object.trafficcone','static_object.bicycle_rack'}:
    if ob.get('interpolation_uncertain') or p['depth_interval'][1]>.3+pr['near_depth']:return None,'static_foreground'
    static.add(tok)
   else:return None,'unreviewed_dynamic_envelope'
 active=[t for t,v in ratios.items() if max(v)>.01]
 if active:
  if any(max(h_ratios[t])>.85 for t in active):return None,'protected_evidence_exhausted'
  if len(active)==1:
   v=np.array(ratios[active[0]])
   if max(v)<.30 or v.mean()<.10 or sum(v>.05)<3:return None,'insufficient_actual_occlusion'
   kind='single_actor'
  else:
   if any(max(ratios[t])<.20 or np.mean(ratios[t])<.08 for t in active):return None,'insufficient_dense_occlusion'
   kind='dense_actors'
 else:kind='background'
 st=normalized_masks(alphas,[p['box'] for p in tr['frames']])
 if any(s['normalized_iou']<.8 or not .85<=s['normalized_area_ratio']<=1.15 for s in st):return None,'mask_continuity'
 return {'type':kind,'protected_instances':active,'occlusion_fraction':ratios,'hole_overlap_upper':h_ratios,'silhouette_stats':st,'static_background_annotations_behind_A':sorted(static)},None

def main(shard,count,phase='reuse'):
 cv2.setNumThreads(1);geo=geometry();pool=read(R8/'all_asset_candidates.json')['candidates'];assets={n:dict(np.load(R8/'assets'/f'{n}.npz')) for n in ['sedan','suv']}
 bysid=defaultdict(list)
 for p in pool:bysid[p['source_id']].append(p)
 if phase=='expand':
  excluded=set(read(R8/'source_split.json')['old_r7_training_scenes'])|{c['receiver_scene'] for c in read(R8/'dataset_catalog.json')['cases'] if c['split']=='train'}
  if (O/'expansion_sources.json').exists():selected=read(O/'expansion_sources.json')['sources']
  else:
   candidates=[]
   for sid,c in geo.sources.items():
    if not sid.startswith('N') or sid not in geo.context:continue
    fs=c['frames'];e=np.array([f['camera_to_world'] for f in fs])[:,:3,3];b=np.array([f['actors'][0]['translation'] for f in fs]);score=np.linalg.norm(e[-1]-e[0])+np.linalg.norm(b[-1]-b[0])
    if score<.5:continue
    candidates.append((c['scene'] in excluded,-score,c['scene'],sid))
   selected=[];scene_counts=Counter()
   for blocked in [False,True]:
    budget=12 if not blocked else 4;selected_this=0
    for bl,score,scene,sid in sorted(candidates):
     if bl!=blocked or scene_counts[scene] or selected_this>=budget:continue
     if not protections(geo,sid):continue
     selected.append(sid);scene_counts[scene]+=1;selected_this+=1
   dump(O/'expansion_sources.json',{'sources':selected,'excluded_all_prior_training_scenes':sorted(excluded),'rule':'first12 unseen training worlds +4 known training worlds by actual ego+B displacement, one source/scene; input only','budget_sources':16,'speed_grid':[0,-4,4],'z_grid':[4,6,8,10,12],'x_grid':[-4,-2,0,2,4]})
  bysid={sid:[] for sid in selected}
 savedroot=O/('temporal_candidates' if phase=='reuse' else 'expansion_candidates');savedroot.mkdir(exist_ok=True);all_rows=[];rows=[]
 # 只使用已经有真实mask及合法静态空间锚点的来源，一次固定五种速度。
 for sid in sorted(bysid)[shard::count]:
  saved=savedroot/f'{sid}.json'
  if saved.exists():r=read(saved);all_rows+=r['candidates'];rows.append(r['summary']);continue
  start=time.time();g=geo.prepare(sid);pm=protections(geo,sid);reject=Counter();chosen=defaultdict(list)
  if g['pass'] and pm:
   # 每source最多6个独立锚点，每锚点5种速度；固定输入排序，不看生成结果。
   anchors=sorted(bysid[sid],key=lambda p:(p['type']=='background',p['asset'],p['offset_longitudinal_m'],p['offset_lateral_m']))[:6]
   if phase=='expand':
    anchors=[]
    for z in [4.,6.,8.,10.,12.]:
     for x in [-4.,-2.,0.,2.,4.]:
      anchor,why=old.trajectory(geo,sid,z,x,[1.85,4.5,1.5])
      if anchor is not None:anchors.append(anchor|{'asset':'sedan'})
   for anchor in anchors:
    for speed in ([0.,-2.,2.,-4.,4.] if phase=='reuse' else [0.,-4.,4.]):
     tr,why=trajectory(geo,anchor,speed)
     if tr is None:reject[why]+=1;continue
     mesh=assets[tr['asset']];aa=[old.silhouette(mesh['vertices'],mesh['faces'],p['actor'],f) for p,f in zip(tr['frames'],geo.sources[sid]['frames'])]
     q,why=exact(geo,tr,aa,pm)
     if q is None:reject[why]+=1;continue
     active={t:pm[t] for t in q['protected_instances']};ba={t:[next(a for a in f['actors'] if a['instance_token']==t) for f in geo.sources[sid]['frames']] for t in active}
     hs=[cv2.dilate(a.astype('uint8'),np.ones((7,7),'uint8'))>0 for a in aa]
     proc=process(geo.sources[sid]['frames'],[p['actor'] for p in tr['frames']],hs,active,ba)
     family='sweep_B' if proc['sweep_over_any_B'] else 'static_A_moving_ego' if proc['static_A_moving_ego_image_change'] else 'visibility_transition' if proc['visibility_transition_any_B'] else None
     if family is None:reject['no_new_actual_temporal_process']+=1;continue
     if len(chosen[family])>=2:continue
     chosen[family].append(tr|q|{'process_family':family,'temporal_process':proc,'edge_mode':'near_hard','mask_dilation_px':3,'human_verdict':None})
  pp=[p for ar in chosen.values() for p in ar];all_rows+=pp;r={'source_id':sid,'scene':geo.sources[sid]['scene'],'counts':dict(Counter(p['process_family'] for p in pp)),'types':dict(Counter(p['type'] for p in pp)),'rejects':dict(reject),'seconds':time.time()-start};rows.append(r);dump(saved,{'summary':r,'candidates':pp})
  dump(O/f'{phase}_factory_state_{shard}.json',{'completed':len(rows),'total':len(sorted(bysid)[shard::count]),'candidates':len(all_rows),'process_counts':dict(Counter(p['process_family'] for p in all_rows)),'pid':os.getpid()});print(sid,r['counts'],round(r['seconds'],1),flush=True)
 dump(O/(f'temporal_pool_{shard}.json' if phase=='reuse' else f'expansion_pool_{shard}.json'),{'stage':'complete','candidates':all_rows,'rows':rows,'process_counts':dict(Counter(p['process_family'] for p in all_rows)),'phase':phase,'speed_grid_mps':[0,-2,2,-4,4] if phase=='reuse' else [0,-4,4],'max_anchors_per_source':6 if phase=='reuse' else 25})
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--count',type=int,default=1);ap.add_argument('--phase',choices=['reuse','expand'],default='reuse');a=ap.parse_args();main(a.shard,a.count,a.phase)
