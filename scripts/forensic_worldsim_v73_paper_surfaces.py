"""从保存曲面读取首面责任、形状、观测支持与Oracle阶梯；不训练模型。"""
import json, sys, time, resource
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/motion_proj')
BASE=Path('/root/autodl-tmp/runs/worldsim_v73')
OUT=ROOT/'docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1'
DATA=BASE/'WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
MODELS={
 'AdaPoinTr':'WS-V73-M2-ADAPOINTR-01/20260907T231000Z__population-full-track-pcn-yup-s7307-r2',
 'VGGT-native':'WS-V73-M2-GLOBAL-FUSION-01/20260907T203500Z__population-native-lidar-fusion-r2',
 'LiDAR-R8':'WS-V73-M2-GLOBAL-ACTOR-01/20260907T212500Z__population-lidar-track-beam-range-s7304-r8',
 'Open-r3':'WS-V73-Q-V2-01/20260909T054000Z__open-charts-lidar-full-track-beam-s7304-r3',
 'Attraction-r4':'WS-V73-Q-V2-01/20260909T063700Z__open-charts-lidar-ray-support-s7304-r4',
 'First-r6':'WS-V73-Q-V2-01/20260909T123000Z__open-charts-lidar-first-surface-s7304-r6',
}
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def tensor(x):return o3d.core.Tensor(np.asarray(x,dtype=np.float32))
def scene_for(v,f):
 s=o3d.t.geometry.RaycastingScene(nthreads=2)
 if len(f):s.add_triangles(tensor(v),o3d.core.Tensor(f.astype(np.uint32)))
 return s
def cast(s,rays,nfaces):
 if not nfaces:return np.full(len(rays),np.inf),np.full(len(rays),-1),np.zeros((len(rays),2))
 d=s.cast_rays(tensor(rays),nthreads=2)
 return d['t_hit'].numpy(),d['primitive_ids'].numpy().astype(np.int64),d['primitive_uvs'].numpy()
def cls(t,r):
 return {'hit':np.isfinite(t)&(np.abs(t-r)<=.2),'early':np.isfinite(t)&(t<r-.2),'miss':~np.isfinite(t)}
def aggregate(rows):
 logs=defaultdict(list)
 for row in rows:
  if row['rays']:logs[row['log_id']].append(row['metrics'])
 pm={k:{m:float(np.mean([r[m] for r in x])) for m in x[0]} for k,x in logs.items()}
 return {'actors':len(rows),'owned_actors':sum(x['rays']>0 for x in rows),'raw_rays':sum(x['rays'] for x in rows),'per_log':pm,'equal_log':{m:float(np.mean([x[m] for x in pm.values()])) for m in next(iter(pm.values()))}}

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
 entries=[e for e in json.loads((DATA/'index.json').read_text())['cases'] if e['role']=='development']
 (OUT/'manifest.json').write_text(json.dumps({'task':'WS-V73-PAPER-FORENSICS-01','models':MODELS,'actor_data':str(DATA),'scope':'original 75 DEV actors, owned heldout rays; no external20 reads; zero optimization','failure_ledger_refs':['V73-F02','V73-F03','V73-F04','V73-F09'],'selection':'one actor per method: >=20 owned rays and >=5 early rays, closest early fraction to median among eligible failures; one median-severity early ray with later hit when available','quality':'q=sum squared edge lengths / (4 sqrt(3) area); q>10 descriptive sliver threshold','component_support':'any build endpoint within 0.2 m unsigned distance to a component; disconnected chart is not automatically a floater','oracles':'any-hit and near-surface use heldout endpoints; remove-early-faces deletes union of DEV early faces then recasts all owned rays; diagnostic only, no deployment edit'},indent=2))
 allrows={k:[] for k in MODELS}; candidates={k:[] for k in MODELS};start=time.time()
 for e in entries:
  c=load(DATA/e['file']); frames=[f for f in c['rays'] if f['role']=='heldout_time']
  origin=np.concatenate([f['origins_actor_m'].numpy()[f['positive_actor'].numpy()] for f in frames]) if frames else np.empty((0,3),np.float32)
  direct=np.concatenate([f['directions_actor'].numpy()[f['positive_actor'].numpy()] for f in frames]) if frames else np.empty((0,3),np.float32)
  ranges=np.concatenate([f['observed_first_range_m'].numpy()[f['positive_actor'].numpy()] for f in frames]) if frames else np.empty(0,np.float32)
  rays=np.c_[origin,direct]; endpoints=origin+direct*ranges[:,None];build=c['points_actor_m'].numpy()
  for name,run in MODELS.items():
   path=BASE/run/(e['owner']+'_surface.pt')
   if not path.exists():path=BASE/run/(e['scene']+'__'+e['owner']+'_surface.pt')
   surf=load(path);v=surf['vertices_actor_m'].numpy();f=surf['faces'].numpy();s=scene_for(v,f)
   first,ids,uv=cast(s,rays,len(f));masks=cls(first,ranges);valid=np.isfinite(first)
   anyhit=np.zeros(len(ranges),bool);later=np.full(len(ranges),np.inf);near=np.zeros(len(ranges),bool)
   near_dist=np.full(len(ranges),np.inf)
   q=np.zeros(len(f)); comp=np.zeros(len(f),int);ncomp=0;csup=np.zeros(0,bool);cn=np.zeros(0,int)
   if len(f):
    tri=v[f];edges=tri[:,[1,2,0]]-tri;cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=-1)/2
    q=np.sum(edges**2,axis=(1,2))/np.maximum(4*np.sqrt(3)*area,1e-16)
    normal=cross/np.maximum(2*area[:,None],1e-16)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
    cc=mesh.cluster_connected_triangles();comp=np.asarray(cc[0]);cn=np.asarray(cc[1]);ncomp=len(cn)
    csup=np.zeros(ncomp,bool)
    if len(build):
     # 精确到组件的最近面距离；每个组件保留其原有三角面。
     for ci in range(ncomp):
      vs=scene_for(v,f[comp==ci]);dist=vs.compute_distance(tensor(build),nthreads=2).numpy();csup[ci]=bool(np.any(dist<=.2))
    if len(ranges):
     near_dist=s.compute_distance(tensor(endpoints),nthreads=2).numpy();near=near_dist<=.2
     inter=s.list_intersections(tensor(rays),nthreads=2);ri=inter['ray_ids'].numpy().astype(int);depth=inter['t_hit'].numpy();good=np.abs(depth-ranges[ri])<=.2
     anyhit[ri[good]]=True;np.minimum.at(later,ri[good],depth[good])
   row={'owner':e['owner'],'scene':e['scene'],'log_id':e['log_id'],'rays':len(ranges),'faces':len(f),'components':ncomp,'supported_components':int(csup.sum()),'counts':{k:int(a.sum()) for k,a in masks.items()}}
   if len(ranges):
    metrics={k:float(x.mean()) for k,x in masks.items()};metrics.update(any_hit=float(anyhit.mean()),near=float(near.mean()),early_with_later=float((masks['early']&anyhit).mean()))
    earlyids=ids[masks['early']];allids=ids[valid]
    metrics.update(early_sliver=float(np.sum(q[earlyids]>10)/len(ranges)),early_unsupported_component=float(np.sum(~csup[comp[earlyids]])/len(ranges)),early_supported_component=float(np.sum(csup[comp[earlyids]])/len(ranges)))
    for tag,keep in [('shape',q<=10),('build_component',csup[comp] if len(f) else np.zeros(0,bool)),('oracle_early_face',~np.isin(np.arange(len(f)),earlyids))]:
     tt,_,_=cast(scene_for(v,f[keep]),rays,int(keep.sum()))
     metrics.update({tag+'_'+k:float(x.mean()) for k,x in cls(tt,ranges).items()})
    row['metrics']=metrics
    row['early_face_q_median']=float(np.median(q[earlyids])) if len(earlyids) else None
    row['early_face_q_max']=float(np.max(q[earlyids])) if len(earlyids) else None
    row['early_incidence_median']=float(np.median(np.abs((normal[earlyids]*direct[masks['early']]).sum(-1)))) if len(earlyids) else None
    if len(ranges)>=20 and masks['early'].sum()>=5:
     candidates[name].append((row,{'vertices':v,'faces':f,'build_points':build,'target_points':endpoints,'origins':origin,'directions':direct,'ranges':ranges,'first':first,'face_ids':ids,'uv':uv,'any_hit':anyhit,'later':later,'q':q,'components':comp,'component_support':csup,'size':c['size_lwh_m'].numpy(),'normals':normal,'near_distance':near_dist}))
   allrows[name].append(row)
  print(e['scene'],e['owner'],flush=True)
 selected={}
 for name,items in candidates.items():
  med=np.median([x[0]['metrics']['early'] for x in items]);row,arr=min(items,key=lambda x:(abs(x[0]['metrics']['early']-med),x[0]['owner']))
  early=np.isfinite(arr['first'])&(arr['first']<arr['ranges']-.2);eligible=np.where(early&arr['any_hit'])[0]
  if not len(eligible):eligible=np.where(early)[0]
  err=arr['ranges'][eligible]-arr['first'][eligible];mid=np.median(err);rid=int(eligible[np.argmin(abs(err-mid))]);fid=int(arr['face_ids'][rid]);arr['selected_ray']=np.array(rid)
  np.savez_compressed(OUT/'cases'/(name+'.npz'),**arr)
  selected[name]={'actor':row,'selected_ray':rid,'face_id':fid,'face_q':float(arr['q'][fid]),'component':int(arr['components'][fid]),'component_supported':bool(arr['component_support'][arr['components'][fid]]),'early_error_m':float(arr['ranges'][rid]-arr['first'][rid]),'later_correct_hit':bool(arr['any_hit'][rid]),'eligible_actors':len(items),'selection_median_early':float(med)}
 result={'status':'done','rows':allrows,'statistics':{k:aggregate(v) for k,v in allrows.items()},'selected':selected,'wall_s':time.time()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'selected':selected,'stats':{k:v['equal_log'] for k,v in result['statistics'].items()},'wall_s':result['wall_s']},indent=2))

if __name__=='__main__':main()
