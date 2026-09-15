"""冻结官方输出的首次回波诊断。标定、目标轨迹、尺度锚点的额外信息显式记录。"""
import os,pathlib,json,time,argparse,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch,open3d as o3d
from scipy.spatial import cKDTree
from geometry_contract import P,R,outputs,build_control
torch.set_num_threads(2)
D=P/'runs/worldsim_v73/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
RAW=P/'runs/worldsim_v73/WS-V73-M1-NATIVE-GEOMETRY-ADAPT-01/20260907T161500Z__native-dpt-surround25-dev6-s7301-r3/build_observations.pt'
def transform(x,T):return x@T[:3,:3].T+T[:3,3]
def mesh_for_case(c,views,cam,conf,K,protocol,scale):
 h,w=cam.shape[1:3];yy,xx=np.mgrid[:h,:w];pix=np.stack([xx,yy,np.ones_like(xx)],-1)
 keep=float(protocol.split('_')[-1]) if protocol.startswith('cal_build_') else 1.
 scalar=scale if 'build' in protocol else 1.;cal=protocol.startswith('cal_')
 vertices=[];faces=[];offset=0;source=[];size=np.asarray(c['size_lwh_m']);indices=list(map(int,c['view_indices']))
 for i,v in enumerate(views):
  idx=v['source_view_index']
  if idx not in indices:continue
  j=indices.index(idx);T=np.linalg.inv(np.asarray(c['camera_from_actor'][j]));cp=cam[i]*scalar
  if cal:cp=(pix@np.linalg.inv(K[i]).T)*cp[...,2:3]
  actor=transform(cp,T);valid=np.isfinite(actor).all(-1)&(cp[...,2]>.2)&(cp[...,2]<100)&(np.abs(actor)<=size/2+1.).all(-1)
  if keep<1:
   vals=conf[i][np.isfinite(conf[i])];thr=np.quantile(vals,1-keep);valid&=conf[i]>=thr
  if valid.sum()<3:continue
  grid=np.arange(h*w).reshape(h,w)
  fs=np.concatenate([np.stack([grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1]],-1).reshape(-1,3),np.stack([grid[1:,1:],grid[1:,:-1],grid[:-1,1:]],-1).reshape(-1,3)])
  fs=fs[valid.ravel()[fs].all(-1)]
  if len(fs)==0:continue
  # 固定断边阈值，防止跨空洞连出额外墙面。
  p=cp.reshape(-1,3)[fs];edges=np.max(np.stack([np.linalg.norm(p[:,0]-p[:,1],axis=-1),np.linalg.norm(p[:,1]-p[:,2],axis=-1),np.linalg.norm(p[:,2]-p[:,0],axis=-1)],-1),-1)
  fs=fs[edges<=np.maximum(.15,.05*np.min(p[...,2],axis=1))]
  used,inv=np.unique(fs,return_inverse=True)
  if len(used)==0:continue
  vertices.append(actor.reshape(-1,3)[used]);faces.append(inv.reshape(-1,3)+offset);source.append(np.column_stack([np.full(len(used),i),used]));offset+=len(used)
 return (np.concatenate(vertices).astype('float32'),np.concatenate(faces).astype('uint32'),np.concatenate(source)) if vertices else (np.empty((0,3),'float32'),np.empty((0,3),'uint32'),np.empty((0,2),'int64'))
def query(c):
 frames=[f for f in c['rays'] if f['role']=='heldout_time'];parts=[];offsets=[0];frame_ids=[]
 for f in frames:
  positive=np.asarray(f['positive_actor']).astype(bool);amb=np.asarray(f['ambiguous_owner']).astype(bool);good=positive&~amb
  parts.append((np.asarray(f['origins_actor_m'])[good],np.asarray(f['directions_actor'])[good],np.asarray(f['observed_first_range_m'])[good]));offsets.append(offsets[-1]+int(good.sum()));frame_ids.extend([int(f['sample_index'])]*int(good.sum()))
 if not parts:return np.empty((0,3)),np.empty((0,3)),np.empty(0),np.array(offsets),np.array(frame_ids)
 return *(np.concatenate([p[i] for p in parts]) for i in range(3)),np.array(offsets),np.array(frame_ids)
def measure(V,F,O,dirs,rg):
 n=len(rg);target=O+dirs*rg[:,None];t=np.full(n,np.inf);ids=np.full(n,-1);dist=np.full(n,np.inf);anygood=np.zeros(n,bool)
 if len(F) and n:
  mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V),o3d.core.Tensor(F));scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(mesh)
  ray=o3d.core.Tensor(np.concatenate([O,dirs],-1).astype('float32'));hits=scene.cast_rays(ray,nthreads=4);t=hits['t_hit'].numpy();rawid=hits['primitive_ids'].numpy();ids=np.where(np.isfinite(t),rawid,-1).astype('int64')
  allhits=scene.list_intersections(ray,nthreads=4);ri=allhits['ray_ids'].numpy();ts=allhits['t_hit'].numpy();ok=np.abs(ts-rg[ri])<=.2;anygood[ri[ok]]=True
  dist=cKDTree(V).query(target,workers=2)[0]
 delta=t-rg;finite=np.isfinite(t)
 counts={'rays':n,'hit':int((finite&(np.abs(delta)<=.2)).sum()),'early':int((finite&(delta<-.2)).sum()),'late':int((finite&(delta>.2)).sum()),'miss':int((~finite).sum()),'near_vertices':int((dist<=.2).sum()),'any_correct_intersection':int(anygood.sum()),'early_with_later_correct':int((anygood&(delta<-.2)).sum()),'intrusion_sum_m':float(np.maximum(rg[finite]-t[finite],0).sum())}
 for eps in [.1,.3,.5]:counts['early_'+str(eps)]=int((delta<-eps).sum())
 return counts,t,ids,dist,anygood
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scenes',nargs='*');ap.add_argument('--methods',nargs='*');args=ap.parse_args()
 scenes=torch.load(RAW,weights_only=False,map_location='cpu',mmap=True);cohort=json.loads((R/'cohort.json').read_text());root=R/'evaluation_bgscale';root.mkdir(exist_ok=True)
 # 这两个可靠性分层在首次 QUERY 求交前固定，不改变原始全分母。
 amend=R/'evaluation_amendment.json'
 if not amend.exists():amend.write_text(json.dumps({'registered_before_query_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'reference_strata':'All positive non-ambiguous returns plus BUILD-supported endpoints <=0.20m (nearest owned BUILD point), reported separately. Static <=0.5m/s. No sample removed.','protocols':['native_base','native_build','cal_base','cal_build_1.0','cal_build_0.9','cal_build_0.75','cal_build_0.5'],'reason':'Separate native point projection error, ordinary metric scale, native depth with known camera intrinsics, and confidence filtering.'},indent=2))
 for f in sorted((R/'predictions').glob('*/*/*/result.json')):
  rec=json.loads(f.read_text())
  if args.scenes and rec['scene'] not in args.scenes:continue
  if args.methods and rec['method'] not in args.methods:continue
  out=root/rec['method']/rec['scene']/rec['variant'];out.mkdir(parents=True,exist_ok=True)
  if (out/'summary.json').exists():continue
  s=next(x for x in scenes if x['scene_id']==rec['scene']);ctrl=build_control(f.parent,s);scalar=ctrl['build_global_scalar'];assert scalar is not None and scalar>0
  rec,views,cam,conf,K,base,cv=outputs(f.parent);rows=[];started=time.time()
  for entry in [x for x in cohort if x['scene']==rec['scene']]:
   c=torch.load(D/entry['file'],weights_only=False,map_location='cpu');O,dirs,rg,offs,frameids=query(c);target=O+dirs*rg[:,None];b=np.asarray(c['points_actor_m']);near=cKDTree(b).query(target)[0] if len(b) and len(rg) else np.full(len(rg),np.inf)
   supported=near<=.2;speed=entry.get('translation_speed_mps');static=(speed<=.5) if speed is not None else None
   dest=out/entry['owner'];dest.mkdir(exist_ok=True)
   if not len(rg):rows.append({'owner':entry['owner'],'status':'NO_OWNED_QUERY','static':static,'rays':0});continue
   for protocol in ['native_base','native_build','cal_base','cal_build_1.0','cal_build_0.9','cal_build_0.75','cal_build_0.5']:
    V,F,source=mesh_for_case(c,views,cam,conf,K,protocol,scalar);counts,t,ids,dist,anygood=measure(V,F,O,dirs,rg)
    sub={};delta=t-rg
    for name,mask in [('build_supported',supported),('query2',frameids==2),('query5',frameids==5)]:
     sub[name]={'rays':int(mask.sum()),'early':int((mask&(delta<-.2)).sum()),'hit':int((mask&np.isfinite(t)&(np.abs(delta)<=.2)).sum()),'miss':int((mask&~np.isfinite(t)).sum()),'near_vertices':int((mask&(dist<=.2)).sum())}
    row={'owner':entry['owner'],'category':entry['category'],'static':static,'speed_mps':entry['translation_speed_mps'],'status':'EVALUATED','protocol':protocol,'vertices':len(V),'faces':len(F),'counts':counts,'strata':sub};rows.append(row)
    np.savez_compressed(dest/(protocol+'_rays.npz'),origins=O,directions=dirs,ranges=rg,first=t,face_ids=ids,near_distance=dist,any_correct=anygood,build_supported=supported,frame_ids=frameids)
    if protocol=='cal_build_1.0':np.savez_compressed(dest/'primary_surface.npz',vertices=V,faces=F,source_view_pixel=source,build_points=b,size=np.asarray(c['size_lwh_m']))
   print(json.dumps({'stage':'ACTOR_DONE','method':rec['method'],'scene':rec['scene'],'variant':rec['variant'],'owner':entry['owner'],'rays':len(rg)}),flush=True)
  summary={'method':rec['method'],'scene':rec['scene'],'log':s['log_id'],'variant':rec['variant'],'build_control':ctrl,'extra_information':'All surface diagnostics use known camera poses and annotated actor motion. cal uses known K. build adds background LiDAR scale. No QUERY used for model/scale/mesh.','rows':rows,'elapsed_s':time.time()-started}
  (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'stage':'WINDOW_DONE','method':rec['method'],'scene':rec['scene'],'variant':rec['variant'],'elapsed_s':summary['elapsed_s']}),flush=True)
if __name__=='__main__':main()
