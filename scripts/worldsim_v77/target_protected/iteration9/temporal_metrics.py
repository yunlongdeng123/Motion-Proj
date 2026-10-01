"""实测遮挡过程；GT cuboid 表面格是对应代理，不能认证真实纹理已见。"""
import numpy as np
from pyquaternion import Quaternion

THRESHOLDS={'world_static_diameter_m':.20,'ego_moving_path_m':.50,
 'image_motion_px':8.,'normalized_sweep_span':.20,'occlusion_range':.15,
 'minimum_overlap_pixels':20,'surface_grid':16}

def motion(points):
 p=np.asarray(points,float)
 return {'endpoint_m':float(np.linalg.norm(p[-1]-p[0])),
  'path_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
  'diameter_m':float(np.linalg.norm(p[:,None]-p[None,:],axis=-1).max())}

def cell_pixels(frame,actor,mask):
 """相机光线与已知B定向框交点；仅在真实SAM2 B像素上采样。"""
 yy,xx=np.where(mask)
 if not len(xx):return yy,xx,np.empty(0,dtype=int)
 K=np.array(frame['intrinsics_1024']);C=np.array(frame['camera_to_world']);R=Quaternion(actor['rotation']).rotation_matrix
 rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T@C[:3,:3].T@R
 origin=(C[:3,3]-np.array(actor['translation']))@R
 dim=np.array([actor['size'][1],actor['size'][0],actor['size'][2]])
 safe=np.where(abs(rays)<1e-10,1e-10,rays)
 lo=(-dim/2-origin)/safe;hi=(dim/2-origin)/safe
 enter=np.minimum(lo,hi).max(1);leave=np.maximum(lo,hi).min(1)
 valid=(leave>=enter)&(enter>0)&np.isfinite(enter)
 hit=origin+rays[valid]*enter[valid,None]
 bins=np.floor((hit/dim+.5)*THRESHOLDS['surface_grid']).astype(int).clip(0,THRESHOLDS['surface_grid']-1)
 key=bins[:,0]*256+bins[:,1]*16+bins[:,2]
 return yy[valid],xx[valid],key

def cells(frame,actor,mask,hole):
 yy,xx,key=cell_pixels(frame,actor,mask);hidden=hole[yy,xx]
 def count(k):
  u,n=np.unique(k,return_counts=True);return dict(zip(u.tolist(),n.tolist()))
 return count(key[hidden]),count(key[~hidden])

def process(frames,actors,holes,protected=None,protected_actors=None,occluder_masks=None):
 e=motion([np.array(f['camera_to_world'])[:3,3] for f in frames]);a=motion([r['translation'] for r in actors])
 cent=[];area=[]
 for h in holes:
  yy,xx=np.where(h);cent.append([float(xx.mean()),float(yy.mean())] if len(xx) else [float('nan')]*2);area.append(int(len(xx)))
 finite=np.array(cent)[np.isfinite(cent).all(1)];diam=float(np.linalg.norm(finite[:,None]-finite[None,:],axis=-1).max()) if len(finite) else 0
 row={'frames':len(frames),'duration_s':(frames[-1]['timestamp']-frames[0]['timestamp'])/1e6,
  'ego_camera_motion':e,'A_world_motion':a,'hole_centroid_path_px':float(np.linalg.norm(np.diff(finite,axis=0),axis=1).sum()) if len(finite) else 0,
  'hole_centroid_diameter_px':diam,'hole_area_min_max':[min(area),max(area)],'hole_centroids':cent,
  'static_A_moving_ego':a['diameter_m']<=.20 and e['path_m']>=.50,
  'static_A_moving_ego_image_change':a['diameter_m']<=.20 and e['path_m']>=.50 and diam>=8,
  'protected':{},'evidence_semantics':'same B instance / GT cuboid surface16 grid with actual Y SAM2 pixels; approximate geometric correspondence, not verified texture evidence'}
 for tok,ms in (protected or {}).items():
  ratios=[];centres=[];hidden=[];visible=[]
  for i,(m,h) in enumerate(zip(ms,holes)):
   yy,xx=np.where(m);hy,hx=np.where(m&h);ratios.append(float(len(hx)/max(1,len(xx))))
   centres.append([(hx.mean()-xx.min())/max(1,np.ptp(xx)),(hy.mean()-yy.min())/max(1,np.ptp(yy))] if len(hx)>=20 and len(xx) else None)
   if protected_actors and tok in protected_actors:
    hd,vi=cells(frames[i],protected_actors[tok][i],m,h);hidden.append(hd);visible.append(vi)
  cc=np.array([c for c in centres if c is not None]);span=np.ptp(cc,axis=0).tolist() if len(cc)>1 else [0.,0.]
  support=[]
  for i,hd in enumerate(hidden):
   other=set().union(*(set(v) for j,v in enumerate(visible) if j!=i));n=sum(hd.values())
   support.append(sum(v for k,v in hd.items() if k in other)/n if n else None)
  defined=[s for s in support if s is not None]
  row['protected'][tok]={'occlusion_fractions':ratios,'occlusion_range':max(ratios)-min(ratios),
   'occlusion_centres_in_B_bbox':centres,'occlusion_centre_span':span,
   'sweep_over_B':max(span)>=.20 and len(cc)>=3,'visibility_transition':max(ratios)-min(ratios)>=.15,
   'hidden_pixels_with_other_frame_cuboid_cell_support':support,
   'approx_other_frame_support_mean':float(np.mean(defined)) if defined else None,
   'no_geometric_other_frame_support':bool(defined and max(defined)==0),
   'B_world_motion':motion([b['translation'] for b in protected_actors[tok]]) if protected_actors and tok in protected_actors else None}
 row['sweep_over_any_B']=any(v['sweep_over_B'] for v in row['protected'].values())
 row['visibility_transition_any_B']=any(v['visibility_transition'] for v in row['protected'].values())
 return row
