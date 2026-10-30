import json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030');o=R/'visibility_control_r1'
t=np.load(o/'target_world_points.npz');P=t['xyz'];frames=t['anchor_frame'];lid=np.load(o/'lidar_replayed.npz');road=np.isin(lid['semantic'],[1,2]);ld,_=cKDTree(lid['xyz'][road]).query(P);lidcounts={}
for tol in [.05,.1,.25,.5]:
 c=np.zeros(len(P),int)
 for f in range(16):
  use=road&(lid['frame']==f)
  d,_=cKDTree(lid['xyz'][use]).query(P);c+=d<=tol
 lidcounts[str(tol)]=c
summary={'schema':'v43_visibility_control_final_r1','human_verdict':None,'train_input':False,'point_weighting':'counts are old-anchor revealed-road pixel samples; overlaps across anchors not deduplicated; not area-weighted','scope':'geometric evidence sufficiency control; visibility cannot prove appearance identifiability or clean unshadowed radiance','selected_trajectory':'hybrid8_z3.5','anchors':{},'visibility_threshold_m':.005,'lidar_proximity_thresholds_m':[.05,.1,.25,.5],'shadow_scope':'32sample shadow toggles at 0/8/15 only; per-pixel difference is finite-sample effect-contrast proxy, never shadow GT or training label','environment_scope':'all evaluated target points have actual independent road/lane firsthit. Simulator nohit environment is not a road surface; learned environment/alpha fill remains a separate rendering diagnosis.'}
for name in ['original','hybrid8_z3.5']:
 v=np.load(o/(name+'_visibility.npz'));visible=v['visible'];cams=json.loads((o/(name+'_cameras.json')).read_text())['frames'];orig=np.array([x['camera_to_world'] for x in cams])[:,:3,3];rays=orig[:,None,:]-P[None,:,:];rays/=np.linalg.norm(rays,axis=-1,keepdims=True);maxang=np.zeros(len(P));maxbase=np.zeros(len(P))
 for i in range(16):
  for j in range(i):
   use=visible[i]&visible[j];angle=np.degrees(np.arccos(np.clip((rays[i]*rays[j]).sum(-1),-1,1)));maxang[use]=np.maximum(maxang[use],angle[use]);maxbase[use]=np.maximum(maxbase[use],np.linalg.norm(orig[i]-orig[j]))
 np.savez_compressed(o/(name+'_support.npz'),observed_frames=visible.sum(0),max_baseline_angle_deg=maxang,max_baseline_m=maxbase,lidar_nearest_distance_m=ld,**{'lidar_frame_count_'+k:c for k,c in lidcounts.items()})
 for f in [0,8,15]:
  use=frames==f;cnt=visible.sum(0)[use];multi=cnt>=2;angles=maxang[use];baselines=maxbase[use]
  item={'pixels':int(use.sum()),'observed_0_frames':int((cnt==0).sum()),'observed_1_frame':int((cnt==1).sum()),'observed_2plus_frames':int((cnt>=2).sum()),'observed_1plus_fraction':float((cnt>0).mean()),'median_visible_frames':float(np.median(cnt)),'baseline_angle_2plus_deg_quantiles_10_50_90':np.quantile(angles[multi],[.1,.5,.9]).tolist() if multi.any() else [],'baseline_m_2plus_quantiles_10_50_90':np.quantile(baselines[multi],[.1,.5,.9]).tolist() if multi.any() else [],'angle_10deg_and_2plus_fraction':float(((angles>=10)&multi).mean()),'original_sparse_lidar_support':{k:{'supported_fraction':float((c[use]>0).mean()),'supported_2plus_sensor_frames_fraction':float((c[use]>=2).mean())} for k,c in lidcounts.items()},'original_lidar_nearest_m_quantiles_10_50_90':np.quantile(ld[use],[.1,.5,.9]).tolist()}
  if name=='original':
   pe=v['nearest_pixel_world_error'][:,use];vv=visible[:,use];item['visible_projected_point_to_nearest_actual_pixel_world_error_m_p50_p90_p99']=np.quantile(pe[vv],[.5,.9,.99]).tolist();item['visible_ray_support_with_nearest_pixel_within10cm_fraction']=float((vv&(pe<=.1)).any(0).mean())
  summary['anchors'].setdefault(str(f),{})[name]=item
  bg=Image.open(R/f'dataset_r2/privileged/references/delete/shadow_on/rgb/{f:06d}.png').convert('RGB');arr=np.asarray(bg).copy();yx=t['yx'][use];colors=np.zeros((len(cnt),3),np.uint8);colors[cnt==0]=[240,40,30];colors[cnt==1]=[255,205,30];colors[cnt>=2]=[0,210,80];arr[yx[:,0],yx[:,1]]=colors;im=Image.fromarray(arr);can=Image.new('RGB',(480,345),'white');can.paste(im,(0,25));ImageDraw.Draw(can).text((4,5),f'{name} anchor{f}: red0 yellow1 green2+ frames',fill='black');can.save(o/f'{name}_anchor{f:06d}_coverage.png')
# Limited available effect proxy in the three original views; do not call clean support.
proxy_support=np.zeros(len(P),int);eligible=np.zeros(len(P),int);v=np.load(o/'original_visibility.npz')['visible'];cams=json.loads((o/'original_cameras.json').read_text())['frames'];proxy_info=[]
for f in [0,8,15]:
 a=np.asarray(Image.open(R/f'dataset_r2/privileged/references/none/shadow_on/rgb/{f:06d}.png'),dtype=float)/255;b=np.asarray(Image.open(R/f'dataset_r2/privileged/references/none/shadow_off/rgb/{f:06d}.png'),dtype=float)/255;d=np.max(abs(a-b),-1);z=np.load(R/f'dataset_r2/privileged/original/{f:06d}.npz');roadmask=np.isin(z['semantic_id'],[1,2]);mask=roadmask&(d>.02)
 c=cams[f];w=np.array(c['world_to_camera']);k=np.array(c['K']);q=P@w[:3,:3].T+w[:3,3];uv=q[:,:2]/q[:,2:]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]];x=np.clip(np.floor(uv[:,0]).astype(int),0,479);y=np.clip(np.floor(uv[:,1]).astype(int),0,319);eligible+=v[f];proxy_support+=v[f]&mask[y,x];proxy_info.append({'frame':f,'original_visible_road_lane_pixels':int(roadmask.sum()),'effect_contrast_proxy_pixels_threshold02':int(mask.sum())})
summary['shadow_proxy_source_regions']=proxy_info
for f in [0,8,15]:
 use=frames==f;summary['anchors'][str(f)]['limited_shadow_proxy']={'target_fraction_visible_in_any_of_3_proxy_keyframes':float((eligible[use]>0).mean()),'target_fraction_with_any_effect_contrast_proxy_hit':float((proxy_support[use]>0).mean()),'not_available':'no off-shadow render for other 13 frames; a below-threshold proxy is not proof of unshadowed radiance'}
sel=json.loads((o/'hybrid8_z3.5_cameras.json').read_text())['frames'];old=json.loads((o/'original_cameras.json').read_text())['frames'];parity={str(f):float(np.max(abs(np.array(sel[f]['camera_to_world'])-np.array(old[f]['camera_to_world'])))) for f in [0,8,15]};assert max(parity.values())==0
summary['trajectory_preflight']={'status':'PASS_GEOMETRIC_COVERAGE_ONLY','camera_count':16,'unchanged_anchor_camera_matrix_max_abs':parity,'trajectory_sha256':hashlib.sha256((o/'hybrid8_z3.5_cameras.json').read_bytes()).hexdigest(),'source_world_unchanged':True,'same_time_object_motion':True,'visibility_1plus_all_target_fraction':1.0,'visibility_2plus_all_target_fraction':float((np.load(o/'hybrid8_z3.5_visibility.npz')['visible'].sum(0)>=2).mean()),'not_completed':['new trajectory actual RGB rendering','new trajectory actual sparse LiDAR acquisition','native initializer integration smoke','reconstruction'],'gauge_requirement':'use common original world-to-model gauge and original spatial_lr_scale/radius; hybrid PCA would otherwise change normalization','trajectory_interpretation':'16 single-camera observations at original timestamps; 13 varied viewpoints and 3 original anchors, not additional cameras per frame; diagnostic acquisition can jump viewpoints, not physically constrained car-mounted path'}
summary['sensor_validation']=json.loads((o/'sensor_validation.json').read_text())
(o/'visibility_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'preflight':summary['trajectory_preflight'],'anchors':summary['anchors']},indent=2))
