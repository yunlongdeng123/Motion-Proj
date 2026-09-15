import pathlib,json,sys,numpy as np,torch,ijson,shutil,tarfile
from geometry_contract import R,P
sys.path.insert(0,str(P/'motion_proj'))
from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex,_transform
from motion_proj.worldsim_v73.native_data import interpolate_pose
torch.set_num_threads(1);owner='204704542f8642dc8ab046ffbd70e0c5';scene='scene-0520';models=['vggt','omega512','dvgt1','pi3x'];out=R/'white_witness';out.mkdir(exist_ok=True)
arr={m:np.load(R/'evaluation_bgscale'/m/scene/'twelve'/owner/'cal_build_1.0_rays.npz') for m in models}
base=arr['vggt'];common=np.logical_and.reduce([(a['first']<a['ranges']-.2)&a['build_supported'] for a in arr.values()]);target=base['origins']+base['directions']*base['ranges'][:,None]
cases=json.loads((R/'cohort.json').read_text());entry=next(x for x in cases if x['owner']==owner);D=P/'runs/worldsim_v73/WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2';c=torch.load(D/entry['file'],weights_only=False,map_location='cpu');size=np.asarray(c['size_lwh_m']);low=target[:,2]<-size[2]/2+.10;body=~low
ids=np.flatnonzero(common&low);rid=int(ids[len(ids)//2]) if len(ids) else int(np.flatnonzero(common)[0])
# 稳定规则：共同早交且近底面集合的中间索引；不是挑最大gap。
rows=[]
for m,a in arr.items():
 p=R/'evaluation_bgscale'/m/scene/'twelve'/owner;mesh=np.load(p/'primary_surface.npz');tri=mesh['vertices'][mesh['faces'][int(a['face_ids'][rid])]];o=a['origins'][rid];d=a['directions'][rid];rg=a['ranges'][rid]
 vertex_gaps=rg-(tri-o)@d;edges=[np.linalg.norm(tri[i]-tri[(i+1)%3]) for i in range(3)];primary=(a['first']<a['ranges']-.2)&a['build_supported']
 controls={}
 for protocol in ['native_base','native_build','cal_base','cal_build_1.0','cal_build_0.9','cal_build_0.75','cal_build_0.5']:
  q=np.load(p/(protocol+'_rays.npz'));t=float(q['first'][rid]);controls[protocol]={'first_m':t if np.isfinite(t) else None,'early_gap_m':float(q['ranges'][rid]-t) if np.isfinite(t) else None}
 rows.append({'method':m,'first_m':float(a['first'][rid]),'range_m':float(rg),'gap_m':float(rg-a['first'][rid]),'triangle_max_edge_m':float(max(edges)),'minimum_vertex_forward_gap_m':float(vertex_gaps.min()),'supported_early_near_bottom':int((primary&low).sum()),'supported_early_above_bottom':int((primary&body).sum()),'all_supported_early':int(primary.sum()),'controls_for_selected_ray':controls})
 np.savez_compressed(out/(m+'.npz'),triangle=tri,origin=o,direction=d,range=rg,first=a['first'][rid])
index=NuScenesCameraIndex(P/'data/worldsim_v4/drivestudio_raw_trainval');trajectory=[]
with (index.metadata_root/'sample_annotation.json').open('rb') as f:
 for row in ijson.items(f,'item'):
  if row['instance_token']==owner:trajectory.append((int(index.sample_by_token[row['sample_token']]['timestamp']),_transform(row['translation'],row['rotation'])))
trajectory.sort(key=lambda x:x[0]);frameid=int(base['frame_ids'][rid]);frame=next(f for f in c['rays'] if f['sample_index']==frameid);sample=frame['sample_id'];channel='CAM_FRONT_RIGHT'
sd=index.sample_data[index.data_by_sample_channel[(sample,channel)]];cal=index.calibrated[sd['calibrated_sensor_token']];ego=index.ego_poses[sd['ego_pose_token']];wc=_transform(ego['translation'],ego['rotation'])@_transform(cal['translation'],cal['rotation']);wa=interpolate_pose(trajectory,int(sd['timestamp']));ca=np.linalg.inv(wc)@wa
shutil.copy2(index.dataset_root/sd['filename'],out/'query_rgb.jpg');np.savez_compressed(out/'query_projection.npz',camera_from_actor=ca,K=cal['camera_intrinsic'],size=size,build_points=np.asarray(c['points_actor_m']),gt_points=target,selected_point=target[rid])
record={'owner':owner,'scene':scene,'selected_ray':rid,'selection':'median index among all four models common early, BUILD-supported, near annotated bottom; preexisting white case; discovery not confirmation','selected_frame':frameid,'selected_gt_point_actor':target[rid].tolist(),'annotated_bottom_actor_z':float(-size[2]/2),'common_early_indices':np.flatnonzero(common).tolist(),'common_low_indices':np.flatnonzero(common&low).tolist(),'common_above_bottom_indices':np.flatnonzero(common&body).tolist(),'reference_ownership':'Annotation box +0.1m proxy, not semantic instance segmentation. Ground/underbody returns must not be called confirmed car-body returns.','interpretation':'Opaque readout blocks a measured beam ending near/below annotated vehicle bottom. Ground material and exact surface identity not independently segmented.','query_rgb':sd['filename'],'query_rgb_time_us':sd['timestamp'],'query_lidar_time_us':frame['timestamp_us'],'query_rgb_only_for_visualization':True,'rows':rows}
(out/'audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
with tarfile.open(R/'white_witness.tar','w') as t:t.add(out,arcname='white_witness')
