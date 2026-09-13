"""固定内部窗口复核；仅消费既有预测，INPUT 尺度与局部平面分表。"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from motion_proj.worldsim_v81.geometry import transform,apply,project,remove_boxes,plane_fit,coverage,depth_metrics,roi_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
A=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2');P=A.parent.parent/'WS-V81-GPU-P2-01';O=A.parent.parent/'WS-V81-CLOSE-01'
idx=json.loads((A/'index.json').read_text());reg=json.loads((O/'registration.json').read_text());rois={r['roi_id']:r for r in map(json.loads,(A/'v81_roi_registry.jsonl').read_text().splitlines())}
old=list(map(json.loads,(P/'evaluation/metrics.jsonl').read_text().splitlines()));output=[];refs=[];clouds={}
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b),0,1))))
def metric(pred,ref):
 m=depth_metrics(pred,ref['depth_z']);ok=np.isfinite(pred)&(pred>0);rays=np.c_[ref['uv'],np.ones(len(pred))]@np.linalg.inv(ref['K']).T;pts=rays[ok]*pred[ok,None];fit=plane_fit(pts)
 m.update(normal_error_deg=angle(fit['normal'],ref['normal']) if fit else None,plane_bending_p95_m=float(np.quantile(fit['residual'],.95)) if fit else None,point_to_plane_m=float(abs((pts-ref['center'])@ref['normal']).mean()) if len(pts) else None)
 return m
def sample(depth,uv,A):
 q=np.c_[uv,np.ones(len(uv))]@np.array(A).T
 return map_coordinates(depth,[q[:,1],q[:,0]],order=1,mode='constant',cval=np.nan)
def input_cloud(m):
 s=m['target_sample']
 if s not in clouds:
  d=idx['sample_data'][s]['LIDAR_TOP'];pts=np.fromfile(Path(idx['root'])/d['filename'],np.float32).reshape(-1,5)[:,:3]
  clouds[s]=remove_boxes(apply(pts,transform(idx['poses'][d['ego_pose_token']])@transform(idx['calibrated'][d['calibrated_sensor_token']])),idx['annotations'].get(s,[]))
 return clouds[s]
for rid in reg['existing_roi_ids']:
 roi=rois[rid];raw=dict(np.load(A/'reference_geometry'/f'{rid}.npz'));box=np.array(roi['box'])+np.array([32,32,-32,-32]);keep=roi_mask(raw['uv'],raw['depth_z'],box)
 ref={**raw,**{k:raw[k][keep] for k in ['uv','xyz','depth_z','source_scan']}};fit=plane_fit(ref['xyz']);scans=np.unique(ref['source_scan']);angles=[]
 if fit:
  ref.update(normal=fit['normal'],center=fit['center'])
  for s in scans:
   sub=plane_fit(ref['xyz'][ref['source_scan']!=s])
   if sub:angles.append(angle(sub['normal'],fit['normal']))
 spread=max(angles) if len(angles)==len(scans) and len(scans)>=2 else None
 rr={'roi_id':rid,'log':roi['log'],'interior_box':box.tolist(),'original_points':len(raw['uv']),'points':len(ref['uv']),'scans':len(scans),'coverage':coverage(ref['uv'],box),'rms_m':fit['rms'] if fit else None,'inlier_fraction':fit['inlier_fraction'] if fit else None,'leave_scan_normal_spread_deg':spread,'reference_input_samples_disjoint':str(raw['input_sample']) not in raw['reference_samples'].tolist(),'human_verdict':None}
 rr['reference_pass']=bool(fit and len(ref['uv'])>=30 and len(scans)>=2 and rr['coverage']>=.25 and fit['rms']<=.12 and fit['inlier_fraction']>=.85 and spread is not None and spread<=5)
 refs.append(rr);m=json.loads((P/'input_manifests'/f'{roi["window_id"]}.json').read_text());target=next(v for v in m['views'] if v['camera']==roi['camera']);world=input_cloud(m);uv,z,xyz=project(world,np.array(target['world_from_camera']),np.array(target['K']))
 pi=roi_mask(uv,z,box);pf=plane_fit(xyz[pi]);plane_pred=np.full(len(ref['uv']),np.nan)
 if pf:
  rays=np.c_[ref['uv'],np.ones(len(ref['uv']))]@np.linalg.inv(ref['K']).T;plane_pred=(pf['center']@pf['normal'])/(rays@pf['normal'])
 plane_metrics=metric(plane_pred,ref) if fit else {}
 rr.update(input_local_plane_points=int(pi.sum()),input_local_plane_rms_m=pf['rms'] if pf else None,input_local_plane=plane_metrics,input_plane_role='VISUAL_PLUS_TARGET_INPUT_LIDAR_PLANE_EXTRA_INFORMATION; fitted before heldout evaluation')
 exclude=roi_mask(uv,z,np.array(roi['box'])+[-16,-16,16,16]);outside=world[~exclude];panels={}
 for row in old:
  if row['roi_id']!=rid or row['texture_case'] or (row['output_key']!='full6' and not row['anchor_camera']):continue
  rp=Path(row['result_path']);result=json.loads(rp.read_text());ratios=[]
  for v in result['views']:
   camera=next(q for q in m['views'] if q['camera']==v['camera']);ou,oz,_=project(outside,np.array(camera['world_from_camera']),np.array(camera['K']));ok=roi_mask(ou,oz,[0,0,1600,900])&(oz<80);ou=ou[ok];oz=oz[ok]
   cell=(ou[:,1]//8).astype(int)*200+(ou[:,0]//8).astype(int);order=np.argsort(oz);_,first=np.unique(cell[order],return_index=True);chosen=order[first];ou=ou[chosen];oz=oz[chosen]
   pred=sample(np.load(rp.parent/(v['camera']+'_depth_z_m.npy')),ou,v['original_to_network_pixel_center']);ok=np.isfinite(pred)&(pred>0);ratios.extend((oz[ok]/pred[ok]).tolist())
  scale=float(np.median(ratios)) if len(ratios)>=100 else None
  v=next(v for v in result['views'] if v['camera']==roi['camera']);depth=np.load(rp.parent/(roi['camera']+'_depth_z_m.npy'));pred=sample(depth,ref['uv'],v['original_to_network_pixel_center']);before=metric(pred,ref) if fit else {};after=metric(pred*scale,ref) if fit and scale else {}
  depth_thresh=max(.5,.05*float(np.median(ref['depth_z'])),3*fit['rms']) if fit else None;normal_thresh=max(10,3*spread) if spread is not None else None
  reasons=[];control_created=[]
  if rr['reference_pass'] and after.get('coverage',0)>=.9:
   for axis,key,thresh in [('DEPTH','mae_m',depth_thresh),('NORMAL','normal_error_deg',normal_thresh),('BENDING','plane_bending_p95_m',.2)]:
    if after[key] is not None and after[key]>thresh:
     (reasons if before.get(key) is not None and before[key]>thresh else control_created).append(axis)
  r={'roi_id':rid,'log':roi['log'],'method':row['method'],'output_key':row['output_key'],'variant':row['variant'],'selection_role':'POSTHOC_DISCOVERY_FIXED_INTERIOR','original_full_roi_mae_m':row['mae_m'],'before':before,'after_input_global_scale':after,'scale':scale,'fit_projection_records':len(ratios),'reference_pass':rr['reference_pass'],'residual_axes':reasons,'control_created_axes':control_created,'depth_threshold_m':depth_thresh,'normal_threshold_deg':normal_thresh,'input_local_plane':plane_metrics,'scale_input_role':'VISUAL_PLUS_OUTSIDE_TARGET_INPUT_LIDAR; not same-information ranking','heldout_used_for_fit':False,'result_path':str(rp)};output.append(r)
  if row['output_key']=='full6':panels[row['method']]=(r,pred,pred*scale if scale else pred*float('nan'))
 if fit and len(panels)==2:
  fig,axs=plt.subplots(2,4,figsize=(14,6));rgb=Image.open(target['image']);crop=rgb.crop(tuple(box));xy=ref['uv']-box[:2]
  for i,method in enumerate(['dvgt','vggt']):
   row,bef,aft=panels[method];axs[i,0].imshow(crop);axs[i,0].scatter(xy[:,0],xy[:,1],s=6,c=ref['source_scan'],cmap='tab10');axs[i,0].set_title(f'{method}: {len(xy)} held-out points / {len(scans)} scans')
   for j,(pred,label,metrics) in enumerate([(bef,'Camera/native metric',row['before']),(aft,'+ Outside-ROI INPUT scale',row['after_input_global_scale']),(plane_pred,'Target INPUT LiDAR plane',plane_metrics)],1):
    sc=axs[i,j].scatter(xy[:,0],xy[:,1],c=np.abs(pred-ref['depth_z']),vmin=0,vmax=2,cmap='magma',s=10);axs[i,j].set_title(f'{label}\nMAE {metrics.get("mae_m",float("nan")):.3f} m | normal {metrics.get("normal_error_deg",float("nan")):.1f} deg',fontsize=9);fig.colorbar(sc,ax=axs[i,j],fraction=.04)
   for ax in axs[i]:ax.set_xlim(0,crop.width);ax.set_ylim(crop.height,0);ax.set_xticks([]);ax.set_yticks([])
  fig.suptitle(rid+f' | fixed 32px inset | reference pass={rr["reference_pass"]}',fontsize=11);fig.tight_layout();fig.savefig(O/'figures'/f'residual_{rid}.png',dpi=145);plt.close(fig)
 print(json.dumps({'roi':rid,'reference':rr,'full6':[r for r in output if r['roi_id']==rid and r['output_key']=='full6']}),flush=True)
(O/'residual_reference.json').write_text(json.dumps(refs,indent=2));(O/'residual_metrics.json').write_text(json.dumps(output,indent=2))
