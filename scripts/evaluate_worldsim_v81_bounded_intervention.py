"""新三日志的同目标度量锚点协议：各变体读取完全相同的当前 LiDAR 投影集合。"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from motion_proj.worldsim_v81.geometry import project,plane_fit,roi_mask,depth_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
O=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01');A=O/'av2_atlas';protocol=json.loads((A/'final_intervention_protocol.json').read_text());rows=[];pairs=[]
def metric(p,r):
 ok=np.isfinite(p)&(p>0);rays=np.c_[r['uv'],np.ones(len(p))]@np.linalg.inv(r['K']).T;pts=rays[ok]*p[ok,None];m=depth_metrics(p,r['depth_z']);normal=None;bending=None;shape=None
 if len(pts)>=12:
  center=pts.mean(0);_,_,vh=np.linalg.svd(pts-center,full_matrices=False);normal=float(np.degrees(np.arccos(np.clip(abs(vh[-1]@r['normal']),0,1))));bending=float(np.quantile(abs((pts-center)@vh[-1]),.95));d=np.log(p[ok]/r['depth_z'][ok]);shape=float(np.sqrt(np.mean((d-d.mean())**2)))
 m.update(normal_error_deg=normal,plane_bending_p95_m=bending,log_shape_rmse=shape);return m
def sample(depth,uv,A):
 q=np.c_[uv,np.ones(len(uv))]@np.array(A).T;return map_coordinates(depth,[q[:,1],q[:,0]],order=1,mode='constant',cval=np.nan)
for r in protocol['selected']:
 ref=dict(np.load(r['reference']));m=json.loads(Path(r['manifest']).read_text());v=m['views'][0];world=np.load(m['input_lidar_world']);uv,z,xyz=project(world,np.array(v['world_from_camera']),np.array(v['K']));W,H=v['size'];box=np.array(r['box']);outside=roi_mask(uv,z,[0,0,W,H])&(z<80)&~roi_mask(uv,z,box+[-16,-16,16,16]);ids=np.flatnonzero(outside);cell=(uv[ids,1]//8).astype(int)*((W+7)//8)+(uv[ids,0]//8).astype(int);order=np.argsort(z[ids]);_,first=np.unique(cell[order],return_index=True);ids=ids[order[first]]
 np.savez_compressed(A/f'{r["roi_id"]}_fixed_input_scale_points.npz',indices_after_static_filter=ids,uv=uv[ids],depth_z=z[ids],input_sample=np.array(m['target_sample']),heldout_used_for_fit=np.array(False))
 pf=plane_fit(xyz[roi_mask(uv,z,box)]);rays=np.c_[ref['uv'],np.ones(len(ref['uv']))]@np.linalg.inv(ref['K']).T;plane=(pf['center']@pf['normal'])/(rays@pf['normal']) if pf else np.full(len(rays),np.nan);plane_m=metric(plane,ref);preds={}
 for method in ['dvgt','vggt']:
  for variant in ['rich','removed','restored']:
   folder=O/'new_predictions'/method/r['roi_id']/variant;result=json.loads((folder/'result.json').read_text());view=result['views'][0];native_unit=result.get('depth_unit')=='native_relative';depth=np.load(folder/(r['camera']+('_depth_z_native.npy' if native_unit else '_depth_z_m.npy')));pred=sample(depth,ref['uv'],view['original_to_network_pixel_center']);fit_pred=sample(depth,uv[ids],view['original_to_network_pixel_center']);ok=np.isfinite(fit_pred)&(fit_pred>0);scale=float(np.median(z[ids][ok]/fit_pred[ok])) if ok.sum()>=100 else None;scaled=pred*scale if scale else np.full(len(pred),np.nan);after=metric(scaled,ref);before=metric(pred,ref)
   if native_unit:
    before={k:before[k] for k in ['coverage','normal_error_deg','log_shape_rmse']};before['absolute_depth']='UNIDENTIFIED_FROM_SINGLE_RGB'
   row={'roi_id':r['roi_id'],'log':r['log'],'method':method,'variant':variant,'role':'NEW_TO_V81_DISCOVERY_VIEW_DIAGNOSTIC','reference_points':len(pred),'reference_scans':len(np.unique(ref['source_scan'])),'fixed_input_scale_projection_records':len(ids),'valid_fit_records':int(ok.sum()),'scale':scale,'before':before,'after_visual_plus_input_lidar_scale':after,'input_local_plane_extra_information':plane_m,'input_plane_points':int(roi_mask(uv,z,box).sum()),'native_unit':result.get('depth_unit','meters_by_native_or_camera_scale'),'heldout_used_for_fit':False,'pure_visual_rank':False,'gpu_peak_gib':result['peak_gpu_allocated_gib'],'elapsed_forward_s':result['elapsed_s'],'human_verdict':None};rows.append(row);preds[(method,variant)]=scaled
  mm={x['variant']:x for x in rows if x['roi_id']==r['roi_id'] and x['method']==method};rich=mm['rich']['after_visual_plus_input_lidar_scale'];sparse=mm['removed']['after_visual_plus_input_lidar_scale'];restore=mm['restored']['after_visual_plus_input_lidar_scale'];eligible=all(q['coverage']>=.9 and q['mae_m'] is not None and q['normal_error_deg'] is not None for q in [rich,sparse,restore]) and len(set(q['valid_fit_records'] for q in mm.values()))==1;dt=sparse['mae_m']-(rich['mae_m']+restore['mae_m'])/2 if eligible else None;dn=sparse['normal_error_deg']-(rich['normal_error_deg']+restore['normal_error_deg'])/2 if eligible else None;threshold=max(.5,.05*float(np.median(ref['depth_z'])));axes=[]
  if eligible and dt>.25 and sparse['mae_m']>threshold:axes.append('DEPTH')
  if eligible and dn>5 and sparse['normal_error_deg']>10:axes.append('NORMAL')
  diff=abs(preds[(method,'restored')]-preds[(method,'rich')]);pairs.append({'roi_id':r['roi_id'],'log':r['log'],'method':method,'eligible':eligible,'status':'COMPARABLE' if eligible else 'POINT_FRAME_OR_COVERAGE_CONTRACT_UNRESOLVED_NOT_SCIENTIFIC_FAILURE','removed_minus_rich_mae_m':dt,'removed_minus_rich_normal_deg':dn,'depth_residual_threshold_m':threshold,'prespecified_degradation_axes':axes,'restored_vs_rich_max_depth_difference_m':float(np.nanmax(diff)) if np.isfinite(diff).any() else None,'future_parallax_deg':r['geometry'][2]['median_parallax_deg']})
 fig,axs=plt.subplots(2,4,figsize=(13,6));crop=Image.open(v['image']).crop(tuple(box));xy=ref['uv']-box[:2]
 for i,method in enumerate(['dvgt','vggt']):
  axs[i,0].imshow(crop);axs[i,0].scatter(xy[:,0],xy[:,1],c=ref['source_scan'],cmap='tab10',s=1);axs[i,0].set_title(f'{method} | {len(xy)} held-out pts\nINPUT local-plane MAE={plane_m["mae_m"]:.3f}m',fontsize=9)
  for j,var in enumerate(['rich','removed','restored'],1):
   rr=next(x for x in rows if x['roi_id']==r['roi_id'] and x['method']==method and x['variant']==var);metrics=rr['after_visual_plus_input_lidar_scale'];sc=axs[i,j].scatter(xy[:,0],xy[:,1],c=abs(preds[(method,var)]-ref['depth_z']),vmin=0,vmax=2,cmap='magma',s=3);fig.colorbar(sc,ax=axs[i,j],fraction=.04);label=f'{var}: MAE={metrics["mae_m"]:.3f}m\nnormal={metrics["normal_error_deg"]:.1f}deg' if metrics['mae_m'] is not None and metrics['normal_error_deg'] is not None else f'{var}: depth/scale unavailable\nnot a validated model failure';axs[i,j].set_title(label,fontsize=9)
  for ax in axs[i]:ax.set_xlim(0,crop.width);ax.set_ylim(crop.height,0);ax.set_xticks([]);ax.set_yticks([])
 fig.suptitle(r['roi_id']+' | Same outside-target INPUT LiDAR records for all variants',fontsize=11);fig.tight_layout();fig.savefig(O/'figures'/f'intervention_{r["roi_id"]}.png',dpi=150);plt.close(fig)
 print(json.dumps([p for p in pairs if p['roi_id']==r['roi_id']]),flush=True)
summary={'logs':len(protocol['selected']),'jobs':len(rows),'all_valid_fit_records_identical_per_target':all(len(set(x['valid_fit_records'] for x in rows if x['roi_id']==r['roi_id']))==1 for r in protocol['selected']),'models':{},'role':'DISCOVERY_NOT_INDEPENDENT_CONFIRMATION','old_jobs_repeated':0}
for method in ['dvgt','vggt']:
 pp=[p for p in pairs if p['method']==method];common=set(['DEPTH','NORMAL'])
 for p in pp:common &= set(p['prespecified_degradation_axes'])
 summary['models'][method]={'n_logs':len(pp),'comparable_logs':sum(p['eligible'] for p in pp),'repeatable_axis_all_three_logs':sorted(common),'logs_with_any_degradation_axis':sum(bool(p['prespecified_degradation_axes']) for p in pp),'restoration_max_difference_m':max(p['restored_vs_rich_max_depth_difference_m'] or 0 for p in pp)}
 model_rows=[r for r in rows if r['method']==method]
 summary['models'][method]['valid_fit_records_identical_per_target']=all(len(set(r['valid_fit_records'] for r in model_rows if r['roi_id']==rid))==1 for rid in {r['roi_id'] for r in model_rows})
(O/'new_intervention_metrics.json').write_text(json.dumps(rows,indent=2));(O/'new_intervention_pairs.json').write_text(json.dumps(pairs,indent=2));(O/'new_intervention_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
