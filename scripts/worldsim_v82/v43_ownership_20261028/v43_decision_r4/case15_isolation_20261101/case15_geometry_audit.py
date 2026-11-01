import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import pathlib,json,math,sys,collections,gc,time
import numpy as np
BASE=pathlib.Path(r'C:\Users\dengyunlong\Documents\Codex\2026-10-31\g\work\actor_diagnostics_20261101')
from scipy.spatial import cKDTree
W=pathlib.Path(__file__).resolve().parent;H=W/'case15';C=BASE/'root/gs_cache';M=BASE/'media/generated';selected=json.loads((H/'shortlist.json').read_text(encoding='utf8'));out={'schema':'case15_full_sequence_geometric_isolation_v1','rows':[],'shared_geometric_pool_rows':[],'new_native_render_queries':0,'limits':['Complete static/dynamic original Gaussian arrays are used, but this is geometry/footprint proximity, not alpha*T ray responsibility or owner GT.','Expanded OBB and effective-opacity thresholds are descriptive sensitivity only; no actor is admitted or rejected by them.','3sigma ellipsoid enclosing AABB intersection is conservative, not exact surface, ray visibility or collision.','A shared original row entering two diagnostic pools is geometric ambiguity, not proof that it serves two visible surfaces.']}
if '--only-legacy' in sys.argv:
 out=json.loads((H/'geometry_audit.json').read_text(encoding='utf8'))
 selected=[r for r in selected if r['scene_key']=='waymo-train-008']
elif '--skip-legacy' in sys.argv:selected=[r for r in selected if r['scene_key']!='waymo-train-008']
for context in dict.fromkeys(r['context_id'] for r in selected):
 cases=[r for r in selected if r['context_id']==context and r['scene_checkpoint_recorded']]
 if not cases:continue
 media=json.loads((M/'scenes'/context/'media.json').read_text(encoding='utf8'))
 if cases[0]['scene_key']=='waymo-train-008':
  meta=json.loads((H/'legacy_export/receipt.json').read_text(encoding='utf8'));T=np.array(meta['local_from_world_metric']);paths={b:H/'legacy_export'/(b+'.npz') for b in ['static','dynamic']};prep=json.loads((H/'remote/evidence/waymo-train-008_a37/preparation_manifest.json').read_text(encoding='utf8'));modeltime={rec['source_index']:rec['model_time'] for rec in prep['frames']}
 else:
  meta=json.loads((C/(context+'_geometry.json')).read_text(encoding='utf8'));T=np.array(meta['geometry']['local_from_world']);paths={b:C/(context+'_'+b+'.npz') for b in ['static','dynamic']};modeltime={int(k):v['model_time'] for k,v in meta['scene']['frames'].items()}
 assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8)
 models={b:dict(np.load(p)) for b,p in paths.items()};print('scene_loaded',context,{b:len(a['xyz']) for b,a in models.items()},flush=True)
 windows={};pools={};states={};lidars={}
 for r in cases:
  key=r['scene_key']+'_'+r['alias'];w=r['native_window'];windows[key]=(w['first_raw_frame'],w['last_raw_frame']);states[key]={'case':key,'scene_key':r['scene_key'],'alias':r['alias'],'frames':[],'source_full_G_rows':{b:len(a['xyz']) for b,a in models.items()},'all_export_fields_finite':all(np.isfinite(v).all() for a in models.values() for v in a.values()),'model_fields':'xyz, anisotropic scales, rotation, opacity, t, sigma_t, velocity, original per-branch ids; color is excluded from geometry test'};pools[key]={b:set() for b in models};lidars[key]=dict(np.load(M/media['actors'][r['track_id']]['lidar']['npz']))
 for raw in range(min(w[0] for w in windows.values()),max(w[1] for w in windows.values())+1):
  tau=modeltime[raw];cached={}
  for b,a in models.items():
   dt=tau-a['t'].reshape(-1);omega=2*math.pi/float(a['cycle']);xyz=a['xyz']+a['velocity']*(np.sin(dt*omega)/omega)[:,None];marginal=np.exp(-.5*(dt/np.maximum(a['sigma_t'].reshape(-1),1e-30))**2);alpha=a['opacity'].reshape(-1)*marginal;cached[b]=(xyz,alpha,marginal)
  for r in cases:
   key=r['scene_key']+'_'+r['alias'];first,last=windows[key]
   if not first<=raw<=last:continue
   actor=media['actors'][r['track_id']];poses={x['raw']:x for x in actor['trajectory']};pose=poses[raw];R=T[:3,:3]@np.array(pose['orientation_world']).reshape(3,3);center=T[:3,:3]@pose['center_world']+T[:3,3];dim=np.array(pose['dimensions']);points=lidars[key]['actor_local_xyz'][lidars[key]['raw_frame']==raw];fr={'raw':raw,'model_time':tau,'actual_box_LiDAR_rows':len(points),'branches':{}};nearest=[];allscales=[]
   for b,a in models.items():
    xyz,alpha,marginal=cached[b];rmax=3*a['scales'].max(1);distance=np.linalg.norm(xyz-center,axis=1);broad=(distance<=np.linalg.norm(dim)/2+np.maximum(rmax,.87))&(alpha>=.001);idx=np.flatnonzero(broad);local=(xyz[idx]-center)@R
    incenter=np.all(abs(local)<=dim/2+.5,axis=1);rotlocal=np.einsum('ij,njk->nik',R.T,a['rotation'][idx]);radii=3*np.sqrt(((rotlocal*a['scales'][idx,None,:])**2).sum(2));intersect=np.all(abs(local)<=dim/2+radii,axis=1)
    counts={str(t):int((incenter&(alpha[idx]>=t)).sum()) for t in [.001,.01,.1]};pick=idx[incenter&(alpha[idx]>=.01)];fp=idx[intersect&(alpha[idx]>=.01)];pools[key][b].update(int(i) for i in a['ids'][fp]);nearest.append((xyz[pick]-center)@R);allscales.extend(a['scales'][pick].max(1).tolist())
    fr['branches'][b]={'center_in_expanded_OBB_by_alpha':counts,'conservative_3sigma_footprint_OBB_intersections_alpha001':len(fp),'footprint_candidates_with_centers_outside_expanded_OBB':int((intersect&~incenter&(alpha[idx]>=.01)).sum()),'footprint_candidates_temporal_marginal_over_005':int((marginal[fp]>.05).sum())}
   near=np.concatenate(nearest) if nearest else np.empty((0,3))
   if len(near) and len(points):
    dist=cKDTree(near).query(points,workers=1)[0];fr['LiDAR_to_cropped_center_NN_m']={'median':float(np.median(dist)),'p90':float(np.quantile(dist,.9))}
   else:fr['LiDAR_to_cropped_center_NN_m']=None
   fr['cropped_gaussian_max_axis_sigma_m']={'median':float(np.median(allscales)),'p90':float(np.quantile(allscales,.9))} if allscales else None
   states[key]['frames'].append(fr)
  if (raw-min(w[0] for w in windows.values()))%12==0:print('raw',raw,'done',flush=True)
 for key,rec in states.items():
  rec['frozen_diagnostic_original_row_pool']={b:len(p) for b,p in pools[key].items()};rec['pool_semantics']='Union of conservative 3sigma-OBB overlap candidates across original49 frames, effective alpha>=.01. Fixed diagnostics only; not a legal R4 candidate U, H0, owner partition or safe route.'
  rec['NN_median_over_frame_medians_m']=float(np.median([f['LiDAR_to_cropped_center_NN_m']['median'] for f in rec['frames'] if f['LiDAR_to_cropped_center_NN_m']])) if any(f['LiDAR_to_cropped_center_NN_m'] for f in rec['frames']) else None
  dest=H/'row_pools';dest.mkdir(exist_ok=True);np.savez_compressed(dest/(key+'.npz'),**{b:np.array(sorted(p),np.int32) for b,p in pools[key].items()});out['rows'].append(rec)
 keys=list(states)
 for i,k in enumerate(keys):
  for q in keys[i+1:]:out['shared_geometric_pool_rows'].append({'case_a':k,'case_b':q,'shared_static_rows':len(pools[k]['static']&pools[q]['static']),'shared_dynamic_rows':len(pools[k]['dynamic']&pools[q]['dynamic']),'meaning':'cross-time conservative footprint/OBB overlap only; not proof of multi-surface photometric responsibility'})
 (H/'geometry_audit.partial.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');del models,cached,lidars;gc.collect()
(H/'geometry_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('GEOMETRY_COMPLETE',len(out['rows']),flush=True)
