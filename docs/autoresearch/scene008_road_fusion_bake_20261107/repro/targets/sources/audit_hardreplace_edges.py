"""只读CPU机制诊断；不改变锁定目标，不保存10px替代目标，不新GPU。"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt,label,binary_fill_holes
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');RUN=CAM/'scene008_road_fusion_bake_20261107_r1';B=RUN/'targets';OLD=CAM/'scene008_road_seam_contact_20261107_r1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(x):
 x=np.asarray(x,dtype=float)
 return dict(count=int(x.size),mean=float(x.mean()) if x.size else None,p50=float(np.median(x)) if x.size else None,p95=float(np.quantile(x,.95)) if x.size else None,max=float(x.max()) if x.size else None)
def pairs(M,domain):
 ends=[]
 for axis in [0,1]:
  a=np.zeros_like(M);b=np.zeros_like(M)
  if axis==0:
   crossing=(M[:-1]!=M[1:])&domain[:-1]&domain[1:];yy,xx=np.where(crossing);y2=yy+1;x2=xx
  else:
   crossing=(M[:,:-1]!=M[:,1:])&domain[:,:-1]&domain[:,1:];yy,xx=np.where(crossing);y2=yy;x2=xx+1
  first=M[yy,xx];ends.append((np.where(first,yy,y2),np.where(first,xx,x2),np.where(first,y2,yy),np.where(first,x2,xx)))
 return tuple(np.concatenate([v[k] for v in ends]) for k in range(4))
def jumps(rgb,obs,inside_y,inside_x,outside_y,outside_x):
 return ((rgb[inside_y,inside_x]-rgb[outside_y,outside_x])-(obs[inside_y,inside_x]-obs[outside_y,outside_x]))
def world_jumps(rgb,obs,mask,primary):
 xyz=mask['local_xyz'];out={}
 for edge,ta in [('near_x',1),('far_x',1),('lateral_y',0)]:
  prefix='band_' if primary else '';minus=mask[f'{prefix}{edge}_minus'];plus=mask[f'{prefix}{edge}_plus'];bins=np.floor(xyz[...,ta]/.2).astype(int);values=[]
  for bi in np.unique(bins[minus|plus]):
   mm=minus&(bins==bi);pp=plus&(bins==bi)
   if mm.sum()>=3 and pp.sum()>=3:
    v=(rgb-obs)[pp].mean(0)-(rgb-obs)[mm].mean(0)
    values.append(dict(bin=int(bi),minus_count=int(mm.sum()),plus_count=int(pp.sum()),abs_channel_jump=float(abs(v).mean())))
  out[edge]=dict(paired_bins=len(values),matched_minus=sum(v['minus_count'] for v in values),matched_plus=sum(v['plus_count'] for v in values),mean=float(np.mean([v['abs_channel_jump'] for v in values])) if values else None,bins=values)
 return out
def main():
 original_manifest_sha=sha(B/'target_manifest.json');manifest=json.loads((B/'target_manifest.json').read_text());results=[];topology=[]
 for row in manifest['rows']:
  z=np.load(B/row['file']);M=z['fusion_replace_mask'];D=distance_transform_edt(M);w=np.minimum(D/10,1);inside=w[M]
  topology.append(dict(raw=row['raw'],role=row['role'],replace_pixels=int(M.sum()),components=int(label(M)[1]),internal_hole_pixels=int((binary_fill_holes(M)&~M).sum()),
    distance_inside_px=stat(D[M]),tenpx_weight=stat(inside),full_atlas_weight_pixels=int((inside>=1).sum())))
  if row['raw'] not in [128,137]:continue
  raw=row['raw'];obs=np.load(CAM/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz');view=np.load(CAM/f'scene008_blue_global_road1k_20261105_r1/inputs/views/{raw:03d}.npz')
  O=z['observation_rgb'];A=z['direct_rgb'];F=z['fusion_rgb'];C=O+w[:,:,None]*(F-O) # Diagnostic only; never written to target NPZ.
  mask=np.load(OLD/'blue/fixed_masks_raw128.npz') if raw==128 else np.load(RUN/f'red_review/world_masks49_v1/{raw:03d}.npz')
  domain=mask['original_road'] if raw==128 else mask['world_patch_clean_road'];domname='PRIMARY raw128 original-road user bbox' if raw==128 else 'SECONDARY raw137 fixed world patch'
  yi,xi,yo,xo=pairs(M,domain)
  guard_a6=obs['visible'][0]|obs['visible_reviewed'][0]|obs['box_candidate'][0]
  guard_a37=obs['visible'][1]|obs['visible_reviewed'][1]|obs['box_candidate'][1]
  other=view['other_vehicle_guard'].astype(bool);fit=z['fit_mask'];support=z['atlas_support_mask'];geom=support|z['imputed_mask']
  reason=np.full(M.shape,'other',dtype='<U40')
  reason[~fit]='nonroad_or_other_fit_exclusion';reason[~fit&guard_a6]='A6_annotation_guard';reason[~fit&guard_a37]='A37_annotation_guard';reason[~fit&other]='other_vehicle_guard'
  reason[fit&~geom]='chart_four_corner_geometry_edge';reason[fit&geom&~support]='source_qualification_edge';reason[fit&support&~M]='scene_depth_reject'
  types=reason[yo,xo]
  methods={'source_target_A':A,'locked_Poisson_target_B':F,'hypothetical_fixed10px_target_C':C}
  saved={}
  for method,file in [('AA_baseline','baseline_delete.npz'),('native_source400','direct_delete.npz'),('native_fusion400','fusion_delete.npz')]:
   p=RUN/f'results/full49/{raw:03d}/{file}';d=np.load(p);methods[method]=d['rgb'].transpose(1,2,0);saved[method]=dict(path=str(p),sha256=sha(p))
   if method=='AA_baseline':ga=d['group_alpha_T'];nw=d['newroad_alpha_T']
  metrics={}
  for kind in ['ALL']+sorted(set(types)):
   sel=np.ones(len(types),bool) if kind=='ALL' else types==kind
   ry,rx,sy,sx=yi[sel],xi[sel],yo[sel],xo[sel]
   metrics[kind]=dict(edge_count=int(sel.sum()),distinct_inside_pixels=int(len(set(zip(ry.tolist(),rx.tolist())))),
     original_abs_gradient=stat(abs(O[ry,rx]-O[sy,sx]).mean(1)),methods={},
     AA_newroadW_inside=stat(nw[ry,rx]),AA_newroadW_outside=stat(nw[sy,sx]),AA_assetW_outside=stat(ga[0,sy,sx]))
   for name,rgb in methods.items():
    v=jumps(rgb,O,ry,rx,sy,sx)
    metrics[kind]['methods'][name]=dict(residual_jump=stat(abs(v).mean(1)),signed_residual_jump_RGB=v.mean(0).astype(float).tolist() if len(v) else None,
       actual_abs_gradient=stat(abs(rgb[ry,rx]-rgb[sy,sx]).mean(1)))
  Dsel=D[M&domain];weights=w[M&domain]
  results.append(dict(raw=raw,state='DELETE native; targets shared KEEP/DELETE',domain=domname,domain_pixels=int(domain.sum()),
    replace_in_domain=int((M&domain).sum()),cross_mask_edge_metrics=metrics,source_files=saved,
    within_domain_tenpx_weight=stat(weights),full_atlas_weight_in_domain=int((weights>=1).sum()),
    locked_target_registered_world_jumps=world_jumps(F,O,mask,raw==128),
    hypothetical_tenpx_registered_world_jumps=world_jumps(C,O,mask,raw==128),
    tenpx_note='C is CPU-only diagnostic formula, not a saved target or a native rendered candidate. C=O+clip(inside_distance(M)/10,0,1)*(B-O); same existing M, all boundary types including holes.',
    classification_note='chart_four_corner_geometry_edge is the texture-chart/measurement eligibility edge, not a physical first-hit or vehicle surface. Annotation guards are conservative, not confirmed occlusion GT.'))
 assert sha(B/'target_manifest.json')==original_manifest_sha
 out=dict(schema='HARDCOPY_VALIDITY_EDGE_AUDIT_V1',target_manifest_sha256=original_manifest_sha,targets_modified=False,new_GPU_forwards=0,results=results,all49_mask_topology=topology,
    reason='World Poisson exterior constraints do not constrain the per-view replacement/identity interface.',
    control_proposal='One fixed10px distance feather of current locked B toward original observed RGB on exactly the same legal M, no grid/source/Poisson/geometry/alpha changes. Compare hardB versus featherB with same fit budget; root authorization required for next fit.',
    inference_limits=['Reduction of C hard-mask residual jump is algebraic by construction, not target/native success.',
      'Thin/holey M can make all pixels low atlas weight: report dilution/fullweight count.',
      'Shared atlas remains fixed; final image target becomes deterministically view-dependent via validity distance and observedRGB, no free per-view colour variables.',
      'Acceptance stays original realRGB/world bands/native49/protection, not just new mask-boundary diagnostic.'])
 (B/'hardreplace_edge_audit_v1.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 print(json.dumps([dict(raw=r['raw'],edges=r['cross_mask_edge_metrics']['ALL'],weights=r['within_domain_tenpx_weight'],fullweight=r['full_atlas_weight_in_domain']) for r in results],ensure_ascii=False))
if __name__=='__main__':main()
