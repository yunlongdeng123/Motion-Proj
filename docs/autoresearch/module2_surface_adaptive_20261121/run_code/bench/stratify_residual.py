"""Read-only fixed-pixel feasibility diagnosis; no proposals, training, or parameter sweep."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_adaptive_20261121_r1');OUT=P/'diagnostics/residual_firstwave'
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(maxroot):
 OUT.mkdir(parents=True,exist_ok=True);proposal=json.loads((P/'proposal/manifest.json').read_text());act=load(P/'proposal/activation.npz');reserve=load(P/'proposal/append_reserve.npz');append=load(P/'proposal/append.npz');selection=json.loads((P/'proposal/append_manifest.json').read_text());selected=np.asarray(selection['selected_reserve_ids']);discarded=np.ones(len(reserve['world_xyz']),bool);discarded[selected]=False;fixed={};rows=[];ready=(maxroot/'manifest.json').exists()and all((maxroot/'rendered'/f'{r:03d}.npz').exists()for r in [86,110,134,179,198])
 def source_pixel_set(d,raw,flag=None):
  j=0 if raw==86 else 1;valid=d['support_frame_ids'][:,j]==raw
  if flag is not None:valid&=flag
  xy=d['source_pixels'][valid,j];return np.unique(xy[:,1]*1152+xy[:,0])
 for raw in [86,110]:
  finalpath=P/f'appearance_fit/rendered/{raw:03d}.npz';final=load(finalpath);d=load(P/f'proposal/domains/{raw:03d}.npz');alpha=final['road_alpha'];maxp=load(maxroot/'rendered'/f'{raw:03d}.npz')if ready else None
  aset=source_pixel_set(act,raw);nset=source_pixel_set(append,raw);dset=source_pixel_set(reserve,raw,discarded)
  for source in ['C','S']:
   certified=d['appearance_'+source]&d['canonical_valid'];residual=certified&(alpha<.95);ids=np.flatnonzero(residual);a=alpha.ravel()[ids];key=f'{raw:03d}_{source}';fixed[key+'_pixel_ids']=ids;fixed[key+'_final_alpha']=a;flags={'explicit_activation_source_pixel':np.isin(ids,aset),'explicit_selected_append_source_pixel':np.isin(ids,nset),'explicit_discarded_reserve_source_pixel':np.isin(ids,dset)}
   distanceK=distance_transform_edt(~final['K']).ravel()[ids];row=dict(raw=raw,source=source,certified_domain_n=int(certified.sum()),final_residual_lt095_n=len(ids),final_no_support_lt005_n=int((a<.05).sum()),final_thin_005_to095_n=int(((a>=.05)&(a<.95)).sum()),inside_fixed_proposal_fit_n=int(d['fit'].ravel()[ids].sum()),nearest_sameview_K_distance_px_quantiles=np.quantile(distanceK,[0,.25,.5,.75,1]).tolist()if len(ids)else[],candidate_source_pixel_matches={k:int(v.sum())for k,v in flags.items()},unconfirmed_editing_pixels_excluded=int((d['appearance_'+source]&~d['canonical_valid']).sum()),final_packet_sha256=sha(finalpath))
   if ready:
    ma=maxp['road_alpha'].ravel()[ids];weight=np.asarray(maxp['new_geometry_weight']).squeeze().ravel()[ids];fixed[key+'_max_allowed_alpha']=ma;fixed[key+'_max_new_geometry_weight']=weight;layers={}
    for name,mask in [('all_residual',np.ones(len(ids),bool)),('final_no_support',a<.05),('final_thin',(a>=.05)&(a<.95))]:
     unreachable=mask&(ma<.05);partial=mask&(ma>=.05)&(ma<.95);coverable=mask&(ma>=.95);no_new=mask&(weight<=0);reaches=mask&(weight>0)
     layers[name]=dict(pixels=int(mask.sum()),max_still_no_support_lt005_n=int(unreachable.sum()),max_partial_005_to095_n=int(partial.sum()),max_can_stably_cover_ge095_n=int(coverable.sum()),actual_new_group_no_contribution_n=int(no_new.sum()),actual_new_group_positive_contribution_n=int(reaches.sum()),no_new_contribution_and_no_explicit_source_candidate_n=int((no_new&~flags['explicit_activation_source_pixel']&~flags['explicit_selected_append_source_pixel']).sum()),no_new_contribution_despite_explicit_source_candidate_n=int((no_new&(flags['explicit_activation_source_pixel']|flags['explicit_selected_append_source_pixel'])).sum()),no_new_contribution_nearest_sameview_K_px_quantiles=np.quantile(distanceK[no_new],[0,.25,.5,.75,1]).tolist()if no_new.any()else[])
    row.update(max_allowed_layers=layers,max_alpha_quantiles=np.quantile(ma,[0,.25,.5,.75,1]).tolist()if len(ids)else[],max_minus_final_alpha_min=float((ma-a).min())if len(ids)else None,max_packet_sha256=sha(maxroot/'rendered'/f'{raw:03d}.npz'))
   rows.append(row)
 np.savez_compressed(OUT/'fixed_residual_pixels.npz',**fixed)
 result=dict(schema='adaptive_certified_residual_feasibility.v1',diagnostic_ready=ready,rows=rows,residual_rule='appearance alpha<.95 AND fixed certified C/S road domain; full unconfirmed editing domain is explicitly excluded from this diagnosis',fixed_pixel_file=str(OUT/'fixed_residual_pixels.npz'),proposal_trace=dict(candidate_pixels=proposal['candidate_pixels'],rejected_K_aggregate=proposal['rejected_K'],multi_view_spatial_merged=proposal['multi_view_spatial_merged'],cap_truncated=proposal['cap_truncated'],pointwise_rejection_table_available=False),cause_boundary='new_geometry_weight is actual road-only unit-alphaT of the existing two new groups at max allowed state; zero identifies no real group reach. Aggregate K rejects cannot attribute a particular residual pixel to K versus sampling; no fabricated pointwise cause.',probe_caveat='opacity1 is optimistic relaxation above registered.999; tangent sigmas use contract caps. The rendered configuration is not a strict pointwise alpha upper bound and is not a persisted method asset. Layer names describe observed values of this one probe, not a universal feasibility certificate.',max_state_root=str(maxroot),max_state_receipt_sha256=sha(maxroot/'receipt.json')if(maxroot/'receipt.json').exists()else None,no_new_sampling=True,no_parameter_sweep=True,no_GPU_calls=True,no_training_updates=True,formal_assets_changed=False)
 (OUT/'stratification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--max-state-root',type=Path,default=P/'residual_feasibility/max_allowed_state');a=ap.parse_args();main(a.max_state_root)
