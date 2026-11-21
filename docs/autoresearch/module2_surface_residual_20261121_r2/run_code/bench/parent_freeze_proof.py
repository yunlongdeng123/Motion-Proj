"""Independent CPU proof of direct-parent row binding and authorized mutable parameters."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2');N=P.parent/'road_surface_adaptive_20261121_r1/appearance_fit/asset'
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def row_ids(p):
 d=load(p)
 for k in ['world_rows','allowed_world_rows','allowed_color_world_rows','allowed_color_rows','world_row_ids','rows','indices','allowed_mask','allowed']:
  if k not in d:continue
  a=np.asarray(d[k])
  if a.dtype==bool:return np.flatnonzero(a)
  if a.ndim==1 and np.issubdtype(a.dtype,np.integer):return a.astype(np.int64)
 raise ValueError('unknown authorized-row schema:'+str(list(d)))
def proof(models):
 parent=load(N/'road.npz');prior=load(N/'provenance.npz');n=len(parent['active']);allowedfile=P/'train_inputs/allowed_color_world_rows.npz'
 if not allowedfile.exists():raise FileNotFoundError('missing exact allowed color list')
 allowed=row_ids(allowedfile);safe=load(P/'k_gate/safe_candidates.npz');stack=load(P/'k_gate/safe_stack_diagnostic.npz');activation=safe['existing_world_rows'][safe['candidate_kind']==0];stackrows=np.unique(stack['parent_active_F32_model_center_world_row']);stackrows=stackrows[stackrows>=0];group=prior['adaptive_append_new_geometry']|prior['adaptive_activate_existing'];qualified=stackrows[group[stackrows]];expected_old=np.union1d(activation,qualified);oldallowed=allowed[allowed<n]
 if np.setdiff1d(oldallowed,expected_old).size:raise ValueError('allowed list expands to unqualified parent rows')
 reuse=np.intersect1d(oldallowed,np.flatnonzero(parent['active']));reusefile=P/'train_inputs/reuse_existing.npz'
 if reusefile.exists():
  declared_reuse=load(reusefile)['ids']
  if not np.array_equal(np.unique(declared_reuse),reuse):raise ValueError('reuse list and allowed-color list disagree')
 mask=np.zeros(n,bool);mask[oldallowed]=True;active_frozen=parent['active']&~mask;all_frozen=~mask;rows=[]
 for model in models:
  asset=Path(model['asset_dir']);asset=asset if asset.is_absolute()else P/asset;current=load(asset/'road.npz');m=len(current['active'])
  if m<n:raise ValueError('direct-parent prefix missing')
  declared=model.get('point_counts',{});actual_activated=int((current['active'][:n]&~parent['active']).sum())
  for key,actual in [('total',m),('active',int(current['active'].sum())),('appended_new_geometry',m-n),('activated_existing',actual_activated),('reuse_existing_newgroup',len(reuse))]:
   # Before the reuse training stage it is valid to declare0 participating reuse rows.
   if key=='reuse_existing_newgroup'and declared.get(key,0)==0:continue
   if key in declared and int(declared[key])!=actual:raise ValueError(model['id']+' incorrect point count '+key)
  if not np.isfinite(current['reference_srgb']).all()or np.any(current['reference_srgb']<0)or np.any(current['reference_srgb']>1):raise ValueError('invalid appearance RGB bounds')
  row=dict(id=model['id'],parent_total=n,current_total=m,parent_active=int(parent['active'].sum()),authorized_old_color_rows=len(oldallowed),reuse_existing_newgroup=len(reuse),authorized_existing_activation=len(activation),allowed_color_source=str(allowedfile),allowed_color_SHA256=hashlib.sha256(allowedfile.read_bytes()).hexdigest(),checks={})
  for key in ['model_xyz','normal_world','minimum_model_depth']:
   same=np.array_equal(current[key][:n],parent[key]);row['checks'][key+'_prefix_exact']=same
   if not same:raise ValueError(model['id']+' moved/changed frozen parent '+key)
  for key in ['road_opacity','gaussian_sigma_m','reference_srgb']:
   a,b=current[key][:n],parent[key];delta=np.abs(a-b);changed=np.any(delta>0,axis=tuple(range(1,delta.ndim)))if delta.ndim>1 else delta>0;unexpected=changed&all_frozen;maxdelta=float(delta[all_frozen].max(initial=0));row['checks'][key+'_outside_allowed_exact']=not unexpected.any();row[key+'_outside_allowed_max']=maxdelta;row[key+'_unexpected_changed_rows']=int(unexpected.sum());row[key+'_allowed_parent_changed_rows']=int((changed&mask).sum())
   if unexpected.any():raise ValueError(model['id']+' changed unauthorized parent '+key)
  if model['id']!='appearance':
   unchanged=np.array_equal(current['reference_srgb'][reuse],parent['reference_srgb'][reuse]);row['checks']['reuse_RGB_frozen_before_appearance']=unchanged
   if not unchanged:raise ValueError('reuse RGB changed before the appearance stage')
  unexpected_active=(current['active'][:n]!=parent['active'])&~np.isin(np.arange(n),activation);row['checks']['parent_activation_scope_exact']=not unexpected_active.any()
  if unexpected_active.any():raise ValueError('unauthorized parent active change')
  if not np.array_equal(current['active'][:n][parent['active']],parent['active'][parent['active']]):raise ValueError('old active point deactivated')
  rows.append(row)
 byid={m['id']:(Path(m['asset_dir'])if Path(m['asset_dir']).is_absolute()else P/m['asset_dir'])for m in models};coverage=load(byid['coverage']/'road.npz');appearance=load(byid['appearance']/'road.npz');appearance_frozen={k:np.array_equal(coverage[k],appearance[k])for k in ['active','model_xyz','normal_world','gaussian_sigma_m','road_opacity','minimum_model_depth']}
 if not all(appearance_frozen.values()):raise ValueError('appearance stage changed frozen geometry or coverage')
 result=dict(schema='r2_direct_parent_freeze_proof.v1',passed=True,parent=str(N),parent_total=n,parent_active=int(parent['active'].sum()),actual_reuse_world_rows=reuse.tolist(),actual_activation_world_rows=activation.tolist(),qualified_R1_newgroup_reuse_candidates=qualified.tolist(),rows=rows,appearance_frozen_coverage_fields=appearance_frozen,checks='all parent geometry frozen; old opacity/scales/colors mutable only on new activation or approved R1-newgroup reuse; new appended rows separate',CPU_only=True)
 return result
