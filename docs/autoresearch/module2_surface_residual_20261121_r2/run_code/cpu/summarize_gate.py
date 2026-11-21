"""CPU-only per-residual source accounting of the fixed actual joint K gate."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2')
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
d=load(P/'proposal/candidates.npz');e=load(P/'k_gate/per_candidate_K_evidence.npz');safe=load(P/'k_gate/safe_candidates.npz');stack=load(P/'k_gate/safe_stack_diagnostic.npz');safeids=set(safe['candidate_id'].tolist());stackids=set(stack['candidate_id'].tolist());rows=[];perpixel=[]
for raw,j in [(86,0),(110,1)]:
 for role,bit in [('C',2),('S',4)]:
  selected=(d['support_frame_ids'][:,j]==raw)&(d['source_roles_per_view'][:,j]==bit);ix=np.flatnonzero(selected);training=np.asarray([int(d['candidate_id'][i])in safeids for i in ix]);isstack=np.asarray([int(d['candidate_id'][i])in stackids for i in ix]);conflict=d['projection_K_conflict'][ix].any(1);kind=d['candidate_kind'][ix];rows.append(dict(raw=raw,source=role,fixed_residual_pixels=len(ix),safe_existing_inactive=int((training&(kind==0)).sum()),safe_spatially_new_append=int((training&(kind==1)).sum()),safe_stack_diagnostic=int(isstack.sum()),rejected_joint_K=int((~training&~isstack).sum()),safe_despite_projection_conflict=int((training&conflict).sum()),safe_stack_projection_conflict=int((isstack&conflict).sum())))
  for i in ix:
   cid=int(d['candidate_id'][i]);x,y=d['source_pixels'][i,j];ixev=np.flatnonzero(e['candidate_id']==cid)
   if len(ixev)!=1:raise ValueError('candidate ID mismatch in K evidence')
   k=int(ixev[0]);uv=e['unsafe_state_view'][k];status='safe_existing_inactive'if cid in safeids and d['candidate_kind'][i]==0 else('safe_spatially_new'if cid in safeids else('safe_same_center_stack'if cid in stackids else'rejected_joint_K'))
   perpixel.append(dict(raw=raw,source=role,x=int(x),y=int(y),flat_pixel_id=int(y*1152+x),candidate_id=cid,parent_alpha=float(d['parent_source_alpha'][i,j]),candidate_kind=int(d['candidate_kind'][i]),status=status,projection_K_conflict=bool(d['projection_K_conflict'][i].any()),native_F32_same_center=bool(d['parent_active_F32_model_center_same'][i]),rejection_reason_code=int(e['rejection_reason'][k]),unsafe_states_views=';'.join(str(s)+':'+str([86,110,134,179,198][v])for s,v in zip(*np.nonzero(uv))),max_VJP_K_mass=float(np.max(e['VJP_K_mass'][k]))))
out=P/'proposal';target=out/'residual_pixel_K_table.csv'
with target.open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(perpixel[0]));w.writeheader();w.writerows(perpixel)
result=dict(schema='r2_fixed_residual_joint_K_audit.v1',rows=rows,source_pixels=len(perpixel),training_geometry_candidates=len(safeids),safe_stacks=len(stackids),not_training_any_stack_copy=True,pointwise_table=str(target),pointwise_table_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),K_evidence=str(P/'k_gate/per_candidate_K_evidence.npz'),K_evidence_sha256=hashlib.sha256((P/'k_gate/per_candidate_K_evidence.npz').read_bytes()).hexdigest(),source='same fixed R1-certified residuals, actual two-state five-view joint gate; no CPU mask expansion or proposals changed',GPU_calls=0,training_updates=0)
(out/'residual_gate_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
