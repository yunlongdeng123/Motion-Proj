"""冻结已锁r2许可与.34，导出小真实CPU接线fixture，无优化/原生调用。"""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from scipy.ndimage import maximum_filter
from teacher_detail_permission import P,Q,N,load,sha,COMMON_HIGH_BOUND
def main(output):
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 source=Q/'permission/ledger_r2';d=load(source/'teacher_evidence.npz');a=load(P/'cpu_exact_r1/A0.npz')
 row=load(Q/'permission/old_audit_r1/row_H_support.npz')
 plan=load(P/'chart_plan_candidate_r1/geometry_plan.npz',['center_uv','weights','geometric_sample_valid'])
 grid=d['world_row_grid'];H,W=grid.shape;rows=d['active_world_rows'];xy=np.rint(plan['center_uv']).astype(int);x,y=xy.T
 margin=(x>=20)&(x<W-20)&(y>=20)&(y<H-20)
 allow=d['teacher_H_allow'][y,x];write=d['common_write_active'];sourcecode=row['source_effective'];C=row['C_mass']>0;S=row['S_mass']>0
 cases={
  'C_unknown_seed':margin&allow&write&C&(sourcecode==0),
  'C_generated_zero_old_H':margin&allow&write&C&(sourcecode==3)&(row['old_Sigma_mH']<=1e-12),
  'S_prior_observed':margin&allow&write&S&(sourcecode==1),
  'S_other_prior':margin&allow&write&S&(sourcecode!=1)}
 masks={};meta=[];uv_cache=np.load(P/'chart_plan_candidate_r1/sample_uv_f64.npy',mmap_mode='r')
 with np.load(N/'road.npz') as r:legacy=r['reference_srgb'].copy()
 def add(name,xx,yy,world,active_index,mode):
  sl=np.s_[yy-20:yy+21,xx-20:xx+21]
  for key,value in [('base_prior',a['rgb_srgb']),('prior_valid',a['valid']),('seed',d['base_seed_lowpass_srgb']),
    ('teacher_valid',d['teacher_material_valid']),('gate_old',d['observed_H_prior']),('gate_new',d['mH_union']),
    ('gate_teacher',d['teacher_gate']),('latent',d['common_latent_mask']),('reg',d['common_reg_support']),
    ('source',d['original_source']),('known',d['original_known']),('unknown',d['original_unknown'])]:
   masks[name+'__'+key]=value[sl].copy()
  if mode=='Sigma':
   masks[name+'__uv']=np.asarray(uv_cache[active_index])-np.array([xx-20,yy-20])
   masks[name+'__weights']=plan['weights'];masks[name+'__geometry']=plan['geometric_sample_valid'][active_index]
  else:
   masks[name+'__uv']=np.array([[20.,20.]]);masks[name+'__weights']=np.ones(1);masks[name+'__geometry']=np.ones(1,bool)
  masks[name+'__legacy']=legacy[world];masks[name+'__write']=np.array(bool(d['common_write_candidate'][yy,xx]))
  meta.append(dict(case=name,world_row=int(world),active_index=int(active_index),mode=mode,origin_uv=[xx-20,yy-20],
   source=int(d['original_source'][yy,xx]),known=bool(d['original_known'][yy,xx]),unknown=bool(d['original_unknown'][yy,xx]),
   prior_valid=bool(a['valid'][yy,xx]),teacher_H_allow=bool(d['teacher_H_allow'][yy,xx])))
 for name,mask in cases.items():
  ids=np.flatnonzero(mask)
  if not len(ids):raise ValueError('no actual fixture:'+name)
  i=ids[len(ids)//2];add(name,int(x[i]),int(y[i]),int(rows[i]),int(i),'Sigma')
 # actual conflicting field point: only NEW teacher component must be zero; old observed path may remain.
 yy,xx=np.mgrid[:H,:W];interior=(xx>=20)&(xx<W-20)&(yy>=20)&(yy<H-20)
 for name,mask in [('HF_conflict',d['teacher_HF_conflict']&interior),
                   ('true_unknown_no_teacher',(~d['common_material_valid'])&d['geometry_support']&interior),
                   ('K_readonly',d['guard']&d['geometry_support']&interior&(maximum_filter(d['common_latent_mask'],size=19)>0))]:
  ids=np.flatnonzero(mask)
  if not len(ids):raise ValueError('no actual field fixture:'+name)
  flat=ids[len(ids)//2];cy,cx=divmod(int(flat),W);world=int(grid[cy,cx]);add(name,cx,cy,world,-1,'point')
 np.savez_compressed(out/'cpu_mini_fixture.npz',**masks)
 shutil.copyfile(source/'teacher_evidence.npz',out/'teacher_evidence.npz')
 shutil.copyfile(source/'conflict_world_nodes.png',out/'conflict_world_nodes.png')
 recipe=dict(schema='teacher_detail_permission_final_CPU.v1',task='MODULE2-TEACHER-DETAIL-CPU-20261119',
  common_high_bound=COMMON_HIGH_BOUND,highpass_sigma_m=.06,highpass_sigma_cells=3,base_seed_lowpass_sigma_m=.12,
  T2_H_initialization_abs_max=.02,prior_loss_weight=0,
  immutable_ledger_sha256=sha(out/'teacher_evidence.npz'),source_r2_sha256=sha(source/'teacher_evidence.npz'),
  common_material_valid_and_base_same_all_five_arms=True,common_latent_and_reg_support_same_all_five_arms=True,
  prior_source_known_unknown_are_ledger_only_not_implicit_masks=True,
  output_H_permission={'observed':'observed_H_prior','union_teacher':'mH_union'},
  source_qualification='HF4code conflict gate, clean realA-HF protect; RGB conflict diagnostic only; localblur<.9 single-effective evidence not dual agreement',
  shapes={'common_base_srgb':'HWC','common_material_valid':'HW bool','observed_H_prior':'HW','teacher_H_allow':'HW bool',
   'mH_union':'HW','common_latent_mask':'HW','common_reg_support':'HW','common_write_active':'151988 bool','active_world_rows':'151988 int64'},
  counts=json.loads((source/'receipt.json').read_text())['counts'],fixture_cases=meta,
  GPU_authorized=False,no_training_controller_created=True,native_calls=0,optimizer_updates=0,default_scene_changed=False,
  actual_native_scope_validation_required_before_future_training=True)
 (out/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
 print(json.dumps(dict(output=str(out),cases=meta,Hbound=COMMON_HIGH_BOUND,old_P_modified=False)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)

