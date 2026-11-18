"""复用已有world教师映射，分离低频RGB差与HF冲突；CPU-only、无Σ全量重遍。"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter
from teacher_detail_permission import P,Q,N,load,lowpass,sample,sha
def stats(a):
 a=np.asarray(a);return dict(count=int(a.size),mean=float(a.mean()) if a.size else None,
  quantiles=np.quantile(a,[0,.05,.25,.5,.75,.95,.99,1]).tolist() if a.size else None)
def filtered(rgb,valid,sigma):
 return lowpass(rgb,valid,sigma),gaussian_filter(valid.astype(float),sigma,mode='constant',truncate=3)
def edge(rgb,valid):
 gray=rgb.mean(-1);gx=np.zeros(gray.shape);gy=np.zeros(gray.shape)
 gx[:,1:-1]=(gray[:,2:]-gray[:,:-2])*.5;gy[1:-1]=(gray[2:]-gray[:-2])*.5
 good=valid.copy();good[:,1:-1]&=valid[:,2:]&valid[:,:-2];good[1:-1]&=valid[2:]&valid[:-2]
 return gx,gy,np.hypot(gx,gy),good
def main(output,old_dir,hf_codes=4):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 old_dir=Path(old_dir);old=load(old_dir/'teacher_evidence.npz')
 a=load(P/'cpu_exact_r1/A0.npz');views=[load(old_dir/('%03d_evidence.npz'%r)) for r in [86,110]]
 geometry=old['geometry_support'];guard=old['guard'];safe=geometry&~guard
 role=[];rgb=[];valid=[];hf=[];hf_mass=[];lf=[];lf_mass=[];ahf=[];amass=[];alf=[]
 for v in views:
  mask=(v['C_valid']|v['S_valid'])&safe;value=np.where(v['C_valid'][...,None],v['C_rgb'],v['S_rgb'])
  low,mass=filtered(value,mask,3);l12,m12=filtered(value,mask,6)
  av=v['A_valid']&safe;al,amd=filtered(v['A_rgb'],av,3);al12,_=filtered(v['A_rgb'],av,6)
  valid.append(mask);rgb.append(value);hf.append(value-low);hf_mass.append(mass);lf.append(l12);lf_mass.append(m12)
  ahf.append(v['A_rgb']-al);amass.append(amd);alf.append(al12);role.append(v['C_valid'].astype(np.uint8)|v['S_valid'].astype(np.uint8)*2)
 both=valid[0]&valid[1];hf_both=both&(hf_mass[0]>=.9)&(hf_mass[1]>=.9)
 rgb_delta=np.max(np.abs(rgb[0]-rgb[1]),-1);hf_delta=np.max(np.abs(hf[0]-hf[1]),-1)
 rgb_conflict=both&(rgb_delta>4/255);hf_conflict=hf_both&(hf_delta>hf_codes/255)
 # fixed owner by available local support, tie raw110; never choose closest RGB to A0.
 owner=np.where(valid[1]&(~valid[0]|(hf_mass[1]>=hf_mass[0])),1,0).astype(np.uint8)
 target_hf=np.where(owner[...,None]==0,hf[0],hf[1]);target_rgb=np.where(owner[...,None]==0,rgb[0],rgb[1])
 owner_mass=np.where(owner==0,hf_mass[0],hf_mass[1])
 realA=views[0]['A_valid']|views[1]['A_valid'];a_conflict=np.zeros(geometry.shape,bool)
 for i,v in enumerate(views):
  reliable=v['A_valid']&(amass[i]>=.9)&(owner_mass>=.9)
  a_conflict|=reliable&(np.max(np.abs(target_hf-ahf[i]),-1)>hf_codes/255)
 material=(valid[0]|valid[1])&safe
 allow=material&~hf_conflict&~a_conflict
 # base owner is LF-support, independent from outputH permit/conflict.
 base_owner=np.where(valid[1]&(~valid[0]|(lf_mass[1]>=lf_mass[0])),1,0).astype(np.uint8)
 base_seed=np.where(base_owner[...,None]==0,lf[0],lf[1])
 a_owner=np.where(views[1]['A_valid']&(~views[0]['A_valid']|(amass[1]>=amass[0])),1,0)
 seed_A=np.where(a_owner[...,None]==0,alf[0],alf[1])
 base_seed=np.where(realA[...,None],seed_A,base_seed)
 seed_mask=material&~a['valid'];common_base=np.where(a['valid'][...,None],a['rgb_srgb'],np.where(seed_mask[...,None],base_seed,0))
 common_valid=a['valid']|seed_mask;obs=a['mH'];gate=allow.astype(float);union=np.maximum(obs,gate)
 # only small old masks loaded with mmap; no old kernel is recomputed or CUDA queried.
 import torch
 torch.set_num_threads(1)
 saved=torch.load(P/'formal_r1/A200/asset/world_field.pt',map_location='cpu',mmap=True,weights_only=False)
 legacy_reg=saved['state']['high_output_mask'][...,0].numpy().copy()>0
 del saved
 reg=(legacy_reg|allow)&safe
 prior_observed=(old['original_known']&(old['original_source']==1)&a['valid'])
 latent=(maximum_filter(reg.astype(np.uint8),size=19,mode='constant')>0)&safe&(prior_observed|material)
 oldrows=load(Q/'permission/old_audit_r1/row_H_support.npz');rows=oldrows['active_world_rows'];grid=old['world_row_grid']
 inverse=np.empty(grid.size,np.int64);inverse[grid.ravel()]=np.arange(grid.size);at=inverse[rows]
 oldscope=oldrows['old_eligible'];rowsafe=oldrows['physical_K_near_safe']
 active_material=material.ravel()[at];active_valid=common_valid.ravel()[at]
 write=(oldscope|active_material)&rowsafe&active_valid
 if (oldscope&~write).any():raise AssertionError('old write scope unexpectedly removed')
 write_grid=np.zeros(grid.size,bool);write_grid[at]=write;write_grid=write_grid.reshape(grid.shape)
 evidence=dict(old)
 evidence.update(prior_observed_support=prior_observed,real_observed_support=realA,realA_support=realA,
  teacher_material_valid=material,teacher_evidence_valid=material,teacher_H_allow=allow,teacher_gate=gate,mH_union=union,
  teacher_only_gate=allow&~realA,teacher_HF_conflict=hf_conflict,teacher_A_HF_conflict=a_conflict,
  teacher_RGB_conflict=rgb_conflict,teacher_conflict=hf_conflict|a_conflict,HF_dual_comparable=hf_both,
  HF_single_effective_evidence=material&~hf_both,HF_local_support_insufficient=material&(owner_mass<.9),
  teacher_hf_srgb=target_hf,teacher_hf_owner_raw=np.where(owner==0,86,110).astype(np.int16),
  base_seed_owner_raw=np.where(base_owner==0,86,110).astype(np.int16),
  common_base_srgb=common_base,common_material_valid=common_valid,base_seed_ledger_mask=seed_mask,
  base_seed_lowpass_srgb=base_seed,base_seed_source_role=np.where(seed_mask,np.where(realA,1,3),0).astype(np.uint8),
  common_latent_mask=latent,common_reg_support=reg,common_write_candidate=write_grid,
  common_write_active=write,active_world_rows=rows)
 np.savez_compressed(output/'teacher_evidence.npz',**evidence)
 base_low=lowpass(common_base,common_valid,3);base_hf=common_base-base_low
 need=target_hf-base_hf
 amp_valid=allow&(owner_mass>=.9);amp_abs=np.max(np.abs(need),-1)
 C=old['teacher_C_support'];S=old['teacher_S_support'];regions={'C':C,'S':S,'all':C|S}
 floor={};amplitude={}
 for name,m in regions.items():
  dual=m&both;reliable=m&hf_both
  floor[name]=dict(node_both=int(dual.sum()),RGB_half_mean_absolute_difference=float(.5*np.abs(rgb[0]-rgb[1])[dual].mean()) if dual.any() else None,
   RGB_absolute_difference=stats(np.abs(rgb[0]-rgb[1])[dual]),
   HF_comparable_nodes=int(reliable.sum()),HF_half_mean_absolute_difference=float(.5*np.abs(hf[0]-hf[1])[reliable].mean()) if reliable.any() else None,
   HF_absolute_difference=stats(np.abs(hf[0]-hf[1])[reliable]),
   RGB_conflict_HF_not_conflict=int((m&rgb_conflict&~hf_conflict).sum()),
   HF_conflict_without_RGB_conflict=int((m&hf_conflict&~rgb_conflict).sum()))
  qualified=m&amp_valid
  amplitude[name]=dict(qualified_nodes=int(qualified.sum()),required_HF_maxchannel=stats(amp_abs[qualified]),
   abs_over_old_point04_fraction=float((amp_abs[qualified]>.04).mean()) if qualified.any() else None,
   abs_over_point08_fraction=float((amp_abs[qualified]>.08).mean()) if qualified.any() else None)
 ex0,ey0,mag0,e0=edge(hf[0],valid[0]&(hf_mass[0]>=.9));ex1,ey1,mag1,e1=edge(hf[1],valid[1]&(hf_mass[1]>=.9))
 em=e0&e1&(mag0>=.02)&(mag1>=.02)
 cosine=np.divide(ex0*ex1+ey0*ey1,mag0*mag1,out=np.zeros_like(mag0),where=mag0*mag1>1e-12)
 edge_stats=dict(dual_strong_edges=int(em.sum()),cosine=stats(cosine[em]),opposed_edges=int((em&(cosine<0)).sum()),used_for_permission_gate=False)
 p99=float(np.quantile(amp_abs[amp_valid],.99)) if amp_valid.any() else 0
 suggestions=dict(pointwise_H_bound_p99_plus4codes=float(np.ceil((p99+4/255)/.01)*.01),
  half_scaled_Highpass_basis_2xp99_plus4codes=float(np.ceil((2*p99+4/255)/.01)*.01),
  formula='HP=.5*b*(tanh(raw)-G6cm(tanh(raw))); b is global infinity upper bound, a sinusoidal highpass generally has only about .5b amplitude',
  not_a_full_renderer_or_global_feasibility_certificate=True)
 # fixed small Sigma subset, not a full 34M pass.
 candidate=amp_valid.ravel()[at]&rowsafe
 groups={'C_generated':candidate&(oldrows['C_mass']>0)&(oldrows['source_effective']==3),
  'C_unknown_seed':candidate&(oldrows['C_mass']>0)&(oldrows['source_effective']==0),
  'S_prior_observed':candidate&(oldrows['S_mass']>0)&(oldrows['source_effective']==1),
  'S_otherprior':candidate&(oldrows['S_mass']>0)&(oldrows['source_effective']!=1)}
 chosen=np.unique(np.concatenate([np.flatnonzero(m)[np.linspace(0,int(m.sum())-1,min(64,int(m.sum())),dtype=int)] for m in groups.values() if m.any()]))
 plan=load(P/'chart_plan_candidate_r1/geometry_plan.npz',['weights','geometric_sample_valid'])
 uv=np.load(P/'chart_plan_candidate_r1/sample_uv_f64.npy',mmap_mode='r')[chosen]
 v,_=sample(common_valid.astype(float),uv);v*=plan['geometric_sample_valid'][chosen]
 g,_=sample(union,uv);w=plan['weights'][None,:];den=(w*v).sum(1)
 residual,_=sample(need*allow[...,None],uv);residual*=plan['geometric_sample_valid'][chosen,...,None]
 delta=(w[...,None]*residual).sum(1)/den[:,None].clip(1e-12)
 gmean=(w*v*g).sum(1)/den.clip(1e-12);required=np.max(np.abs(delta),axis=1)/gmean.clip(1e-12)
 sigma_stats=dict(rows=int(len(chosen)),quad_points=int(len(chosen)*len(plan['weights'])),
  required_H_bound_for_fixedLF_HF_component=stats(required),
  over_point04_fraction=float((required>.04).mean()) if len(required) else None,
  interpretation='conditional HF-component proxy on fixed common base; L/H bands overlap, no global infeasibility claim')
 np.savez_compressed(output/'small_Sigma_hf.npz',active_indices=chosen,world_rows=rows[chosen],HF_delta=delta,union_gate_mean=gmean,required_pointwise_H_bound=required)
 from PIL import Image
 picture=np.zeros(grid.shape+(3,),np.uint8);picture[material]=[30,100,140];picture[allow]=[50,180,110];picture[rgb_conflict&~hf_conflict]=[220,155,50];picture[hf_conflict|a_conflict]=[210,60,85];picture[seed_mask]=[145,75,200]
 Image.fromarray(picture).save(output/'conflict_world_nodes.png')
 counts=dict(teacher_material_nodes=int(material.sum()),H_allowed_nodes=int(allow.sum()),HF_conflict_nodes=int(hf_conflict.sum()),
  realA_HF_conflict_nodes=int(a_conflict.sum()),rawRGB_conflict_nodes=int(rgb_conflict.sum()),H_new_on_old_gate_zero=int((allow&(obs==0)).sum()),
  common_latent_nodes=int(latent.sum()),common_reg_nodes=int(reg.sum()),base_seed_nodes=int(seed_mask.sum()),
  common_write_active=int(write.sum()),old84266_retained=int((oldscope&write).sum()),old_excluded=int((oldscope&~write).sum()),
  new_write_rows=int((write&~oldscope).sum()),new_C_prior_unavailable=int((write&~oldscope&(oldrows['C_mass']>0)&~oldrows['source_effective'].astype(bool)).sum()),
  single_effective_HF_evidence=int((material&~hf_both).sum()),local_HF_support_insufficient=int((material&(owner_mass<.9)).sum()))
 assert np.array_equal(common_base[a['valid']],a['rgb_srgb'][a['valid']])
 result=dict(schema='teacher_HF_permission_CPU_candidate.v2',CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,
  counts=counts,node_association_conflict=floor,HF_amplitude=amplitude,edge_diagnostic=edge_stats,bound_suggestions=suggestions,
  small_Sigma_proxy=sigma_stats,conflict_HF_max_channel_codes=hf_codes,blur_support_threshold=.9,
  highpass_sigma_m=.06,base_seed_lowpass_sigma_m=.12,
  teacher_H_gate_semantics='bool1 footprint qualification + reliable HF conflict / clean realA-HF protection; RGB offset not a hard H veto',
  base_seed_semantics='prior valid A0 exact; absentbase only LF from clean realA if available else highest local blur-support teacher owner, tie110. No teacherHF baked into base.',
  common_arm_semantics='same base/valid/write/latent/reg; mH_obs vs max(mH_obs,teacher_bool1); prior source/known/unknown never re-mask new H',
  floor_boundary='equalweight node-associated two-color L1 proxy, not actual native renderer/global lower bound; fixed C/S train/eval denominators unchanged',
  mask_erosion='all4 lowres neighboring teacher pixels each have full4x4 native permission. No extra9cell world erosion of outputmH; T2 raw initialization needs its independent9cell envelope erosion.',
  no_GPU_authorization=True,status='CPU_READY_PENDING_CENTRAL_BOUND_AND_SCOPE_LOCK',
  source_projection_from=str(old_dir),original_P_unmodified=True,seconds=time.monotonic()-start)
 (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--old-dir',type=Path,default=Q/'permission/ledger_r1');a=ap.parse_args();main(a.output,a.old_dir)

