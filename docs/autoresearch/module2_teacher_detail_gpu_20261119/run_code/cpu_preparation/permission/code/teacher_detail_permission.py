"""仅CPU的独立教师证据/高频许可；绝不改原source/known/unknown或旧实验。"""
from pathlib import Path
import os
for key in ['CUDA_VISIBLE_DEVICES']:os.environ[key]=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json,time
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
Q=P.parent/'road_teacher_detail_cpu_20261119_r1'
V=P.parent/'road_capacity_cpu_20261117_r1/supervision_v4_candidate_r1'
N=P.parent/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
BASE=P.parent/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz'
COMMON_HIGH_BOUND=.34
def load(path,keys=None):
 with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in (d.files if keys is None else keys)}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def sample(a,uv):
 x,y=uv[...,0],uv[...,1];finite=np.isfinite(x)&np.isfinite(y);x=np.where(finite,x,0);y=np.where(finite,y,0)
 ix,iy=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64);dx,dy=x-ix,y-iy;H,W=a.shape[:2]
 tail=a.shape[2:];out=np.zeros(x.shape+tail,np.float64);inside=finite&(ix>=0)&(ix+1<W)&(iy>=0)&(iy+1<H)
 for oy,ox,b in [(0,0,(1-dx)*(1-dy)),(0,1,dx*(1-dy)),(1,0,(1-dx)*dy),(1,1,dx*dy)]:
  value=a[np.clip(iy+oy,0,H-1),np.clip(ix+ox,0,W-1)]
  out+=value*(b*inside)[(...,)+(None,)*len(tail)]
 return out,inside
def four(a,uv):
 x,y=uv[...,0],uv[...,1];finite=np.isfinite(x)&np.isfinite(y);x=np.where(finite,x,0);y=np.where(finite,y,0)
 ix,iy=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64);H,W=a.shape
 valid=finite&(ix>=0)&(ix+1<W)&(iy>=0)&(iy+1<H)
 for oy,ox in [(0,0),(0,1),(1,0),(1,1)]:
  valid&=a[np.clip(iy+oy,0,H-1),np.clip(ix+ox,0,W-1)]
 return valid
def lowpass(rgb,valid,sigma):
 den=gaussian_filter(valid.astype(np.float64),sigma,mode='constant',truncate=3)
 num=np.stack([gaussian_filter(rgb[...,i]*valid,sigma,mode='constant',truncate=3) for i in range(3)],-1)
 return np.divide(num,den[...,None],out=np.zeros_like(num),where=den[...,None]>1e-12)
def camera(bank,raw):
 receipt=json.loads((bank/'receipt.json').read_text());frame=next(r['id'] for r in receipt['frames'] if r['raw']==raw)
 obs=json.loads((bank/'observations'/('%03d.json'%raw)).read_text())
 lines=(bank/'calib'/(frame+'.txt')).read_text().splitlines()
 cal={r.split(':')[0]:np.fromstring(r.split(':')[1],sep=' ') for r in lines}
 K=cal['P0'].reshape(3,4)[:,:3].copy();K[:2]*=480/obs['camera_size'][0]
 T=np.eye(4);T[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);T=T@np.linalg.inv(np.asarray(obs['ego_world']).reshape(4,4))
 return K,T
def world_view(xyz,d,K,T,max_world_error=.05):
 shape=xyz.shape[:-1];world=xyz.reshape(-1,3);pc=world@T[:3,:3].T+T[:3,3]
 with np.errstate(invalid='ignore',divide='ignore'):uv=(pc@K.T)[:,:2]/pc[:,2:3]-.5
 base=(pc[:,2]>0)&np.isfinite(world).all(1)&(np.linalg.norm(world-np.linalg.inv(T)[:3,3],axis=1)<=60)
 ground=d['ground_world_xyz'];gp=ground@T[:3,:3].T+T[:3,3]
 inv=np.divide(1,gp[...,2],out=np.zeros(gp.shape[:2]),where=gp[...,2]>1e-8)
 iz,_=sample(inv,uv);ray=np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(K).T
 depth=np.divide(1,iz,out=np.zeros_like(iz),where=iz>1e-12);hit=(ray*depth[:,None]-T[:3,3])@T[:3,:3]
 err=np.linalg.norm(world-hit,axis=1);base&=err<=max_world_error
 result={'uv':uv.reshape(shape+(2,)),'world_error':err.reshape(shape)}
 for role,mask,key in [('A',d['coverage_A'],'target_A_srgb'),('C',d['coverage_C'],'target_C_srgb'),('S',d['coverage_S'],'target_S_srgb')]:
  permission=base&four(mask&~d['K'],uv)
  rgb,_=sample(np.asarray(d[key]).transpose(1,2,0),uv)
  result[role+'_valid']=permission.reshape(shape);result[role+'_rgb']=rgb.reshape(shape+(3,))
 return result
def merge_evidence(views,geometry,guard,threshold):
 val=[v['C_valid']|v['S_valid'] for v in views];av=[v['A_valid'] for v in views]
 rgb=[np.where(v['C_valid'][...,None],v['C_rgb'],v['S_rgb']) for v in views]
 count=val[0].astype(np.uint8)+val[1];acount=av[0].astype(np.uint8)+av[1]
 teacher=np.divide(rgb[0]*val[0][...,None]+rgb[1]*val[1][...,None],count[...,None],out=np.zeros_like(rgb[0]),where=count[...,None]>0)
 ar=np.divide(views[0]['A_rgb']*av[0][...,None]+views[1]['A_rgb']*av[1][...,None],acount[...,None],out=np.zeros_like(rgb[0]),where=acount[...,None]>0)
 dual=val[0]&val[1];delta=np.max(np.abs(rgb[0]-rgb[1]),axis=-1);aconf=(acount>0)&(count>0)&(np.max(np.abs(teacher-ar),axis=-1)>threshold)
 conflict=(dual&(delta>threshold))|aconf
 valid=(count>0)&geometry&~guard&~conflict
 role=np.zeros(count.shape,np.uint8)
 for v in views:role|=v['C_valid'].astype(np.uint8)|v['S_valid'].astype(np.uint8)*2
 return dict(teacher_C_support=views[0]['C_valid']|views[1]['C_valid'],teacher_S_support=views[0]['S_valid']|views[1]['S_valid'],
  teacher_evidence_valid=valid,teacher_view_count=count,teacher_role_bits=role,teacher_conflict=conflict,
  crossview_both_valid=dual,crossview_RGB_max=delta,teacher_vs_realA_conflict=aconf,
  realA_support=acount>0,teacher_rgb_srgb=teacher,realA_rgb_srgb=ar,
  teacher_H_allow=valid,teacher_material_valid=valid)
def permission_query_cpu(base,prior_valid,seed,teacher_valid,low,high,gate,uv,weights,geometry_valid):
 """小patch真实CPU autograd接线。新有效质量不再乘回legacy valid/known/rawmask。"""
 import torch
 if any(t.device.type!='cpu' for t in [base,prior_valid,seed,teacher_valid,low,high,gate,uv,weights,geometry_valid]):
  raise ValueError('CPU tensors only')
 import torch.nn.functional as F
 def bilinear(value):
  H,W,C=value.shape;grid=torch.stack((uv[...,0]*2/(W-1)-1,uv[...,1]*2/(H-1)-1),-1)[None]
  return F.grid_sample(value.double().permute(2,0,1)[None],grid.double(),align_corners=True,padding_mode='zeros')[0].permute(1,2,0)
 effective=prior_valid|teacher_valid
 frozen=torch.where(prior_valid[...,None],base,torch.where(teacher_valid[...,None],seed,torch.zeros_like(seed)))
 v=bilinear(effective[...,None].double())*geometry_valid[...,None]
 num=bilinear(frozen*effective[...,None])*geometry_valid[...,None]+v*(bilinear(low)+bilinear(gate[...,None].double())*bilinear(high))
 mass=(weights[None,:,None]*v).sum(1)
 return ((weights[None,:,None]*num).sum(1)/mass.clamp_min(1e-12)).float(),mass[...,0]
def highpass_cpu(raw,common_latent_mask,bound=COMMON_HIGH_BOUND,sigma_cells=3.):
 import torch
 import torch.nn.functional as F
 if raw.device.type!='cpu':raise ValueError('CPU-only preparation')
 value=torch.tanh(raw)*common_latent_mask[...,None]
 radius=int(np.ceil(3*sigma_cells));x=torch.arange(-radius,radius+1,dtype=value.dtype)
 kernel=torch.exp(-.5*(x/sigma_cells)**2);kernel/=kernel.sum();C=value.shape[-1]
 a=value.permute(2,0,1)[None]
 a=F.conv2d(F.pad(a,(radius,radius,0,0),mode='replicate'),kernel[None,None,None,:].expand(C,1,1,-1),groups=C)
 a=F.conv2d(F.pad(a,(0,0,radius,radius),mode='replicate'),kernel[None,None,:,None].expand(C,1,-1,1),groups=C)
 return .5*bound*(value-a[0].permute(1,2,0))
def main(output,conflict_codes=4):
 output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 a=load(P/'cpu_exact_r1/A0.npz');b=load(BASE,['surface_world_xyz','grid_world_row'])
 prior=load(N/'provenance.npz',['known','unknown','source','surface_supported','guarded_world_rows'])
 road=load(N/'road.npz',['active','minimum_model_depth']);old=load(P/'method_scope_r1/scope.npz');native=load(P/'stage0_native_r1/scope.npz')
 grid=b['grid_world_row'];geometry=prior['surface_supported'][grid]&np.isfinite(b['surface_world_xyz']).all(2)
 guard=prior['guarded_world_rows'].copy();guard[native['active_world_rows']]|=native['conservative_K_guard'];nodeguard=guard[grid]
 bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
 views=[];source_files=[P/'cpu_exact_r1/A0.npz',BASE,N/'provenance.npz',N/'road.npz',P/'method_scope_r1/scope.npz',P/'stage0_native_r1/scope.npz']
 for raw in [86,110]:
  d=load(V/('%03d.npz'%raw));K,T=camera(bank,raw);v=world_view(b['surface_world_xyz'],d,K,T)
  v['raw']=raw;views.append(v);source_files.append(V/('%03d.npz'%raw))
  np.savez_compressed(output/('%03d_evidence.npz'%raw),**{k:v for k,v in v.items() if k!='raw'},K=K,world_to_camera=T)
 evidence=merge_evidence(views,geometry,nodeguard,conflict_codes/255)
 valid=evidence['teacher_evidence_valid'];seed_source=np.where(evidence['realA_support'][...,None],evidence['realA_rgb_srgb'],evidence['teacher_rgb_srgb'])
 seed=lowpass(seed_source,valid,6.) # sigma12cm on inherited approximately2cm chart
 seed_mask=valid&~a['valid'];common_base=np.where(a['valid'][...,None],a['rgb_srgb'],np.where(seed_mask[...,None],seed,0))
 common_valid=a['valid']|seed_mask
 observed=(prior['known'][grid]&(prior['source'][grid]==1)&a['valid'])
 mHobs=a['mH'].copy();teacher_gate=valid.astype(np.float64);mHunion=np.maximum(mHobs,teacher_gate)
 # same latent/regularizer support for both arms; no source1 or legacy-known reset.
 common_reg=(mHobs>0)|valid
 common_latent=(maximum_filter(common_reg.astype(np.uint8),size=19,mode='constant')>0)&geometry&~nodeguard&((mHobs>0)|valid)
 active=road['active'][grid];near=road['minimum_model_depth'][grid]>0;rowold=np.zeros(len(road['active']),bool);rowold[old['active_world_rows']]=old['eligible']
 common_write_candidate=(rowold[grid]|valid)&active&geometry&~nodeguard&~near&common_valid
 np.savez_compressed(output/'teacher_evidence.npz',world_row_grid=grid,world_xyz=b['surface_world_xyz'],**evidence,
  geometry_support=geometry,guard=nodeguard,old_scope=rowold[grid],old_A0_valid=a['valid'],
  original_source=prior['source'][grid],original_known=prior['known'][grid],original_unknown=prior['unknown'][grid],
  real_observed_support=observed|evidence['realA_support'],observed_H_prior=mHobs,teacher_gate=teacher_gate,mH_union=mHunion,
  common_latent_mask=common_latent,common_reg_support=common_reg,common_write_candidate=common_write_candidate,
  base_seed_ledger_mask=seed_mask,base_seed_lowpass_srgb=seed,common_base_srgb=common_base,common_material_valid=common_valid,
  base_seed_source_role=np.where(seed_mask,np.where(evidence['realA_support'],1,3),0).astype(np.uint8))
 oldrow=load(Q/'permission/old_audit_r1/row_H_support.npz')
 yy,xx=np.unravel_index(np.arange(grid.size),grid.shape);inverse=np.empty(len(road['active']),np.int64);inverse[grid.ravel()]=np.arange(grid.size)
 candidate=common_write_candidate.ravel()[inverse[oldrow['active_world_rows']]]
 current=old['eligible'];unknown=~native['material_available'];C=oldrow['C_mass']>0;S=oldrow['S_mass']>0
 summary=dict(chart_nodes=int(grid.size),teacher_valid_nodes=int(valid.sum()),C_nodes=int(evidence['teacher_C_support'].sum()),S_nodes=int(evidence['teacher_S_support'].sum()),
  teacher_conflict_nodes=int(evidence['teacher_conflict'].sum()),dual_valid_nodes=int(evidence['crossview_both_valid'].sum()),
  teacher_new_H_where_old_gate_zero=int((valid&(mHobs==0)).sum()),base_seed_prior_invalid_nodes=int(seed_mask.sum()),
  base_seed_active_nodes=int((seed_mask&active).sum()),common_write_active_candidate=int(common_write_candidate.sum()),
  newly_write_candidate=int((candidate&~current).sum()),newly_write_C_prior_unavailable=int((candidate&~current&C&unknown).sum()),
  newly_write_C_original_known=int((candidate&~current&C&prior['known'][oldrow['active_world_rows']]).sum()),
  newly_write_S=int((candidate&~current&S).sum()),C_prior_unavailable_not_qualified=int((C&unknown&~candidate).sum()),
  crossview_threshold_counts={str(code):int((evidence['crossview_both_valid']&(evidence['crossview_RGB_max']>code/255)).sum()) for code in [1,2,4,8]})
 receipt=dict(schema='independent_teacher_H_permission_CPU_candidate.v1',task='MODULE2-TEACHER-DETAIL-CPU-20261119',
  status='CPU_CANDIDATE_PENDING_ROOT_SCOPE_CONTRACT',CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,
  counts=summary,conflict_max_channel_sRGB_codes=conflict_codes,single_view_validity_is_not_calibrated_correctness=True,
  sampler='frozen480 target bilinear, all4 lowres neighbors individually inherit original full4x4 native permission; runtime pinhole pixel centres; perspective-correct ground depth check .05m; no renderer imported',
  material_rule='A0 valid values exact; prior-invalid but geometry+teacher qualified only receives sigma12cm lowpass frozen seed, not teacherHF; original source-known-unknown unchanged',
  common_factor_rule='same common base/material_valid/write candidate/latent/reg support allarms; only output mH differs old observed vs union teacher. Root must finalize scope before any GPU.',
  highpass_basis_note='common_latent_mask does not multiply legacy source/known/valid, teacher_H permit independent; no H initial tensor or optimizer created',
  inherited_H_bound=.04,lowpass_seed_sigma_chart_cells=6,source_grid_approx_m=.02,
  full_evaluation_denominators_unchanged=dict(C=10830,S=3131,Sfull=3187,D=38416,K=77266),
  original_inputs_unmodified=True,source_sha256={str(path):sha(path) for path in source_files},seconds=time.monotonic()-started)
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--conflict-codes',type=float,default=4)
 args=ap.parse_args();main(args.output,args.conflict_codes)

