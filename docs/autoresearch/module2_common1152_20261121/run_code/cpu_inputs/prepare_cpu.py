"""CPU-only canonical 1152 maps, inherited-rule teacher world evidence and fixed reads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[k]='2'
from pathlib import Path
from types import SimpleNamespace
import argparse,importlib.util,json,sys,time,hashlib
import numpy as np
from scipy.ndimage import maximum_filter
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
P=B/'road_common1152_20261121';Q=B/'road_teacher_detail_cpu_20261119_r1';M=B/'road_multiscale_material_20261118_r1'
N=B/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
BASE=B/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz'
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def maps():
 out=P/'maps';out.mkdir(exist_ok=True);road=load(N/'road.npz');a=load(BASE);prov=load(N/'provenance.npz');grid=a['grid_world_row'];xyz=a['surface_world_xyz']
 import shutil
 projection_source=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/road_surface_projection.py');projection_copy=P/'cpu_inputs/road_surface_projection.py'
 if not projection_copy.exists():shutil.copyfile(projection_source,projection_copy)
 projection=module('road_surface_projection',projection_copy)
 camera=json.loads((P/'camera_manifest.json').read_text());s=float(road['meter_to_model']);g=SimpleNamespace(grid_world_row=grid,surface_world_xyz=xyz)
 world=np.empty((len(road['active']),3));uv=np.empty((len(road['active']),2));yy,xx=np.indices(grid.shape);world[grid.ravel()]=xyz.reshape(-1,3);uv[grid.ravel()]=np.stack((xx,yy),-1).reshape(-1,2)
 rows=[]
 for c in camera['rows']:
  raw=int(c['frame_key']);dest=out/f'{raw:03d}.npz'
  if dest.exists(): rows.append({'raw':raw,'retained_existing':True});continue
  start=time.monotonic();T=np.asarray(c['W'])@road['model_from_world'];T[:3]/=s
  K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]],np.float64)
  m=projection.project_surface_mesh(g,prov['surface_supported'],K,T,(c['height'],c['width']),road['minimum_model_depth']/s)
  valid=m['valid'];ids=m['triangle_world_rows'][valid];weights=m['triangle_weights'][valid];vertices=world[ids];uvertices=uv[ids];X=(vertices*weights[...,None]).sum(1);U=(uvertices*weights[...,None]).sum(1)
  y,x=np.nonzero(valid);R=T[:3,:3];O=-R.T@T[:3,3];direction=np.c_[((np.stack((x,y),1)+.5)-K[:2,2])/np.diag(K)[:2],np.ones(len(X))]@R
  normal=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);den=(normal*direction).sum(1);t=(normal*(vertices[:,0]-O)).sum(1)/den
  Jxy=np.empty((len(X),2,2))
  for axis in (0,1):
   dr=R.T[:,axis]/K[axis,axis];dt=-t*(normal@dr)/den;Jxy[:,:,axis]=(t[:,None]*dr+direction*dt[:,None])[:,:2]
  Ex=np.stack((vertices[:,1,:2]-vertices[:,0,:2],vertices[:,2,:2]-vertices[:,0,:2]),-1);Eu=np.stack((uvertices[:,1]-uvertices[:,0],uvertices[:,2]-uvertices[:,0]),-1)
  Juv=np.einsum('nij,njk->nik',Eu@np.linalg.inv(Ex),Jxy)
  center=np.full((*valid.shape,2),-4.,np.float64);jac=np.zeros((*valid.shape,2,2));surface=np.zeros((*valid.shape,3));center[valid]=U;jac[valid]=Juv;surface[valid]=X
  parent=load(P/f'native/parent/{raw:03d}.npz');keep=parent['K'];D=valid&~keep
  recon=float(np.abs(X-m['world_xyz'][valid]).max());rayerr=float(np.abs(O+t[:,None]*direction-X).max());reproj=float(np.nanmax(m['reprojection_error_px']))
  if max(recon,rayerr)>1e-7 or reproj>1e-4 or not np.isfinite(Juv).all():raise ValueError('new exact canonical map qualification failed')
  np.savez_compressed(dest,surface_world_xyz=surface,ground_world_xyz=surface,center_uv=center,pixel_uv_jacobian=jac,canonical_mesh_hit_valid=valid,triangle_world_rows=m['triangle_world_rows'],triangle_weights=m['triangle_weights'],camera_K=K,world_to_camera=T,camera_depth=m['camera_depth_m'],fixed_evaluation_domain=D,plane_fallback=np.zeros(valid.shape,bool),K=keep,pixel_center_offset=np.asarray(.5))
  row=dict(raw=raw,shape=list(valid.shape),canonical_pixels=int(valid.sum()),K_pixels=int(keep.sum()),reconstruction_max_m=recon,ray_error_max_m=rayerr,reprojection_max_px=reproj,seconds=time.monotonic()-start);rows.append(row);print(json.dumps(row),flush=True)
 write(out/'receipt.json',dict(schema='common1152_exact_canonical_maps.v1',rows=rows,source_camera_sha256=sha(P/'camera_manifest.json'),source_atlas_sha256=sha(BASE),no_old_image_resize=True,CUDA_calls=0))
def inputs():
 out=P/'cpu_inputs';out.mkdir(exist_ok=True);rows=[]
 for raw in [86,110]:
  t=load(P/f'teacher/raw{raw:03d}/supervision.npz');m=load(P/f'maps/{raw:03d}.npz');n=load(P/f'native/parent/{raw:03d}.npz');alpha=n['road_alpha'];K=n['K']|t['K'];legal=m['canonical_mesh_hit_valid']&~K
  d={'fixed_alpha':alpha,'K':K,'canonical_valid':m['canonical_mesh_hit_valid'],'teacher_write_full':t['teacher_write_full']&legal}
  for role in ['A','C','S']:
   src=t[role]&~K;qualified=src&m['canonical_mesh_hit_valid'];d[role+'_source']=src;d[role+'_qualified']=qualified;d[role]=qualified&(alpha>.9);d[role+'_core95']=qualified&(alpha>=.95);d['target_'+role]=t['target_'+role+'_srgb'];d['q_'+role]=d[role].astype(np.float32)
  d['fullS']=d['S_qualified'];np.savez_compressed(out/f'{raw:03d}_targets.npz',**d)
  rows.append(dict(raw=raw,source={k:int(d[k+'_source'].sum())for k in ['A','C','S']},supported90={k:int(d[k].sum())for k in ['A','C','S']},core95={k:int(d[k+'_core95'].sum())for k in ['A','C','S']},teacher_sha256=sha(P/f'teacher/raw{raw:03d}/supervision.npz')))
 footprints=[]
 for raw in [86,110]:
  d=load(out/f'{raw:03d}_targets.npz');m=load(P/f'maps/{raw:03d}.npz');J=m['pixel_uv_jacobian'][d['C']|d['S']];footprints.append(np.sqrt(np.sum(J*J,axis=-1)/12))
 sigma=np.median(np.concatenate(footprints),axis=0)
 write(out/'source_footprint.json',dict(schema='common1152_source_footprint.v1',sigma_uv=sigma.tolist(),sigma_u=float(sigma[0]),sigma_v=float(sigma[1]),rule='median over fixed supported C union S in raw86/110 of sqrt(sum_pixelaxes(J_uv[axis,:]^2)/12)',resolution=[1152,768],source_fields='maps/{raw:03d}.npz pixel_uv_jacobian; cpu_inputs/{raw:03d}_targets.npz C|S',no_student_error_selection=True))
 write(out/'targets_manifest.json',dict(schema='common1152_training_targets.v1',resolution=[1152,768],rows=rows,training_masks='2D source & canonical valid & ~K & fixed parent alpha>.9',evaluation_masks='stable color: same canonical domain parent alpha>=.95; blackgap fixed denominator: complete legal2D source & ~K, canonical missing retained',teacher_manifest=str(P/'teacher/manifest.json')))
 print(json.dumps(rows),flush=True)
def permission():
 code=Q/'permission/code';sys.path.insert(0,str(code));from teacher_detail_permission import world_view,lowpass
 out=P/'common_permission';out.mkdir(exist_ok=True);old=load(Q/'permission/final_r1/teacher_evidence.npz');a=load(M/'cpu_exact_r1/A0.npz');b=load(BASE);grid=b['grid_world_row'];geom=old['geometry_support'];guard=old['guard'];safe=geom&~guard;views=[]
 for raw in [86,110]:
  t=load(P/f'teacher/raw{raw:03d}/supervision.npz');m=load(P/f'maps/{raw:03d}.npz');d={'ground_world_xyz':m['surface_world_xyz'],'K':t['K']|m['K']}
  for role in ['A','C','S']:d['coverage_'+role]=t[role]&m['canonical_mesh_hit_valid'];d['target_'+role+'_srgb']=t['target_'+role+'_srgb']
  v=world_view(b['surface_world_xyz'],d,m['camera_K'],m['world_to_camera']);views.append(v);np.savez_compressed(out/f'{raw:03d}_evidence.npz',**v)
 from scipy.ndimage import gaussian_filter
 rgb=[];valid=[];hf=[];mass=[];lf=[];lm=[];ahf=[];am=[];alf=[]
 for v in views:
  mask=(v['C_valid']|v['S_valid'])&safe;value=np.where(v['C_valid'][...,None],v['C_rgb'],v['S_rgb']);low=lowpass(value,mask,3);low12=lowpass(value,mask,6);av=v['A_valid']&safe;al=lowpass(v['A_rgb'],av,3)
  valid.append(mask);rgb.append(value);hf.append(value-low);mass.append(gaussian_filter(mask.astype(float),3,mode='constant',truncate=3));lf.append(low12);lm.append(gaussian_filter(mask.astype(float),6,mode='constant',truncate=3));ahf.append(v['A_rgb']-al);am.append(gaussian_filter(av.astype(float),3,mode='constant',truncate=3));alf.append(lowpass(v['A_rgb'],av,6))
 both=valid[0]&valid[1];comparable=both&(mass[0]>=.9)&(mass[1]>=.9);conflict=comparable&(np.max(np.abs(hf[0]-hf[1]),-1)>4/255)
 owner=np.where(valid[1]&(~valid[0]|(mass[1]>=mass[0])),1,0);targethf=np.where(owner[...,None]==0,hf[0],hf[1]);owner_mass=np.where(owner==0,mass[0],mass[1]);aconflict=np.zeros(safe.shape,bool)
 for i,v in enumerate(views):aconflict|=v['A_valid']&(am[i]>=.9)&(owner_mass>=.9)&(np.max(np.abs(targethf-ahf[i]),-1)>4/255)
 material=(valid[0]|valid[1])&safe;allow=material&~conflict&~aconflict;base_owner=np.where(valid[1]&(~valid[0]|(lm[1]>=lm[0])),1,0);seed=np.where(base_owner[...,None]==0,lf[0],lf[1]);realA=views[0]['A_valid']|views[1]['A_valid'];aowner=np.where(views[1]['A_valid']&(~views[0]['A_valid']|(am[1]>=am[0])),1,0);seed=np.where(realA[...,None],np.where(aowner[...,None]==0,alf[0],alf[1]),seed)
 seedmask=material&~a['valid'];base=np.where(a['valid'][...,None],a['rgb_srgb'],np.where(seedmask[...,None],seed,0));cv=a['valid']|seedmask;obs=a['mH'];union=np.maximum(obs,allow.astype(float));prior=old['prior_observed_support']
 import torch
 torch.set_num_threads(1);saved=torch.load(M/'formal_r1/A200/asset/world_field.pt',map_location='cpu',mmap=True,weights_only=False);legacy_reg=saved['state']['high_output_mask'][...,0].numpy().copy()>0;del saved
 reg=(legacy_reg|allow)&safe;latent=(maximum_filter(reg.astype(np.uint8),size=19,mode='constant')>0)&safe&(prior|material)
 rs=load(P/'native/basis/086/row_scope.npz');rows=rs['active_world_rows'];inverse=np.empty(grid.size,np.int64);inverse[grid.ravel()]=np.arange(grid.size);at=inverse[rows];oldscope=load(Q/'permission/old_audit_r1/row_H_support.npz')['old_eligible'];writeactive=(oldscope|material.ravel()[at])&rs['eligible_active']&cv.ravel()[at];writegrid=np.zeros(grid.size,bool);writegrid[at]=writeactive;writegrid=writegrid.reshape(grid.shape)
 evidence=dict(world_row_grid=grid,world_xyz=b['surface_world_xyz'],geometry_support=geom,guard=guard,common_base_srgb=base,common_material_valid=cv,observed_H_prior=obs,mH_union=union,teacher_gate=allow.astype(float),teacher_H_allow=allow,teacher_material_valid=material,teacher_evidence_valid=material,teacher_C_support=views[0]['C_valid']|views[1]['C_valid'],teacher_S_support=views[0]['S_valid']|views[1]['S_valid'],teacher_HF_conflict=conflict,teacher_A_HF_conflict=aconflict,teacher_conflict=conflict|aconflict,teacher_hf_srgb=targethf,teacher_hf_owner_raw=np.where(owner==0,86,110).astype(np.int16),base_seed_owner_raw=np.where(base_owner==0,86,110).astype(np.int16),base_seed_ledger_mask=seedmask,base_seed_lowpass_srgb=seed,common_latent_mask=latent,common_reg_support=reg,common_write_candidate=writegrid,common_write_active=writeactive,active_world_rows=rows,realA_support=realA,real_observed_support=realA,prior_observed_support=prior)
 np.savez_compressed(out/'teacher_evidence.npz',**evidence)
 write(out/'receipt.json',dict(schema='common1152_reprojected_permission.v1',resolution=[1152,768],base_A0_exact=bool(np.array_equal(base[a['valid']],a['rgb_srgb'][a['valid']])),teacher_material_nodes=int(material.sum()),H_allowed_nodes=int(allow.sum()),HF_conflict_nodes=int(conflict.sum()),realA_HF_conflict_nodes=int(aconflict.sum()),common_write_active=int(writeactive.sum()),native_eligible=int(rs['eligible_active'].sum()),old_permission_used_only_for='frozen geometry, guards, old A0/observed prior and scope; no old teacher RGB/evidence retained',rule='same world_view all4 source neighbors/depth .05m; sigma3cells HF/6cells LF; HF conflict4/255 support.9; owner max support tie110',teacher_manifest_sha256=sha(P/'teacher/manifest.json'),CUDA_calls=0))
 print((out/'receipt.json').read_text(),flush=True)
def operators():
 compiler=module('common1152_fixed_read',Q/'code/fixed_material_read_operator.py');stream=module('common1152_pixel_stream',Q/'pixel_render/code/pixel_quadrature_stream.py');ledger=P/'common_permission/teacher_evidence.npz';d=load(ledger);out=P/'pixel_operators';out.mkdir(exist_ok=True);rows=[]
 for raw in [86,110,134,179,198]:
  dest=out/f'{raw:03d}'
  if dest.exists():rows.append({'raw':raw,'retained_existing':True});continue
  start=time.monotonic();source=P/f'maps/{raw:03d}.npz';m=load(source);selected=np.flatnonzero(m['canonical_mesh_hit_valid']);c=compiler.ReadCompiler(d['common_base_srgb'],d['common_material_valid'],{'observed':d['observed_H_prior'],'teacher':d['mH_union']},low_stride=6);ids=[];samples=0
  for pix,uv,w,meta in stream.iter_pixel_chunks(m['center_uv'],m['pixel_uv_jacobian'],selected,density=2):c.add(uv,w);ids.append(pix);samples+=meta['points']
  identity=np.concatenate(ids)
  if not np.array_equal(np.sort(identity),selected):raise ValueError('pixel identities lost')
  op=c.finish(identity,dict(reader='pixel_world_local_box',raw=raw,density=2,UV_axes='outputUV,inputPixel',map_source=str(source),map_sha256=sha(source),field_ledger=str(ledger),field_ledger_sha256=sha(ledger),quadrature='nx ceil2normJx,ny ceil2normJy; wholepixel,no cap',no_gaussian_Sigma_in_pixel_filter=True,camera_cache_not_world_asset=True,invalid_G_s_fallback='retain native eligible averaged-ci contribution'));op.save(dest)
  row=dict(raw=raw,pixels=len(identity),samples=samples,low_nnz=op.KL.nnz,high_nnz={k:v.nnz for k,v in op.KH.items()},zero_material_mass=int((op.mass<=1e-12).sum()),seconds=time.monotonic()-start);rows.append(row);print(json.dumps(row),flush=True);del op,c
 write(out/'receipt.json',dict(schema='common1152_pixel_fixed_reads.v1',rows=rows,density=2,CUDA_calls=0,native_calls=0))
def average():
 compiler=module('common1152_fixed_read_avg',Q/'code/fixed_material_read_operator.py');d=load(P/'common_permission/teacher_evidence.npz');cached=Q/'permission/avg_plan_gh31_r1';plan=load(cached/'geometry_plan.npz');oldrows=plan['query_world_rows'];positions={int(r):i for i,r in enumerate(oldrows)};active=np.flatnonzero(d['common_write_active']);rows=d['active_world_rows'][active]
 if not all(int(r)in positions for r in rows):raise ValueError('new writable rows absent from immutable GH31 geometry cache')
 at=np.asarray([positions[int(r)]for r in rows]);uv=np.load(cached/'sample_uv_f64.npy',mmap_mode='r');valid=np.load(cached/'geometric_sample_valid.npy',mmap_mode='r');c=compiler.ReadCompiler(d['common_base_srgb'],d['common_material_valid'],{'observed':d['observed_H_prior'],'teacher':d['mH_union']},low_stride=6);start=time.monotonic()
 for begin in range(0,len(rows),256):
  take=at[begin:begin+256];c.add(uv[take],plan['weights'],valid[take])
  if begin%8192==0:print(json.dumps(dict(average_rows_done=min(begin+256,len(rows)),total=len(rows),seconds=time.monotonic()-start)),flush=True)
 out=P/'common_permission/average_operator';op=c.finish(rows,dict(mode='Gaussian_footprint_average',kernel_order=31,quadrature_points_per_row=961,field_ledger=str(P/'common_permission/teacher_evidence.npz'),field_ledger_sha256=sha(P/'common_permission/teacher_evidence.npz'),geometry_cache=str(cached),geometry_cache_recipe_sha256=sha(cached/'recipe.json'),same_physical_geometry=True,readonly_active_rows_retain_original_RGB=True));op.save(out)
 chosen=np.linspace(0,len(rows)-1,min(64,len(rows)),dtype=int);u=uv[at[chosen]];v=valid[at[chosen]];mass=compiler.bilinear(d['common_material_valid'],u)[...,0]*v;weight=plan['weights'][None];den=(mass*weight).sum(1);expected=(compiler.bilinear(d['common_base_srgb']*d['common_material_valid'][...,None],u)*v[...,None]*weight[...,None]).sum(1)/np.maximum(den[:,None],1e-12);error=float(np.abs(expected-op.base[chosen]).max())
 if error>1e-10:raise ValueError('GH31 recompiled base direct check failed')
 np.savez_compressed(P/'common_permission/row_mapping.npz',query_world_rows=rows,active_local_indices=active,full_active_world_rows=d['active_world_rows'])
 write(P/'common_permission/average_receipt.json',dict(schema='common1152_recompiled_GH31.v1',completed=True,rows=len(rows),order=31,base_direct_max=error,old_geometry_cache_only=True,new_base_and_masks_recompiled=True,seconds=time.monotonic()-start,CUDA_calls=0))
def finalize():
 checks=[P/'maps/receipt.json',P/'cpu_inputs/targets_manifest.json',P/'common_permission/receipt.json',P/'common_permission/average_receipt.json',P/'pixel_operators/receipt.json',P/'cpu_inputs/source_footprint.json']
 if not all(x.exists()for x in checks):raise ValueError('CPU dependencies incomplete')
 data=dict(schema='common1152_cpu_inputs.v1',complete=True,permission_ledger=str(P/'common_permission/teacher_evidence.npz'),average_operator=str(P/'common_permission/average_operator'),average_row_mapping=str(P/'common_permission/row_mapping.npz'),pixel_operator_root=str(P/'pixel_operators'),targets_root=str(P/'cpu_inputs'),source_footprint_json=str(P/'cpu_inputs/source_footprint.json'),maps_root=str(P/'maps'),teacher_manifest=str(P/'teacher/manifest.json'),source_sha256={str(x.relative_to(P)):sha(x)for x in checks},CPU_only=True)
 write(P/'cpu_inputs/manifest.json.tmp',data);(P/'cpu_inputs/manifest.json.tmp').replace(P/'cpu_inputs/manifest.json');print(json.dumps(data),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['maps','inputs','permission','operators','average','finalize','all']);a=ap.parse_args()
 for name,fn in [('maps',maps),('inputs',inputs),('permission',permission),('average',average),('operators',operators),('finalize',finalize)]:
  if a.phase in ['all',name]:fn()
