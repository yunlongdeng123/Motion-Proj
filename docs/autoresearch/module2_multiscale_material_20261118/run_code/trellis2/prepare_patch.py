"""本轮唯一固定道路patch：只读当前几何与86/110合法参考，CPU-only。"""
from pathlib import Path
import os,sys,json,hashlib,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from PIL import Image
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');P=B/'road_multiscale_material_20261118_r1';N=B/'road_activation_20261118_r1';C=B/'road_capacity_cpu_20261117_r1'
sys.path.insert(0,str(N/'code'))
from prepare_activation_supervision import load,write,corner_values,source_roles,choose_source

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def transform(p,T):return p@T[:3,:3].T+T[:3,3]
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only')
 out=P/'trellis2/input_r1';out.mkdir(parents=True,exist_ok=False)
 parent=N/'B32_beam95_native_r1/asset_guarded'
 r=load(parent/'road.npz',['model_xyz','model_from_world','normal_world'])
 p=load(parent/'provenance.npz',['world_keys','world_xyz','surface_supported'])
 k=p['world_keys'];selected=(k[:,0]>=-99)&(k[:,0]<=28)&(k[:,1]>=-66)&(k[:,1]<=61)
 ids=np.flatnonzero(selected);grid=k[ids];world=p['world_xyz'][ids];normal=r['normal_world'][ids]
 assert len(ids)==16384 and p['surface_supported'][ids].all()
 assert np.array_equal(grid,np.stack(np.meshgrid(np.arange(-99,29),np.arange(-66,62),indexing='ij'),-1).reshape(-1,2))
 actual=transform(r['model_xyz'][ids],np.linalg.inv(r['model_from_world']));world_error=float(np.abs(world-actual).max());assert world_error<1e-7
 ni=normal.mean(0);ni/=np.linalg.norm(ni);u=world[128]-world[0];u-=ni*np.dot(u,ni);u/=np.linalg.norm(u);z=np.cross(u,ni);axes=np.stack([u,ni,z]);assert np.linalg.det(axes)>.999999
 local=world@axes.T;center=(local.min(0)+local.max(0))/2;scale=.98/np.ptp(local,axis=0).max();W=np.eye(4);W[:3,:3]=axes*scale;W[:3,3]=-center*scale;canonical=transform(world,W)
 c=(canonical.min(0)+canonical.max(0))/2;s=.99999/np.ptp(canonical,axis=0).max();norm=np.eye(4);norm[:3,:3]*=s;norm[:3,3]=-c*s
 output_to_world=np.linalg.inv(W)@np.linalg.inv(norm);expected_output=transform(canonical,norm)
 i,j=np.mgrid[:127,:127];a=(i*128+j).ravel();faces=np.stack([np.c_[a,a+128,a+1],np.c_[a+128,a+129,a+1]],1).reshape(-1,3).astype(np.int32)
 with (out/'canonical_yup_open.obj').open('w') as f:
  f.write('# current fixed Gaussian world centers, open road sheet, no new geometry\n')
  for v in canonical:f.write('v '+' '.join(format(float(x),'.17g') for x in v)+'\n')
  for face in faces:f.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
 np.savez_compressed(out/'fixed_geometry.npz',source_world_rows=ids,world_keys=grid,world_vertices=world,canonical_yup_vertices=canonical,faces=faces,expected_official_output_yup=expected_output)
 write(out/'transforms.json',dict(world_to_canonical_yup=W.tolist(),official_normalization=norm.tolist(),official_output_yup_to_world=output_to_world.tolist(),world_bbox=[world.min(0).tolist(),world.max(0).tolist()],mean_normal_world=ni.tolist()))
 spec=importlib.util.spec_from_file_location('frozen_Brown',B/'road_material_field_20261112_r1/source/world_feature_field.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 cfg=json.loads((B/'scene065_material_loop_20261114_r1/config.json').read_text());ev=[];sources=[];coverage=[]
 for raw in [86,110]:
  case=f'scene065_raw{raw:03d}';prep=B/'road_mask_decoupling_20261116_r1/prepared'/case;teacher=B/'road_mask_decoupling_20261116_r1/predictions'/case/'B/raw_native.png'
  obs=Path(cfg['bank'])/f'observations/{raw:03d}.json';cam=h.camera_from_observation(json.loads(obs.read_text()));uv,depth,inside=h.project_world_to_camera(world,**{key:cam[key] for key in ['camera_K_native','world_to_camera','image_hw','distortion_Brown']})
  d=np.linalg.inv(cam['world_to_camera'])[:3,3]-world;dist=np.linalg.norm(d,axis=1);cos=np.maximum(np.sum(d*normal,axis=1)/dist,0)
  roles=load(prep/'B/roles_native.npz',['road_observed','C_mat','visible_support']);common=load(prep/'common_native.npz',['physical','H_B','K'])
  guard_path=B/'road_mask_decoupling_20261116_r1/output_guard_probe'/case/'native_guard_roles.npz';guard=load(guard_path,['W_guard','K'])
  route_path=C/f'supervision_source_identity_r1/{raw:03d}_native_routing_evidence.npz';route=load(route_path,['target_shadow_interior','other_shadow_prediction','original_teacher_W_guard']);assert np.array_equal(guard['W_guard'],route['original_teacher_W_guard'])
  A=roles['road_observed']&roles['C_mat']&roles['visible_support']&~common['H_B']&~common['K']
  e=source_roles(inside,dist,cos,corner_values(guard['W_guard'],uv),corner_values(guard['K'],uv),corner_values(A,uv),corner_values(common['physical'],uv),corner_values(route['target_shadow_interior'],uv),corner_values(route['other_shadow_prediction'],uv))
  role=np.where(e['A_permission'],1,np.where(e['C_permission'],2,np.where(e['S_permission'],3,0))).astype(np.uint8)
  image=np.asarray(Image.open(teacher).convert('RGB'),np.float32)/255;tc=h.bilinear_sample(image,uv,cam['image_hw']);del image
  image=np.asarray(Image.open(prep/'observation.png').convert('RGB'),np.float32)/255;ac=h.bilinear_sample(image,uv,cam['image_hw']);del image
  ev.append(dict(auth=e['authorized'],role=role,rgb=np.where((role==1)[:,None],ac,tc),uv=uv,score=cos/(dist**2+4)))
  sources.append(dict(raw=raw,teacher=str(teacher),teacher_sha256=sha(teacher),observation=str(prep/'observation.png'),Brown_calibration=str(obs),original_teacher_guard=str(guard_path),trusted_shadow=str(route_path),qualified_reference_rows=int(e['authorized'].sum())))
  v=load(C/f'supervision_v4_candidate_r1/{raw:03d}.npz',['S_target_shadow','S_target_shadow_photo','triangle_world_rows','triangle_weights'])
  alpha=load(N/f'B32_beam95_native_r1/{raw:03d}_native.npz',['road_alpha'])['road_alpha'];tri=v['triangle_world_rows'];weight=v['triangle_weights'];valid=(tri>=0).all(2);pm=np.zeros(valid.shape,bool);pm[valid]=np.all(selected[tri[valid]]|(weight[valid]<=1e-7),axis=1)
  S=v['S_target_shadow'];new=S&~v['S_target_shadow_photo']&(alpha>.9);coverage.append(dict(raw=raw,fullS=int(S.sum()),patch_fullS=int((S&pm).sum()),new_supported=int(new.sum()),patch_new_supported=int((new&pm).sum())))
  np.savez_compressed(out/f'{raw:03d}_patch_domains.npz',patch_geometry=pm,S_full=S,new_supported_S=new,K=v.get('K',np.zeros_like(S)))
 auth=np.stack([e['auth'] for e in ev]);role=np.stack([e['role'] for e in ev]);rgb=np.stack([e['rgb'] for e in ev]);score=np.stack([e['score'] for e in ev]);best,valid,ref=choose_source(auth,role,rgb,score);row=np.arange(len(ids));ref[~valid]=.5
 raw=np.where(valid,np.array([86,110],np.int16)[best],-1);chosen_role=np.where(valid,role[best,row],0).astype(np.uint8);uv=np.stack([e['uv'] for e in ev])[best,row];uv[~valid]=np.nan
 def image_grid(a):return a.reshape(128,128,*a.shape[1:]).swapaxes(0,1)[::-1]
 Image.fromarray(np.rint(image_grid(ref)*255).astype(np.uint8)).save(out/'reference_worldwarp_rgb128.png')
 Image.fromarray(image_grid(valid.astype(np.uint8))*255).save(out/'reference_valid.png')
 color=np.array([[128,128,128],[60,180,90],[245,155,40],[100,120,240]],np.uint8);Image.fromarray(image_grid(color[chosen_role])).save(out/'reference_roles.png')
 np.savez_compressed(out/'reference_sources.npz',world_rows=ids,valid=valid,unknown=~valid,source_raw=raw,source_role=chosen_role,source_uv_native=uv,rgb_srgb=ref.astype(np.float32),per_view_valid=auth,per_view_rgb_srgb=rgb,per_view_role=role,per_view_uv_native=np.stack([e['uv'] for e in ev]))
 manifest=dict(schema='trellis2_new_fixed_road_patch_input.v1',task='MODULE2-MULTISCALE-MATERIAL-20261118',status='PASS_CPU_INPUT',parent_asset=str(parent),mesh=str(out/'canonical_yup_open.obj'),reference=str(out/'reference_worldwarp_rgb128.png'),geometry=str(out/'fixed_geometry.npz'),transforms=str(out/'transforms.json'),reference_sources=str(out/'reference_sources.npz'),mesh_sha256=sha(out/'canonical_yup_open.obj'),reference_sha256=sha(out/'reference_worldwarp_rgb128.png'),
  patch=dict(world_key_min=[-99,-66],world_key_max=[28,61],vertices=16384,faces=len(faces),geometry_supported_vertices=16384,full_texel_footprint_m=[2.56,2.56],world_bbox=[world.min(0).tolist(),world.max(0).tolist()],mean_normal_world=ni.tolist(),actual_GS_world_vertex_error_m=world_error,shape_generated=False),
  source_frames=[86,110],diagnostic_frames_not_consumed=[134,179,198],sources=sources,coverage=coverage,fullS_denominator=3187,newS_denominator=909,
  reference_valid_rows=int(valid.sum()),reference_unknown_rows=int((~valid).sum()),reference_role_counts={str(int(k)):int(v) for k,v in zip(*np.unique(chosen_role,return_counts=True))},reference_raw_counts={str(int(k)):int(v) for k,v in zip(*np.unique(raw,return_counts=True))},
  reference_policy='original observed visible A first; otherwise original frozen C/target-shadow S, original Brown projection/all4 Wguard and ~K, inherited distance<60/cos>.025; no crossview averaging',unknown_reference_fill='RGB128 neutral; explicit unknown map, not claimed as observed/clean teacher',reference_image_orientation='image X=+world chart i, image Y=-world chart j; canonical mesh X=+i,Y=up, Z=-j',
  seed=101,resolution=512,texture_size=1024,sampling_steps=12,preprocess_image=False,training_updates=0,new_inference_required=True,historical_generated_output_reused=False,
  cached_reuse='weights and compiled operators only; shape/conditioning/texture must run anew for this input',inherited_BUILD_frames=199,blind_test=False,
  output_policy='baseColor prior only; generated alpha does not change GS opacity; roughness/metallic diagnostic; later worldUV/S_Sigma sampling + fresh K and actual native GS',exterior_policy='same root A0; patch source never claimed to cover full street',GPU_operations=0,human_verdict=None)
 write(out/'manifest.json',manifest);print(json.dumps(dict(manifest=str(out/'manifest.json'),reference_valid=int(valid.sum()),reference_unknown=int((~valid).sum()),roles=manifest['reference_role_counts'],coverage=coverage)))
if __name__=='__main__':main()
