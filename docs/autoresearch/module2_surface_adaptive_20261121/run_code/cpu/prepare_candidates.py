"""CPU-only missing-alpha diagnosis and fixed-road-surface Gaussian proposals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from types import SimpleNamespace
import json,time,hashlib,shutil,sys,importlib.util
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');OLD=B/'road_common1152_20261121';P=B/'road_surface_adaptive_20261121_r1';N=OLD/'formal/r3/G_NUMERIC/asset';BASE=B/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz'
FRAMES=[86,110,134,179,198];ACT_CAP=4096;APPEND_CAP=8192
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def hwc(a):return a.transpose(1,2,0)if a.shape[0]==3 else a
def frames(n):
 n=np.asarray(n,float);n/=np.linalg.norm(n,axis=-1,keepdims=True);u=np.empty_like(n);v=np.empty_like(n)
 for i,z in enumerate(n):
  a=np.cross([0.,0.,1.],z);K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]);R=np.eye(3)+K+K@K/max(1+z[2],1e-12)if z[2]>-1+1e-8 else np.diag([1.,-1.,-1.]);u[i]=R[:,0];v[i]=R[:,1]
 return n,u,v
def project(X,m):
 T=m['world_to_camera'];x=np.asarray(X)@T[:3,:3].T+T[:3,3];K=m['camera_K'];uv=np.divide(x[:,:2],x[:,2:],out=np.full((len(x),2),1e10),where=x[:,2:]>0)*np.diag(K)[:2]+K[:2,2]-.5;return uv,x
def main():
 started=time.monotonic();out=P/'proposal';out.mkdir(parents=True,exist_ok=True);(out/'domains').mkdir(exist_ok=True);road=load(N/'road.npz');prior=load(N/'provenance.npz');atlas=load(BASE);grid=atlas['grid_world_row'];world=prior['world_xyz'];shape=grid.shape
 maps={r:load(OLD/f'maps/{r:03d}.npz')for r in FRAMES};oldmaps=maps.copy();native={r:load(OLD/f'exports/r3/G_NUMERIC/{r:03d}.npz')for r in FRAMES};teachers={r:load(OLD/f'teacher/raw{r:03d}/supervision.npz')for r in [86,110]};fit={};retained={};evidence=[];counts=[];training_world=[]
 # Pure geometry audit rerasterizes finite coordinates, without conferring a surface certificate.
 projection_copy=P/'cpu/road_surface_projection.py';projection_copy.parent.mkdir(exist_ok=True)
 if not projection_copy.exists():shutil.copyfile(OLD/'cpu_inputs/road_surface_projection.py',projection_copy)
 sys.path.insert(0,str(projection_copy.parent));from road_surface_projection import project_surface_mesh
 # Re-certify previously unrequested nodes using the original measured plane and exact original gates.
 asset_module_path=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/road_asset.py');spec=importlib.util.spec_from_file_location('adaptive_fixed_road_asset',asset_module_path);assetmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(assetmodule)
 anchor_path=B/'road_asset_20261116_r1/asset_r8_guarded/measurement_anchors.npz';anchors=load(anchor_path);oldreceipt=json.loads((B/'road_asset_20261116_r1/asset_r8_guarded/receipt.json').read_text())['interpolation'];oldworld=world.copy()
 candidate_mask=~prior['surface_supported']&~road['active']&~prior['protected_geometry']&~prior['guarded_world_rows']&np.isfinite(world).all(1)
 world,newnorm,cert,cert_receipt=assetmodule.infer_measured_plane_surface(world,road['normal_world'],candidate_mask,anchors['world_xyz'],oldreceipt['plane_normal_world'],oldreceipt['plane_offset_world'],radius_m=1.5,maximum_edge_m=3.,maximum_plane_residual_m=.05)
 recovered=cert['surface_interpolated'];supported=prior['surface_supported']|recovered
 if not np.array_equal(world[road['active']],oldworld[road['active']]):raise ValueError('re-certification moved an old active node')
 atlas=dict(atlas);atlas['surface_world_xyz']=world[grid];np.savez_compressed(out/'surface_certificate.npz',recovered_world_rows=np.flatnonzero(recovered),target_world_xyz=world[recovered],target_normal_world=newnorm[recovered],anchor_ids=cert['interpolation_anchor_ids'][recovered],nearest_anchor_m=cert['interpolation_radius_m'][recovered],support_triangle_max_edge_m=cert['interpolation_triangle_max_edge_m'][recovered],anchor_plane_residual_m=cert['interpolation_height_residual_m'][recovered],inside_hull=cert['interpolation_inside_measurement_hull'][recovered]);write(out/'surface_certificate.json',dict(cert_receipt,source_anchors=str(anchor_path),source_anchors_sha256=sha(anchor_path),source_plane_receipt=str(B/'road_asset_20261116_r1/asset_r8_guarded/receipt.json'),original_active_geometry_exact=True,all_original_parent_files_unchanged=True))
 triuv=np.empty((len(world),2));gy,gx=np.indices(shape);triuv[grid.ravel()]=np.stack((gx,gy),-1).reshape(-1,2)
 def rebuild_map(raw):
  old=oldmaps[raw];m=project_surface_mesh(SimpleNamespace(grid_world_row=grid,surface_world_xyz=atlas['surface_world_xyz']),supported,old['camera_K'],old['world_to_camera'],old['canonical_mesh_hit_valid'].shape,road['minimum_model_depth']/float(road['meter_to_model']));v=m['valid'];ids=m['triangle_world_rows'][v];w=m['triangle_weights'][v];vertices=world[ids];U=triuv[ids];X=(vertices*w[...,None]).sum(1);center=np.full((*v.shape,2),-4.);center[v]=(U*w[...,None]).sum(1);T=old['world_to_camera'];R=T[:3,:3];K=old['camera_K'];O=-R.T@T[:3,3];y,x=np.nonzero(v);direction=np.c_[((np.stack((x,y),1)+.5)-K[:2,2])/np.diag(K)[:2],np.ones(len(X))]@R;n=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);den=(n*direction).sum(1);t=(n*(vertices[:,0]-O)).sum(1)/den;Jxy=np.empty((len(X),2,2))
  for axis in [0,1]:
   dr=R.T[:,axis]/K[axis,axis];dt=-t*(n@dr)/den;Jxy[:,:,axis]=(t[:,None]*dr+direction*dt[:,None])[:,:2]
  Ex=np.stack((vertices[:,1,:2]-vertices[:,0,:2],vertices[:,2,:2]-vertices[:,0,:2]),-1);Eu=np.stack((U[:,1]-U[:,0],U[:,2]-U[:,0]),-1);jac=np.zeros((*v.shape,2,2));jac[v]=Eu@np.linalg.inv(Ex)@Jxy;surface=np.zeros((*v.shape,3));surface[v]=X;d=dict(old);d.update(surface_world_xyz=surface,ground_world_xyz=surface,center_uv=center,pixel_uv_jacobian=jac,canonical_mesh_hit_valid=v,triangle_world_rows=m['triangle_world_rows'],triangle_weights=m['triangle_weights'],camera_depth=m['camera_depth_m']);return d
 maps={r:rebuild_map(r)for r in FRAMES}
 finite_nodes=np.isfinite(atlas['surface_world_xyz']).all(-1);finite_world=np.zeros(len(world),bool);finite_world[grid.ravel()]=finite_nodes.ravel()
 fixed_mesh=SimpleNamespace(grid_world_row=grid,surface_world_xyz=atlas['surface_world_xyz'])
 for raw in [86,110]:
  m=maps[raw];t=teachers[raw];n=native[raw];K=n['K']|t['K']|t['protected'];C=t['C']&~K;S=t['S']&~K;valid=m['canonical_mesh_hit_valid'];cs=C|S;seam=ndi.binary_dilation(cs,iterations=3)&valid&t['A']&~K;f=(cs|seam)&valid;fit[raw]=f;retained[raw]=K;training_world.append(m['surface_world_xyz'][f])
  fullmesh=project_surface_mesh(fixed_mesh,finite_world,m['camera_K'],m['world_to_camera'],valid.shape,road['minimum_model_depth']/float(road['meter_to_model']))
  fullvalid=fullmesh['valid'];missing=~valid;oldmissing=~oldmaps[raw]['canonical_mesh_hit_valid'];alpha=n['road_alpha'];p=hwc(n['road_rgb']);eff=np.divide(p,alpha[...,None],out=np.zeros_like(p),where=alpha[...,None]>1e-8)
  rows=[]
  for role,domain in [('C',C),('S',S)]:
   ref=hwc(t['target_'+role+'_srgb']);low=domain&(alpha<.95);no=domain&(alpha<.05);thin=domain&(alpha>=.05)&(alpha<.95);dark=domain&(alpha>=.95)&(eff.mean(-1)<.05)&(ref.mean(-1)>=.10)
   tr=m['triangle_world_rows'][valid&domain];material_invalid=np.any(~prior['effective_material_available'][tr],axis=1)if len(tr)else np.empty(0,bool)
   rows.append(dict(source=role,denominator=int(domain.sum()),no_support_n=int(no.sum()),thin_n=int(thin.sum()),high_alpha_dark_n=int(dark.sum()),old_canonical_missing_n=int((domain&oldmissing).sum()),certified_recovered_rayhit_n=int((domain&oldmissing&valid).sum()),canonical_missing_n=int((domain&missing).sum()),finite_mesh_hit_but_geometry_unqualified_n=int((domain&missing&fullvalid).sum()),no_finite_mesh_hit_n=int((domain&missing&~fullvalid).sum()),rayhit_low_alpha_repairable_pixels=int((low&valid).sum()),rayhit_material_invalid_vertex_pixels=int(material_invalid.sum()),static_occlusion_classification='unavailable: standalone road alpha is not scene alphaT'))
  counts.append(dict(raw=raw,domains=rows))
  d=dict(fit=f,outside=valid&~f&~K,retained=K,appearance_A=t['A']&valid&~K,appearance_C=C,appearance_S=S,C_source=C,S_source=S,canonical_valid=valid,old_canonical_valid=oldmaps[raw]['canonical_mesh_hit_valid'],certified_recovered_rayhit=oldmissing&valid,canonical_missing_C_S=cs&missing,finite_unqualified_hit_C_S=cs&missing&fullvalid,no_mesh_hit_C_S=cs&missing&~fullvalid,seam=seam,retained_confirmed_A=t['A']&valid&~f&~K)
  np.savez_compressed(out/f'domains/{raw:03d}.npz',**d)
  gap=f&(alpha<.95);ys,xs=np.nonzero(gap);xyz=m['surface_world_xyz'][gap];tri=m['triangle_world_rows'][gap];bary=m['triangle_weights'][gap];uv=m['center_uv'][gap];verts=world[tri];normal=np.cross(verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True);oldnormal=road['normal_world'][tri].mean(1);normal[np.sum(normal*oldnormal,axis=1)<0]*=-1
  # Pixel-to-world differential follows the actual fixed face, not a resized UV map.
  U=triuv[tri];E=np.stack((verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]),-1);F=np.stack((U[:,1]-U[:,0],U[:,2]-U[:,0]),-1);J=np.einsum('nij,njk->nik',E@np.linalg.inv(F),m['pixel_uv_jacobian'][gap]);sv=np.linalg.svd(J,compute_uv=False);sigma=np.clip(.70*sv[:,-1],.004,.018)
  role=np.where(C[gap],2,np.where(S[gap],4,8)).astype(np.uint8);yy=np.floor(uv[:,1]).astype(int).clip(0,shape[0]-2);xx=np.floor(uv[:,0]).astype(int).clip(0,shape[1]-2);du=uv[:,0]-xx;dv=uv[:,1]-yy;face=2*(yy*(shape[1]-1)+xx)+(du<dv)
  for i in range(len(xyz)):evidence.append(dict(raw=raw,pixel=(int(xs[i]),int(ys[i])),xyz=xyz[i],tri=tri[i],bary=bary[i],uv=uv[i],normal=normal[i],face=int(face[i]),sigma=float(sigma[i]),role=int(role[i]),deficit=float(1-alpha[ys[i],xs[i]])))
 # Diagnostic domains are the same frozen world neighborhood projected into their own cameras.
 worldtree=cKDTree(np.concatenate(training_world)[:,:2])
 for raw in [134,179,198]:
  m=maps[raw];v=m['canonical_mesh_hit_valid'];f=np.zeros(v.shape,bool);dist,_=worldtree.query(m['surface_world_xyz'][v,:2],workers=2);f[v]=dist<=.06;K=native[raw]['K'];fit[raw]=f&~K;retained[raw]=K;np.savez_compressed(out/f'domains/{raw:03d}.npz',fit=fit[raw],outside=v&~fit[raw]&~K,retained=K,appearance_A=np.zeros(v.shape,bool),appearance_C=np.zeros(v.shape,bool),appearance_S=np.zeros(v.shape,bool),canonical_valid=v)
 guard_trees={r:cKDTree(np.stack(np.nonzero(retained[r]),-1)[:,::-1])for r in FRAMES}
 def ksafe(X,cap):
  # Projection of a full4.5sigma world ball bounds every tangent yaw and the thin normal.
  R=4.5*float(cap)
  for raw in FRAMES:
   uv,cam=project(np.asarray(X)[None],maps[raw]);z=cam[0,2]
   if z+R<=.2:continue
   if z-R<=.2:return False
   K=maps[raw]['camera_K'];q=cam[0];radius=R/(z-R)*np.hypot(K[0,0]*(1+abs(q[0])/z),K[1,1]*(1+abs(q[1])/z))+4.5*np.sqrt(.3)+2
   if len(guard_trees[raw].data)and guard_trees[raw].query(uv[0])[0]<=radius:return False
  return True
 def visible(X):
  support=[]
  for raw in [86,110]:
   uv,cam=project(np.asarray(X)[None],maps[raw]);x,y=np.floor(uv[0]+.5).astype(int)
   if not(0<=x<1152 and 0<=y<768)or not fit[raw][y,x]:continue
   surf=maps[raw]['surface_world_xyz'][y,x];T=maps[raw]['world_to_camera'];surfz=(surf@T[:3,:3].T+T[:3,3])[2]
   if abs(surfz-cam[0,2])>.05:continue
   t=teachers[raw];role=2 if t['C'][y,x]else(4 if t['S'][y,x]else 8);ref=hwc(t['target_C_srgb']if role==2 else(t['target_S_srgb']if role==4 else t['target_A_srgb']))[y,x];support.append((raw,x,y,role,ref))
  return support
 sg=supported[grid];tri0=sg[:-1,:-1]&sg[:-1,1:]&sg[1:,1:];tri1=sg[:-1,:-1]&sg[1:,1:]&sg[1:,:-1];hasface=np.zeros_like(sg);hasface[:-1,:-1]|=tri0|tri1;hasface[:-1,1:]|=tri0;hasface[1:,1:]|=tri0|tri1;hasface[1:,:-1]|=tri1;faceworld=np.zeros(len(world),bool);faceworld[grid.ravel()]=hasface.ravel()
 eligible=~road['active']&supported&faceworld&~prior['guarded_world_rows']&~prior['protected_geometry']&(road['minimum_model_depth']<=0)&np.isfinite(world).all(1)&(np.linalg.norm(world-oldworld,axis=1)<=1e-7)&(~recovered|(np.linalg.norm(road['normal_world']-newnorm,axis=1)<=1e-6))
 inactive_rows=np.flatnonzero(eligible);inactive_tree=cKDTree(world[inactive_rows]);activation={};reserve=[];accepted_xyz=[];accepted_sigma=[];recent=[];spacing_tree=None;merged=0;rejectedK=0;rejectedvisible=0;cap_discard=0;node_grid_uv=np.empty((len(world),2));gy,gx=np.indices(shape);node_grid_uv[grid.ravel()]=np.stack((gx,gy),-1).reshape(-1,2)
 evidence.sort(key=lambda e:(-e['deficit'],e['raw'],e['pixel'][1],e['pixel'][0]))
 for e in evidence:
  X=e['xyz'];sigma=e['sigma'];spacing=1.35*sigma
  duplicate=False
  if spacing_tree is not None:
   distance,index=spacing_tree.query(X)
   if distance<max(spacing,1.35*accepted_sigma[index]):duplicate=True
  if not duplicate and recent:
   rr=np.asarray([x[0]for x in recent]);ds=np.linalg.norm(rr-X,axis=1);limits=np.maximum(spacing,np.asarray([x[1]*1.35 for x in recent]));duplicate=bool(np.any(ds<limits))
  if duplicate:merged+=1;continue
  distances,which=inactive_tree.query(X,k=min(8,len(inactive_rows)));selected=None
  for distance,pos in zip(np.atleast_1d(distances),np.atleast_1d(which)):
   if distance>max(.02,1.5*sigma):break
   wid=int(inactive_rows[int(pos)])
   if wid in activation: selected=('existing',wid);break
   initial=float(max(road['gaussian_sigma_m'][wid,:2]));cap=min(.03,initial*1.6)
   if not ksafe(world[wid],cap):continue
   support=visible(world[wid])
   if not support:continue
   if len(activation)>=ACT_CAP:cap_discard+=1;continue
   q=dict(e);q.update(xyz=world[wid].copy(),normal=road['normal_world'][wid].copy(),sigma=initial,cap=cap,support=support,existing=wid,uv=node_grid_uv[wid].copy());activation[wid]=q;selected=('existing',wid);break
  if selected is None:
   cap=min(.03,sigma*1.6)
   if not ksafe(X,cap):rejectedK+=1;continue
   support=visible(X)
   if not support:rejectedvisible+=1;continue
   if len(reserve)>=APPEND_CAP:cap_discard+=1;continue
   q=dict(e);q.update(cap=cap,support=support,existing=-1);reserve.append(q);selected=('append',len(reserve)-1)
  accepted_xyz.append(X);accepted_sigma.append(sigma);recent.append((X,sigma))
  if len(recent)>=256:spacing_tree=cKDTree(np.asarray(accepted_xyz));recent=[]
 def package(items,existing):
  count=len(items);X=np.asarray([x['xyz']for x in items],float).reshape(-1,3);normal,u,v=frames(np.asarray([x['normal']for x in items],float).reshape(-1,3));scales=np.array([[x['sigma'],x['sigma'],.003]for x in items]).reshape(-1,3);caps=np.array([[x['cap'],x['cap']]for x in items]).reshape(-1,2);bary=[];tri=[];face=[];roles=[];sourcepix=np.full((count,2,2),-1,np.int32);supportids=np.full((count,2),-1,np.int32);viewbits=np.zeros(count,np.uint8);rgbs=[];opacity=np.full(count,.15,np.float32)
  for i,x in enumerate(items):
   uv=x['uv'];ix=int(np.floor(uv[0]).clip(0,shape[1]-2));iy=int(np.floor(uv[1]).clip(0,shape[0]-2));du=uv[0]-ix;dv=uv[1]-iy;kind=int(du<dv);ids=[grid[iy,ix],grid[iy,ix+1],grid[iy+1,ix+1]]if kind==0 else[grid[iy,ix],grid[iy+1,ix+1],grid[iy+1,ix]];bw=[1-du,du-dv,dv]if kind==0 else[1-dv,du,dv-du]
   if existing:
    found=False;wid=x['existing']
    for cy in [int(uv[1])-1,int(uv[1])]:
     for cx in [int(uv[0])-1,int(uv[0])]:
      if not(0<=cx<shape[1]-1 and 0<=cy<shape[0]-1):continue
      for ck,ci in [(0,[grid[cy,cx],grid[cy,cx+1],grid[cy+1,cx+1]]),(1,[grid[cy,cx],grid[cy+1,cx+1],grid[cy+1,cx]])]:
       if wid in ci and supported[ci].all():ids=ci;bw=[float(z==wid)for z in ci];ix=cx;iy=cy;kind=ck;found=True;break
      if found:break
     if found:break
    if not found:raise ValueError('existing anchor lacks certified fixed mesh face')
   tri.append(ids);bary.append(bw);face.append(2*(iy*(shape[1]-1)+ix)+kind);role=0;values=[]
   for raw,px,py,rgbrole,value in x['support']:
    j=0 if raw==86 else 1;supportids[i,j]=raw;sourcepix[i,j]=[px,py];viewbits[i]|=1<<j;role|=rgbrole;values.append(value)
   roles.append(role);base=road['reference_srgb'][x['existing']]if existing and prior['effective_material_available'][x['existing']]else np.mean(values,axis=0);rgbs.append(base)
  tria=np.asarray(tri,np.int64).reshape(-1,3);data=dict(world_xyz=X,normal_world=normal,tangent_u_world=u,tangent_v_world=v,scales_m=scales,initial_sigma_m=scales,max_sigma_m=caps,teacher_rgb=np.asarray(rgbs,np.float32).reshape(-1,3),triangle_id=np.asarray(face,np.int64),triangle_world_rows=tria,triangle_vertex_world=world[tria],barycentric=np.asarray(bary,float).reshape(-1,3),source_roles=np.asarray(roles,np.uint8),support_frame_ids=supportids,support_view_bits=viewbits,source_pixels=sourcepix,source_deficit=np.asarray([x['deficit']for x in items]),opacity_initial=opacity,existing_world_rows=np.asarray([x['existing']for x in items],np.int64),new_geometry=np.full(count,not existing,bool),geometry_inferred_from_fixed_measurements=np.any(recovered[tria],axis=1))
  if count:
   reconstruct=(world[data['triangle_world_rows']]*data['barycentric'][...,None]).sum(1);err=np.max(np.linalg.norm(reconstruct-X,axis=1))
   if err>1e-6:raise ValueError('proposal not on fixed canonical triangle')
  return data
 act=package(list(activation.values()),True);new=package(reserve,False);np.savez_compressed(out/'activation.npz',**act);np.savez_compressed(out/'append_reserve.npz',**new)
 # Every append reserve is genuine off-existing-grid geometry; duplicates are disallowed.
 all_tree=cKDTree(oldworld);nearest=all_tree.query(new['world_xyz'])[0]if len(new['world_xyz'])else np.array([])
 if np.any(nearest<1e-7):raise ValueError('reserve accidentally reuses old grid node; not genuine append')
 write(out/'gap_diagnosis.json',dict(schema='adaptive_surface_gap_diagnosis.v1',rows=counts,denominator='complete legal C/S source; no-hit retained',canonical_rule='old map only geometry surface_supported triangles, no material or write mask involved',finite_unqualified_policy='diagnostic only; finite xyz without independent surface certificate never appends geometry',teacher_used_for_candidate_selection=False))
 contract=dict(schema='adaptive_surface_sampling_contract.v1',frames=FRAMES,fit_frames=[86,110],diagnostic_frames=[134,179,198],resolution=[1152,768],gap_rule='actual standalone road alpha<.95 on fullC/S plus3pixel confirmed-road seam; RGB not used',surface_rule='fixed original canonical mesh, three supported vertices, exact barycentric hit; no plane fallback',activation_rule='qualified inactive existing row within max(.02m,1.5*targetsigma), nearest eligible first, 8 neighbors; retain fixed XYZ',sigma_rule='new equal tangent sigma=clip(.70*minimum singular value of actual pixel-to-world face Jacobian,.004,.018)m; max=1.6initial cap.03m; normal.003m',spacing_rule='greedy gap-deficit-first, world3D separation1.35max(local sigmas), merged across86/110, no RGB-based selection',point_caps=dict(activation=ACT_CAP,append=APPEND_CAP),K_rule='all5frames protected/K nearest-pixel sphere bound at4.5maxsigma +sqrt(.3)covariance blur*4.5 +2pixel rounding; actual native exact0 gate still required',guard_scope='five stated views only; inherited old guarded/protected flags for existing rows; no all199-frame/anyview claim',opacity_initial=.15,schedule=dict(activation_coverage_updates=200,joint_coverage_updates=400,appearance_updates=1200),history_892='765+95+32 historically activated grid rows; never count as appended geometry')
 contract.update(surface_rule='fixed supported canonical mesh plus independently re-certified original measured plane, exact original anchor hull/1.5m radius/3m edge/.05m residual gates',source_roles_definition={'1':'realA','2':'holeC','4':'shadowS','8':'confirmed-road3px seam uses realA'},recovered_anchor_reuse='existing XYZ <=1e-7m and normal vector <=1e-6 from certified original plane; otherwise genuine append only',uncertified_C_semantics='full2D editing domain retained, not all confirmed road and not all repairable by densification')
 write(out/'sampling_contract.json',contract)
 manifest=dict(schema='adaptive_surface_proposal.v1',complete=True,parent=str(N),parent_total=len(road['active']),parent_active=int(road['active'].sum()),activate_existing_world_rows=len(act['world_xyz']),append_new_geometry_reserve=len(new['world_xyz']),activation_npz=str(out/'activation.npz'),append_reserve_npz=str(out/'append_reserve.npz'),domains_root=str(out/'domains'),candidate_pixels=len(evidence),multi_view_spatial_merged=merged,rejected_K=rejectedK,rejected_visibility=rejectedvisible,cap_truncated=cap_discard,surface_diagnosis=str(out/'gap_diagnosis.json'),contract=str(out/'sampling_contract.json'),source_sha256={str(f):sha(f)for f in [N/'road.npz',N/'provenance.npz',BASE,OLD/'teacher/manifest.json',out/'activation.npz',out/'append_reserve.npz']},new_geometry_min_existing_node_distance_m=float(nearest.min())if len(nearest)else None,CPU_only=True,CUDA_calls=0,seconds=time.monotonic()-started)
 manifest.update(recovered_geometry_nodes=int(recovered.sum()),recovered_existing_activation_rows=int(recovered[act['existing_world_rows']].sum()),surface_certificate=str(out/'surface_certificate.npz'),surface_certificate_sha256=sha(out/'surface_certificate.npz'),normal_qualification_fix_prior_revision=str(out/'revision0'),old_proposal_engineering_only=True)
 write(out/'manifest.json.tmp',manifest);(out/'manifest.json.tmp').replace(out/'manifest.json');print(json.dumps(manifest),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  write(P/'proposal/failed.json',dict(error=str(e),traceback=traceback.format_exc(),CPU_only=True));raise
