"""Audit-only residual rays: keep every certified residual, no Poisson thinning or K pre-prune."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from types import SimpleNamespace
import json,time,hashlib,sys,shutil,traceback
import numpy as np
from scipy.spatial import cKDTree
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');FIRST=B/'road_surface_adaptive_20261121_r1';OLD=B/'road_common1152_20261121';P=B/'road_surface_residual_20261121_r2';N=FIRST/'appearance_fit/asset';BASE=B/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz';VIEWS=[86,110,134,179,198];TOL=.001
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def tangent_frames(normal):
 n=normal/np.linalg.norm(normal,axis=1,keepdims=True);u=np.zeros_like(n);v=np.zeros_like(n)
 for i,x in enumerate(n):
  a=np.cross([0.,0.,1.],x);K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]]);R=np.eye(3)+K+K@K/(1+x[2])if x[2]>-1+1e-8 else np.diag([1.,-1.,-1.]);u[i]=R[:,0];v[i]=R[:,1]
 return n,u,v
def main():
 start=time.monotonic();out=P/'proposal';out.mkdir(parents=True,exist_ok=True);road=load(N/'road.npz');prior=load(N/'provenance.npz');base=load(BASE);cert=load(FIRST/'proposal/surface_certificate.npz');grid=base['grid_world_row'];original_count=grid.size;world=prior['world_xyz'];mesh_world=np.zeros((original_count,3));mesh_world[grid.ravel()]=base['surface_world_xyz'].reshape(-1,3);mesh_world[cert['recovered_world_rows']]=cert['target_world_xyz'];surface=mesh_world[grid]
 oldgeo=load(OLD/'formal/r3/G_NUMERIC/asset/provenance.npz');supported=oldgeo['surface_supported'].copy();supported[cert['recovered_world_rows']]=True
 copy=P/'cpu/road_surface_projection.py';copy.parent.mkdir(exist_ok=True)
 if not copy.exists():shutil.copyfile(FIRST/'cpu/road_surface_projection.py',copy)
 sys.path.insert(0,str(copy.parent));from road_surface_projection import project_surface_mesh
 headers={r:load(OLD/f'maps/{r:03d}.npz')for r in VIEWS};parents={r:load(FIRST/f'appearance_fit/rendered/{r:03d}.npz')for r in VIEWS};evidence=[];domainrows=[]
 for raw in [86,110]:
  h=headers[raw];d=load(FIRST/f'proposal/domains/{raw:03d}.npz');alpha=parents[raw]['road_alpha'];m=project_surface_mesh(SimpleNamespace(grid_world_row=grid,surface_world_xyz=surface),supported,h['camera_K'],h['world_to_camera'],alpha.shape,road['minimum_model_depth'][:original_count]/float(road['meter_to_model']))
  for source,bit in [('C',2),('S',4)]:
   certified=d['appearance_'+source]&d['canonical_valid'];mask=certified&(alpha<.95)
   if np.any(mask&~m['valid']):raise ValueError('fixed certified residual ray lost mesh certificate')
   y,x=np.nonzero(mask);ids=m['triangle_world_rows'][mask];w=m['triangle_weights'][mask];verts=mesh_world[ids];X=(verts*w[...,None]).sum(1);normal=np.cross(verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True);ref=road['normal_world'][ids].mean(1);normal[np.sum(normal*ref,axis=1)<0]*=-1
   T=h['world_to_camera'];R=T[:3,:3];K=h['camera_K'];origin=-R.T@T[:3,3];direction=np.c_[((np.stack((x,y),1)+.5)-K[:2,2])/np.diag(K)[:2],np.ones(len(X))]@R;den=(normal*direction).sum(1);t=(normal*(verts[:,0]-origin)).sum(1)/den;J=np.empty((len(X),3,2))
   for axis in [0,1]:
    dr=R.T[:,axis]/K[axis,axis];dt=-t*(normal@dr)/den;J[:,:,axis]=t[:,None]*dr+direction*dt[:,None]
   sigma=np.clip(.70*np.linalg.svd(J,compute_uv=False)[:,-1],.004,.018);domainrows.append(dict(raw=raw,source=source,certified_domain_n=int(certified.sum()),fixed_residual_n=len(X),unconfirmed_editing_excluded_n=int((d['appearance_'+source]&~d['canonical_valid']).sum())))
   for i in range(len(X)):evidence.append(dict(raw=raw,source=source,bit=bit,pixel=(int(x[i]),int(y[i])),alpha=float(alpha[y[i],x[i]]),xyz=X[i],normal=normal[i],sigma=float(sigma[i]),triangle=ids[i],bary=w[i],face=int(m['triangle_face_id'][y[i],x[i]])))
 # Merge only a one-to-one cross-view match within1mm on the same connected fixed face neighborhood.
 clusters=[];crossmerge=0;indices86=[]
 for e in evidence:
  target=None
  if e['raw']==110 and indices86:
   if 'tree86'not in locals():tree86=cKDTree(np.asarray([clusters[i]['xyz']for i in indices86]))
   for j in tree86.query_ball_point(e['xyz'],TOL):
    i=indices86[j];c=clusters[i]
    if any(q['raw']==110 for q in c['sources']):continue
    same_surface=np.intersect1d(c['triangle'],e['triangle']).size>0 and np.dot(c['normal'],e['normal'])>=.999
    if same_surface:target=i;break
  if target is None:
   c=dict(e);c['sources']=[e];clusters.append(c)
   if e['raw']==86:indices86.append(len(clusters)-1)
  else:clusters[target]['sources'].append(e);crossmerge+=1
 # Existing anchors are reused only at the residual's actual location (<=1mm), never a large-footprint association.
 recovered=np.zeros(len(world),bool);recovered[cert['recovered_world_rows']]=True;mesh_match=np.ones(len(world),bool);mesh_match[:original_count]=np.linalg.norm(world[:original_count]-mesh_world,axis=1)<=1e-7;normal_match=np.ones(len(world),bool);normal_match[cert['recovered_world_rows']]=np.linalg.norm(road['normal_world'][cert['recovered_world_rows']]-cert['target_normal_world'],axis=1)<=1e-6;geo=np.zeros(len(world),bool);geo[:original_count]=supported
 eligible=~road['active']&geo&mesh_match&normal_match&~prior['guarded_world_rows']&~prior['protected_geometry']&(road['minimum_model_depth']<=0);inactive=np.flatnonzero(eligible);itree=cKDTree(world[inactive]);alltree=cKDTree(world);used=set();near_active=0;exact_active=0;records=[];gy,gx=np.indices(grid.shape);uvworld=np.zeros((original_count,2));uvworld[grid.ravel()]=np.stack((gx,gy),-1).reshape(-1,2)
 def anchor_face(wid):
  uv=uvworld[wid]
  for cy in [int(uv[1])-1,int(uv[1])]:
   for cx in [int(uv[0])-1,int(uv[0])]:
    if not(0<=cx<grid.shape[1]-1 and 0<=cy<grid.shape[0]-1):continue
    for kind,tri in [(0,[grid[cy,cx],grid[cy,cx+1],grid[cy+1,cx+1]]),(1,[grid[cy,cx],grid[cy+1,cx+1],grid[cy+1,cx]])]:
     if wid in tri and supported[tri].all():return np.asarray(tri),np.asarray([float(k==wid)for k in tri]),2*(cy*(grid.shape[1]-1)+cx)+kind
  return None
 for c in clusters:
  X=c['xyz'];dist0,oldid=alltree.query(X);isactive=bool(road['active'][oldid]);near_active+=int(isactive and dist0<=TOL);exact_active+=int(isactive and dist0<=1e-7);wid=-1;af=None
  distance,which=itree.query(X,k=min(8,len(inactive)))
  for dd,j in zip(np.atleast_1d(distance),np.atleast_1d(which)):
   if dd>TOL:break
   row=int(inactive[int(j)])
   if row in used or any(np.linalg.norm(world[row]-e['xyz'])>TOL for e in c['sources']):continue
   af=anchor_face(row)
   if af is not None:wid=row;used.add(row);break
  q=dict(c);q.update(existing=wid,nearest_parent_world_row=int(oldid),nearest_parent_distance=float(dist0),nearest_parent_active=isactive)
  if wid>=0:q.update(xyz=world[wid].copy(),normal=road['normal_world'][wid].copy(),sigma=float(np.max(road['gaussian_sigma_m'][wid,:2])),triangle=af[0],bary=af[1],face=af[2])
  q['cap']=min(.03,1.6*q['sigma']);records.append(q)
 n=len(records);X=np.asarray([q['xyz']for q in records]);normal,u,v=tangent_frames(np.asarray([q['normal']for q in records]));caps=np.asarray([[q['cap'],q['cap']]for q in records]);scales=np.asarray([[q['sigma'],q['sigma'],.003]for q in records]);supportids=np.full((n,2),-1,np.int32);pixels=np.full((n,2,2),-1,np.int32);originxyz=np.zeros((n,2,3));sourcealpha=np.full((n,2),-1.,np.float32);rolesper=np.zeros((n,2),np.uint8);sourcebits=np.zeros(n,np.uint8)
 for i,q in enumerate(records):
  for e in q['sources']:
   j=0 if e['raw']==86 else 1;supportids[i,j]=e['raw'];pixels[i,j]=e['pixel'];originxyz[i,j]=e['xyz'];sourcealpha[i,j]=e['alpha'];rolesper[i,j]=e['bit'];sourcebits[i]|=e['bit']
 projection_conflict=np.zeros((n,5),bool);near_plane=np.zeros((n,5),bool);distK=np.full((n,5),1e10);radius=np.zeros((n,5));projection=np.zeros((n,5,2))
 for j,raw in enumerate(VIEWS):
  h=headers[raw];T=h['world_to_camera'];K=h['camera_K'];cam=X@T[:3,:3].T+T[:3,3];z=cam[:,2];uv=np.divide(cam[:,:2],z[:,None],out=np.full((n,2),1e10),where=z[:,None]>0)*np.diag(K)[:2]+K[:2,2]-.5;projection[:,j]=uv;Rball=4.5*np.max(caps,axis=1);visible=z+Rball>.2;near_plane[:,j]=visible&(z-Rball<=.2);ok=visible&~near_plane[:,j];r=np.zeros(n);r[ok]=Rball[ok]/(z[ok]-Rball[ok])*np.hypot(K[0,0]*(1+abs(cam[ok,0])/z[ok]),K[1,1]*(1+abs(cam[ok,1])/z[ok]))+4.5*np.sqrt(.3)+2;radius[:,j]=r;ky,kx=np.nonzero(parents[raw]['K']);tree=cKDTree(np.stack((kx,ky),-1));distK[:,j]=tree.query(uv,workers=2)[0];projection_conflict[:,j]=near_plane[:,j]|(ok&(distK[:,j]<=r))
 tris=np.asarray([q['triangle']for q in records],np.int64);bary=np.asarray([q['bary']for q in records]);kind=np.asarray([int(q['existing']<0)for q in records],np.uint8);rgbtree=cKDTree(world[road['active']]);donordistance,donorlocal=rgbtree.query(X);donor=np.flatnonzero(road['active'])[donorlocal]
 data=dict(candidate_id=np.arange(n,dtype=np.int64),candidate_kind=kind,existing_world_rows=np.asarray([q['existing']for q in records],np.int64),world_xyz=X,normal_world=normal,tangent_u_world=u,tangent_v_world=v,scales_m=scales,initial_sigma_m=scales,max_sigma_m=caps,opacity_initial=np.full(n,.15,np.float32),triangle_id=np.asarray([q['face']for q in records],np.int64),triangle_world_rows=tris,triangle_vertex_world=mesh_world[tris],barycentric=bary,geometry_inferred_from_fixed_measurements=np.any(recovered[tris],axis=1),support_frame_ids=supportids,source_pixels=pixels,source_roles=sourcebits,source_roles_per_view=rolesper,origin_ray_world_xyz=originxyz,parent_source_alpha=sourcealpha,projection_K_conflict=projection_conflict,projection_near_plane=near_plane,projection_K_distance_px=distK,projection_K_radius_px=radius,projection_center_px=projection,nearest_parent_world_row=np.asarray([q['nearest_parent_world_row']for q in records],np.int64),nearest_parent_distance_m=np.asarray([q['nearest_parent_distance']for q in records]),nearest_parent_active=np.asarray([q['nearest_parent_active']for q in records],bool),new_geometry=kind.astype(bool),initial_rgb=road['reference_srgb'][donor],initial_color_donor_world_row=donor,initial_color_donor_distance_m=donordistance)
 error=float(np.max(np.abs(np.einsum('ni,nij->nj',bary,data['triangle_vertex_world'])-X)))
 if error>1e-6:raise ValueError('residual proposal left the certified fixed surface')
 np.savez_compressed(out/'candidates.npz',**data)
 for name,flag in [('activation',kind==0),('append_reserve',kind==1)]:np.savez_compressed(out/(name+'.npz'),**{k:v[flag]for k,v in data.items()})
 inherited=json.loads((FIRST/'proposal/manifest.json').read_text());projection_any=projection_conflict.any(1);summary=dict(schema='surface_residual_candidate_audit.v1',residual_pixels=len(evidence),merged_cross_view_same_position_1mm=crossmerge,candidates=n,activate_existing_n=int((kind==0).sum()),append_new_geometry_n=int((kind==1).sum()),near_existing_active_center_within1mm_n=near_active,exact_existing_active_center_within1e7m_n=exact_active,projection_K_conflict_n=int(projection_any.sum()),projection_K_clear_n=int((~projection_any).sum()),projection_K_conflict_by_view={str(raw):int(projection_conflict[:,j].sum())for j,raw in enumerate(VIEWS)},domains=domainrows,no_CPU_K_discard=True,no_Poisson_or_footprint_thinning=True,maximum_total_candidates=len(evidence),surface_reconstruction_max_m=error,firstwave_history=dict(rejected_K_aggregate=inherited['rejected_K'],multi_view_spatial_merged=inherited['multi_view_spatial_merged'],cap_truncated=inherited['cap_truncated'],actual_append=json.loads((FIRST/'proposal/append_manifest.json').read_text())['appended_new_geometry']))
 write(out/'audit.json',summary);contract=dict(schema='surface_residual_proposal_contract.v1',parent=str(N),resolution=[1152,768],fit_frames=[86,110],diagnostic_frames=[134,179,198],residual='firstwave final alpha<.95 AND unchanged certified road C/S mask, no uncertified editing pixels',surface='unchanged canonical mesh plus the same348 independently certified frozen-measurement-plane nodes; no new plane fitting or wider gates',merging='one-to-one cross-view only, world distance<=.001m, connected same face neighborhood and normal dot>=.999; no within-view thinning',inactive_reuse='eligible inactive original anchor only if world distance<=.001m and unchanged XYZ/normal already match certified surface; each row once',new_sigma='same original .70*min singular(pixel-world Jacobian), clip[.004,.018]m, equal tangent; cap=min(.03m,1.6initial); normal.003m',K_projection='same conservative4.5sigma sphere, native.3 covariance blur and2px rounding; flags only, never prune',required_GPU_gate='actual joint new contribution on full5frame K exactly0, including frozen original occlusion; projection flags are not a safety verdict',training_started=False,source_roles={'2':'hole C','4':'shadow S'},candidate_kind={'0':'activate unchanged existing inactive','1':'append explicit new Gaussian row'},budget_cap='no more candidates than fixed residual pixels after cross-view duplicate merging')
 write(out/'contract.json',contract);manifest=dict(schema='surface_residual_candidates.v1',complete=True,audit_only=True,training_authorized_by_this_file=False,parent=str(N),parent_total=len(road['active']),parent_active=int(road['active'].sum()),candidates=n,activate_existing_world_rows=int((kind==0).sum()),append_new_geometry_reserve=int((kind==1).sum()),candidates_npz=str(out/'candidates.npz'),activation_npz=str(out/'activation.npz'),append_reserve_npz=str(out/'append_reserve.npz'),audit=str(out/'audit.json'),contract=str(out/'contract.json'),source_sha256={str(p):sha(p)for p in [N/'road.npz',N/'provenance.npz',BASE,FIRST/'proposal/surface_certificate.npz',FIRST/'proposal/surface_certificate.json',FIRST/'proposal/manifest.json',FIRST/'appearance_fit/rendered/086.npz',FIRST/'appearance_fit/rendered/110.npz',out/'candidates.npz',out/'activation.npz',out/'append_reserve.npz',Path(__file__)]},CPU_only=True,CUDA_calls=0,seconds=time.monotonic()-start)
 write(out/'manifest.json.tmp',manifest);(out/'manifest.json.tmp').replace(out/'manifest.json');print(json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write(P/'proposal/failed.json',dict(error=str(e),traceback=traceback.format_exc(),CPU_only=True));raise
