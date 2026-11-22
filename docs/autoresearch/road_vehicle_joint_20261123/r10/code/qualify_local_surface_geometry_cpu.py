"""Read-only local nonplanar LiDAR surface qualification; no old geometry edits."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree,Delaunay

TRAIN=[60,70,80,86,95,110,125,145,165]
HELDOUT=[134,179,198]
UV_BOUNDS=[245,62,579,213]

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):
    x=np.asarray(x);return {'count':int(x.size),'mean':float(x.mean()),'median':float(np.median(x)),
      'p95':float(np.quantile(x,.95)),'max':float(x.max())} if x.size else {'count':0}
def design(xy,centre):
    d=xy-centre;x,y=d[:,0],d[:,1];return np.stack([np.ones(len(x)),x,y,x*x,x*y,y*y],1)
def predict(xy,centre,coeff):return design(xy,centre)@coeff
def normal(xy,centre,b):
    d=xy-centre;dx=b[1]+2*b[3]*d[:,0]+b[4]*d[:,1];dy=b[2]+b[4]*d[:,0]+2*b[5]*d[:,1]
    n=np.c_[-dx,-dy,np.ones(len(dx))];return n/np.linalg.norm(n,axis=1,keepdims=True)
def brown(world,T,intr):
    camera=world@T[:3,:3].T+T[:3,3];z=camera[:,2];xy=camera[:,:2]/np.maximum(z[:,None],1e-8)
    x,y=xy[:,0],xy[:,1];fx,fy,cx,cy,k1,k2,p1,p2,k3=intr;r=x*x+y*y;rad=1+k1*r+k2*r*r+k3*r*r*r
    return np.stack([fx*(x*rad+2*p1*x*y+p2*(r+2*x*x))+cx-.5,
      fy*(y*rad+p1*(r+2*y*y)+2*p2*x*y)+cy-.5],1),z

def measured_normals(points,centre,coeff):
    if len(points)<10:return np.empty(0),np.zeros(len(points),bool),np.zeros_like(points)
    distance,ids=cKDTree(points[:,:2]).query(points[:,:2],k=min(20,len(points)))
    good=np.zeros(len(points),bool);normals=np.zeros_like(points);angle=np.full(len(points),np.nan)
    expected=normal(points[:,:2],centre,coeff)
    for i in range(len(points)):
        choose=ids[i][distance[i]<=.5]
        if len(choose)<10:continue
        q=points[choose];cov=np.cov(q-q.mean(0),rowvar=False);eig,vec=np.linalg.eigh(cov)
        if eig[0]/max(eig.sum(),1e-12)>.05:continue
        n=vec[:,0];n=n if n[2]>=0 else -n;normals[i]=n;good[i]=True
        angle[i]=np.degrees(np.arccos(np.clip(n@expected[i],-1,1)))
    return angle[good],good,normals

def collect(raw,C,xy_bounds,zref,bank):
    obs_path=bank/'observations'/f'{raw:03d}.json';obs=read(obs_path);lidar=bank/obs['lidar_path']
    original=np.fromfile(lidar,dtype='<f4').reshape(-1,6);ids=np.arange(len(original),dtype=np.int32)
    Tego=np.asarray(obs['ego_world'],np.float64).reshape(4,4);world=original[:,:3].astype(np.float64)@Tego[:3,:3].T+Tego[:3,3]
    region=np.isfinite(world).all(-1)&(world[:,:2]>=xy_bounds[0]).all(-1)&(world[:,:2]<=xy_bounds[1]).all(-1)&(abs(world[:,2]-zref)<=1.5)
    world=world[region];ids=ids[region];record={'raw':raw,'input_rows':len(original),'local_XY_broad_Z_rows':len(world)}
    boxed=np.zeros(len(world),bool)
    for box in obs['boxes']:
        if 'world_from_actor' not in box:continue
        pose=np.asarray(box['world_from_actor'],np.float64).reshape(4,4);half=.5*np.asarray(box['dimensions'],np.float64)+[.15,.15,.08]
        boxed|=(abs((world-pose[:3,3])@pose[:3,:3])<=half).all(-1)
    record['inside_OBB_rejected']=int(boxed.sum());world=world[~boxed];ids=ids[~boxed]
    record['OBB_filtered_LiDAR_points']=len(world)
    cal_path=C/f'road_multisource_20261122_r1/sources/qualified_observations/{raw:03d}.npz'
    with np.load(cal_path,allow_pickle=False) as cal:T=cal['world_to_camera']
    sem_path=bank.parent/'semantics'/f'{raw:03d}.png';labels=np.asarray(Image.open(sem_path));road=binary_erosion(labels==0,iterations=2)
    uv,depth=brown(world,T,np.asarray(obs['camera_intrinsic_Brown'],np.float64));finite=np.isfinite(uv).all(-1)
    ix=np.rint(np.nan_to_num(uv[:,0])).astype(int);iy=np.rint(np.nan_to_num(uv[:,1])).astype(int)
    bounded=finite&(depth>.1)&(ix>=0)&(iy>=0)&(ix<road.shape[1])&(iy<road.shape[0]);keep=np.zeros(len(world),bool)
    q=np.flatnonzero(bounded);keep[q]=road[iy[q],ix[q]]
    point_labels=np.full(len(world),255,np.uint8);point_labels[q]=labels[iy[q],ix[q]]
    record['photo_road0_before_erosion']=int((point_labels==0).sum())
    record['photo_nonroad_bounded']=int((bounded&(point_labels!=0)).sum())
    record['photo_query_outside']=int((~bounded).sum())
    record['road_semantic_retained']=int(keep.sum());record['semantic_is_original_photo_proxy']=True
    return world[keep],ids[keep],record,[pin(obs_path),pin(lidar),pin(cal_path),pin(sem_path)],{'world':world,'lidar_ids':ids,'photo_labels':point_labels,'strict_photo_road':keep,'photo_bounded':bounded}

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();C=a.campaign_root
    out=a.output;out.mkdir(parents=True,exist_ok=True);start=time.time()
    grid_path=C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz';g=load(grid_path)
    M=g['world_from_uv_xy'][:2,:2];origin=g['world_from_uv_xy'][:2,3];u0,v0,u1,v1=UV_BOUNDS
    yy,xx=np.mgrid[v0:v1,u0:u1];uv=np.stack([xx,yy],-1);xy=uv@M.T+origin;flat=xy.reshape(-1,2)
    xy_bounds=np.stack([flat.min(0)-.4,flat.max(0)+.4]);centre=.5*(xy_bounds[0]+xy_bounds[1])
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    zref=float(np.median([np.asarray(read(bank/f'observations/{r:03d}.json')['ego_world']).reshape(4,4)[2,3] for r in TRAIN]))
    contract={'schema':'R10.local_nonplanar_geometry_qualification_contract.v1','written_before_local_point_fit':True,
      'train_raw':TRAIN,'heldout_raw':HELDOUT,'bounds_uvxy_half_open':UV_BOUNDS,'nominal_spacing_m':.02,
      'XY_fit_context_bbox':xy_bounds.tolist(),'broad_Z_window':'median training ego origin Z +/-1.5m, before fit; heldout residual is not trimmed',
      'strict_point_filter':'finite rawXYZ->ego_world; original3DOBB margin[.15,.15,.08]; original real-photo road0 erode2 proxy',
      'surface':'local quadratic height IRLS8 Huber .03m; one measured medoid per0.1m XY cell; no whole-scene constant-Z plane',
      'heldout_height_median_max_m':.04,'heldout_height_p95_max_m':.10,'heldout_minimum_points_per_view':50,
      'normal_PCA':'same heldout-view nearest20 within .5m, >=10, eigen fraction<=.05; no model-residual-based normal selection',
      'heldout_normal_median_max_deg':10.,'heldout_normal_p95_max_deg':20.,'normal_minimum_valid_per_view':20,
      'normal_minimum_valid_total':100,'witness_triangle_max_edge_m':1.6,'nearest_measured_cell_max_m':.8,
      'distinct_source_frames_within0p8m_min':3,'direction_quadrants_within1m_min':3,
      'witness_triangle_fit_height_delta_max_m':.05,'witness_triangle_normal_delta_max_deg':15.,
      'geometry_only_nonphoto_road_candidates':'separate diagnostic counts, never silently added to strict fit/PASS',
      'source_grid':pin(grid_path),'original_grid_edited':False,'GPU_forward':0}
    contract_path=out/'geometry_qualification_contract.json';assert not contract_path.exists();write(contract_path,contract)
    points=[];raws=[];rowids=[];records=[];inputs=[];uncut={}
    for raw in TRAIN+HELDOUT:
        X,ids,record,pins,others=collect(raw,C,xy_bounds,zref,bank)
        points.append(X);raws.append(np.full(len(X),raw,np.int16));rowids.append(ids);records.append(record);inputs.extend(pins);uncut[raw]=others
        print(json.dumps(record),flush=True)
    X=np.concatenate(points);raw=np.concatenate(raws);lidar_ids=np.concatenate(rowids);train=np.isin(raw,TRAIN)
    np.savez_compressed(out/'strict_measurements.npz',world_xyz=X,source_raw=raw,source_lidar_row_id=lidar_ids,used_for_fit=train)
    write(out/'input_readback_SHA.json',{'complete':True,'actual_readback_inputs':inputs,'no_OS_open_tracing_claim':True,
      'LiDAR_column6_not_used_as_semantic_class':True,'point_semantic_class_input_available':False})
    if train.sum()<100:
        write(out/'certificate.json',{'complete':True,'decision':'INSUFFICIENT','reason':'fewer than100 strict training points',
          'geometry_npz':None,'frame_counts':records,'constant_z_plane_used':False,'contract':pin(contract_path)});return
    ti=np.flatnonzero(train);cell=np.floor((X[ti,:2]-xy_bounds[0])/.1).astype(int)
    order=np.lexsort((cell[:,1],cell[:,0]));sorted_ids=ti[order];sorted_cell=cell[order]
    _,starts=np.unique(sorted_cell,axis=0,return_index=True);starts=np.sort(starts);selected=[]
    for lo,hi in zip(starts,np.r_[starts[1:],len(sorted_ids)]):
        ids=sorted_ids[lo:hi];mid=np.median(X[ids],axis=0);selected.append(ids[np.argmin(np.linalg.norm(X[ids]-mid,axis=1))])
    selected=np.asarray(selected,np.int64);P=X[selected];A=design(P[:,:2],centre);coeff=np.linalg.lstsq(A,P[:,2],rcond=None)[0]
    for _ in range(8):
        residual=A@coeff-P[:,2];weight=np.minimum(1,.03/np.maximum(abs(residual),1e-12))
        coeff=np.linalg.lstsq(A*np.sqrt(weight[:,None]),P[:,2]*np.sqrt(weight),rcond=None)[0]
    evaluations=[];normal_payload={};heldout_pass=True;norm_count=0
    for r in HELDOUT:
        hit=raw==r;error=abs(predict(X[hit,:2],centre,coeff)-X[hit,2]);angles,valid,normals=measured_normals(X[hit],centre,coeff)
        heights=stats(error);ns=stats(angles);norm_count+=len(angles)
        passed=len(error)>=50 and heights['median']<=.04 and heights['p95']<=.10 and len(angles)>=20 and ns['median']<=10 and ns['p95']<=20
        heldout_pass&=passed;evaluations.append({'raw':r,'height_abs_error_m_all_strict_points':heights,'normal_angle_deg':ns,
          'normal_unmeasurable_points':int(hit.sum())-len(angles),'thresholds_pass':passed,'heldout_points_trimmed_by_prediction':False})
        normal_payload[f'{r}_world_xyz']=X[hit];normal_payload[f'{r}_normal_measured']=normals;normal_payload[f'{r}_normal_valid']=valid
    heldout_pass&=norm_count>=100;np.savez_compressed(out/'heldout_normal_measurements.npz',**normal_payload)
    geometry_only=[]
    for r,record in zip(TRAIN+HELDOUT,records):
        q=uncut[r];values=q['world'];error=abs(predict(values[:,:2],centre,coeff)-values[:,2]);angles,normal_ok,normals=measured_normals(values,centre,coeff)
        angle_map=np.full(len(values),np.inf);angle_map[normal_ok]=angles
        compatible=(error<=.10)&normal_ok&(angle_map<=20)
        omitted=q['photo_bounded']&(q['photo_labels']!=0)&compatible
        record['photo_nonroad_but_height_normal_compatible_diagnostic']=int(omitted.sum())
        record['photo_road0_eroded_but_height_normal_compatible_diagnostic']=int(((q['photo_labels']==0)&~q['strict_photo_road']&compatible).sum())
        geometry_only.append({'raw':r,'world_xyz':values,'source_lidar_ids':q['lidar_ids'],'photo_labels':q['photo_labels'],
          'photo_bounded':q['photo_bounded'],'strict_photo_road':q['strict_photo_road'],'height_normal_compatible':compatible})
    np.savez_compressed(out/'geometry_only_diagnostic_points.npz',**{f'{x["raw"]}_{k}':v for x in geometry_only for k,v in x.items() if k!='raw'})
    tree=cKDTree(P[:,:2]);nearest,_=tree.query(flat);triangulation=Delaunay(P[:,:2]);simplex=triangulation.find_simplex(flat);inside=simplex>=0
    tri_indices=triangulation.simplices[np.maximum(simplex,0)];V=P[tri_indices];bary=np.zeros((len(flat),3))
    T=triangulation.transform[np.maximum(simplex,0)];bary[:,:2]=np.einsum('nij,nj->ni',T[:,:2],flat-T[:,2]);bary[:,2]=1-bary[:,:2].sum(1)
    height=predict(flat,centre,coeff);normals=normal(flat,centre,coeff)
    edge=np.maximum.reduce([np.linalg.norm(V[:,i]-V[:,j],axis=1) for i,j in [(0,1),(1,2),(2,0)]])
    witness_z=(V[:,:,2]*bary).sum(1);tn=np.cross(V[:,1]-V[:,0],V[:,2]-V[:,0]);tn/=np.maximum(np.linalg.norm(tn,axis=1,keepdims=True),1e-12)
    tn=np.where(tn[:,2:]<0,-tn,tn);normal_angle=np.degrees(np.arccos(np.clip((tn*normals).sum(1),-1,1)))
    frame_support=np.zeros(len(flat),np.int16)
    for r in TRAIN:
        dist,_=cKDTree(X[(raw==r),:2]).query(flat) if (raw==r).any() else (np.full(len(flat),np.inf),None)
        frame_support+=(dist<=.8)
    neighbourhood=tree.query_ball_point(flat,1.);quadrants=np.zeros(len(flat),np.uint8)
    for i,ids in enumerate(neighbourhood):
        delta=P[ids,:2]-flat[i];quadrants[i]=len(np.unique((delta[:,0]>=0).astype(int)*2+(delta[:,1]>=0))) if len(ids) else 0
    flags={'inside_measured_XY_hull':inside,'nearest_measurement_le0p8m':nearest<=.8,'witness_edge_le1p6m':edge<=1.6,
      'witness_height_agreement_le0p05m':abs(height-witness_z)<=.05,'witness_normal_angle_le15deg':normal_angle<=15,
      'at_least3_training_frames':frame_support>=3,'at_least3_support_quadrants':quadrants>=3}
    valid=np.logical_and.reduce(list(flags.values()))&heldout_pass;shape=uv.shape[:2]
    world=np.concatenate([flat,height[:,None]],1).reshape(*shape,3);normals=normals.reshape(*shape,3)
    original=g['geometry_valid'][v0:v1,u0:u1];R=np.eye(3);R[:2,0]=M[:,0]/np.linalg.norm(M[:,0]);R[:,1]=np.cross(R[:,2],R[:,0]);origin3=np.r_[centre,predict(centre[None],centre,coeff)[0]]
    pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=origin3;local=((world-origin3)@R).astype(np.float32)
    geometry_path=out/'local_surface_geometry.npz';np.savez_compressed(geometry_path,
      world_xyz=world,normal_world=normals,geometry_valid=valid.reshape(shape),valid_mask=valid.reshape(shape),
      uv_global=uv.astype(np.float64),shared_atlas_uv=uv.astype(np.float64),local_xyz=local,world_from_patch=pose,
      original_geometry_supported=original,new_inferred_geometry=valid.reshape(shape)&~original,
      bounds_uvxy=np.asarray(UV_BOUNDS),spacing_m=np.asarray(.02),parent_geometry_sha256=pin(grid_path)['sha256'],
      witness_source_raw=raw[selected][tri_indices].reshape(*shape,3),witness_lidar_row_ids=lidar_ids[selected][tri_indices].reshape(*shape,3),
      witness_triangle_world_xyz=V.reshape(*shape,3,3),witness_barycentric=bary.reshape(*shape,3),
      nearest_measured_distance_m=nearest.reshape(shape),distinct_source_frame_support=frame_support.reshape(shape),support_quadrants=quadrants.reshape(shape),
      fit_coefficients=coeff,fit_XY_centre=centre,**{k:v.reshape(shape) for k,v in flags.items()})
    certificate={'schema':'R10.local_nonplanar_geometry_certificate.v1','complete':True,'decision':'PASS' if heldout_pass and valid.any() else 'INSUFFICIENT',
      'scope':'Only geometry_valid=true texels; global metric failures force no valid texels. Invalid points are not rendering support.',
      'geometry_npz':pin(geometry_path),'spacing_m':.02,'bounds_uvxy':UV_BOUNDS,'constant_z_plane_used':False,
      'surface_Z_source':'Robust local quadratic height on strict training LiDAR; old world_from_uv XY only. No old constant Z is used.',
      'train_raw':TRAIN,'heldout_raw':HELDOUT,'frame_counts':records,'heldout_evaluation':evaluations,'global_heldout_thresholds_PASS':bool(heldout_pass),
      'total_patch_texels':len(flat),'certified_texels':int(valid.sum()),'parent_geometry_supported_texels':int(original.sum()),
      'newly_inferred_certified_texels':int((valid.reshape(shape)&~original).sum()),'invalid_by_independent_rule':{k:int((~v).sum()) for k,v in flags.items()},
      'fit_coefficients':coeff.tolist(),'fit_XY_centre':centre.tolist(),'normal_valid_total':norm_count,
      'geometry_only_candidates_not_used_for_strict_fit_or_PASS':True,'original_g_and_models_modified':False,
      'contract':pin(contract_path),'actual_input_SHA_list':pin(out/'input_readback_SHA.json'),'strict_measurements':pin(out/'strict_measurements.npz'),
      'new_GPU_forward':0,'updates':0,'seconds':time.time()-start}
    write(out/'certificate.json',certificate);print(json.dumps(certificate),flush=True)

if __name__=='__main__':main()
