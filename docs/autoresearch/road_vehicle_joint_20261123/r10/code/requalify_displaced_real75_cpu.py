"""Read-only fresh-photo qualification of displaced inherited real atlas addresses."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_k]='2'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion,binary_dilation,gaussian_filter,laplace,map_coordinates
from scipy.spatial import Delaunay,cKDTree

sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
from drivingeditor_dynamic_edit.v44_road_observation import transform,project,ray_box_blocked,build_foreground_guard

A9='NTjq-OhZ_97OMJ4xwJAxXg'
def load(p):
    with np.load(p,allow_pickle=False) as n:return {k:n[k].copy() for k in n.files}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def brown(X,E,intr):
    c=transform(X,E);z=c[:,2];x,y=(c[:,:2]/np.maximum(z[:,None],1e-8)).T
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intr;r=x*x+y*y;f=1+k1*r+k2*r*r+k3*r*r*r
    return np.c_[fx*(x*f+2*p1*x*y+p2*(r+2*x*x))+cx-.5,fy*(y*f+p1*(r+2*y*y)+2*p2*x*y)+cy-.5],z
def all4(mask,uv,cval=False):
    xy=np.floor(np.nan_to_num(uv,nan=-1e6,posinf=1e6,neginf=-1e6)).astype(np.int64);h,w=mask.shape
    inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<w-1)&(xy[:,1]<h-1)
    out=np.full(len(uv),bool(cval));x,y=xy[inside].T
    out[inside]=mask[y,x]&mask[y,x+1]&mask[y+1,x]&mask[y+1,x+1]
    return out
def sample(a,uv,order=1,cval=0):return map_coordinates(np.asarray(a,np.float32),[uv[:,1],uv[:,0]],order=order,mode='constant',cval=cval)
def bilinear(image,uv):
    h,w=image.shape[:2];xy=np.floor(uv).astype(int);x=np.clip(xy[:,0],0,w-2);y=np.clip(xy[:,1],0,h-2);d=uv-xy
    return sum(linear(image[y+j,x+i].astype(np.float32)/255)*((d[:,0] if i else 1-d[:,0])*(d[:,1] if j else 1-d[:,1]))[:,None] for i,j in [(0,0),(1,0),(0,1),(1,1)]).astype(np.float32)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['r7','r10','geometry','displacement','training-points','output']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.time()
    C=a.r10.parent;B=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1');BANK=B/'inputs';S=C/'road_multisource_20261122_r1/sources';ANN=C/'waymo_val065_road_completion_20261111_r1/raw_road_annotations'
    d=load(a.displacement);g=load(a.geometry);t=load(a.training_points);Q=d['new_candidate_worldXYZ'];UV=d['global_uv'].astype(np.int32);n=len(Q);assert n==75
    guv=g['uv_global'].astype(int);lookup={tuple(u):i for i,u in enumerate(guv.reshape(-1,2))};ids=np.array([lookup[tuple(u)] for u in UV]);norm=g['normal_world'].reshape(-1,3)[ids]
    gains=np.array([r['gain'] for r in read(a.r7/'exposure_gain.json')['records']],np.float32);quality=np.asarray(read(S/'source_quality.json')['quality'])
    frames=list(range(90,171,5));assert list(t['reference_frames'])==frames
    globalpins=[pin(p) for p in [a.geometry,a.displacement,a.training_points,a.r7/'exposure_gain.json',S/'source_quality.json',Path(__file__)]]
    records=[];frame_summary=[];allpins=[]
    for raw in frames:
        op=BANK/f'observations/{raw:03d}.json';annp=ANN/f'{raw:03d}.json';cp=S/f'qualified_observations/{raw:03d}.npz'
        o=read(op);ann=read(annp);cam=load(cp);E=cam['world_to_camera'];K=cam['camera_K'];uvpin,depth=project(Q,K,E);uvB,_=brown(Q,E,np.asarray(o['camera_intrinsic_Brown']));uvc=(uvB+.5)*.25-.5
        pp=BANK/o['image_path'];semp=B/'semantics'/f'{raw:03d}.png';fgp=B/'repaired_masks_all_r4'/f'{raw:03d}/dynamic_union.png';labp=fgp.with_name('labels.png');lidp=BANK/o['lidar_path']
        image=np.asarray(Image.open(pp).convert('RGB'));h,w=image.shape[:2];semfull=np.asarray(Image.open(semp));sem=np.asarray(Image.fromarray(semfull).resize((480,320),Image.Resampling.NEAREST));road=binary_erosion(sem==0,iterations=1)
        fg=np.asarray(Image.open(fgp))>0;labels=np.asarray(Image.open(labp))>0;rects=np.asarray([b['xyxy'] for b in ann['front_camera_rectangles']]).reshape(-1,4)*.25
        guard=build_foreground_guard((320,480),rects,binary_dilation(fg|labels,iterations=2),margin_px=2)
        small=np.asarray(Image.fromarray(image).resize((480,320),Image.Resampling.LANCZOS),np.float32)/255;lum=linear(small)@np.array([.2126,.7152,.0722],np.float32)
        clear=road&~guard;den=gaussian_filter(clear.astype(np.float32),12);local=gaussian_filter(lum*clear,12)/np.maximum(den,1e-6)
        shadow=(lum<.72*local)&((local-lum)>.02)&(den>.2)&road
        centers=np.asarray([b['center'] for b in ann['boxes']]).reshape(-1,3);halves=np.asarray([b['half'] for b in ann['boxes']]).reshape(-1,3);axes=np.asarray([b['axes'] for b in ann['boxes']]).reshape(-1,3,3)
        origin=np.linalg.inv(E)[:3,3];ray=origin-Q;cos=abs(np.sum(ray/np.maximum(np.linalg.norm(ray,axis=1)[:,None],1e-8)*norm,1));area=K[0,0]*K[1,1]*.2**2*cos/np.maximum(depth**2,1e-8);foot=K[0,0]*K[1,1]*16*.02**2*cos/np.maximum(depth**2,1e-8)
        near=np.zeros(n,bool);a9near=np.zeros(n,bool);a9indices=[]
        for k,(box,c,half,ax) in enumerate(zip(ann['boxes'],centers,halves,axes)):
            if box['type'] in [1,2,4]:
                this=(abs(((Q-c)@ax)[:,:2])<=half[:2]+2).all(1);near|=this
                if box.get('track')==A9:a9near|=this;a9indices.append(k)
        qlum=sample(lum,uvc);qlocal=sample(local,uvc);near_shadow=near&(qlum<.85*qlocal)&((qlocal-qlum)>.012)
        shadowbad=~all4(~shadow,uvc)|near_shadow;a9shadow=a9near&(qlum<.85*qlocal)&((qlocal-qlum)>.012)
        inbounds=(depth>.1)&np.isfinite(uvB).all(1)&((uvB>=[0,0])&(uvB<[w-1,h-1])).all(1);roadclear=all4(road,uvc);guardclear=all4(~guard,uvc)
        sourceS=np.zeros(n,bool);extra_pins=[]
        if raw in [86,110]:
            mp=S/f'maps/{raw:03d}.npz';m=load(mp);sourceS=sample(m['requested_S'],(uvpin+.5)*2.4-.5,0,1)>0;extra_pins.append(pin(mp))
        blocked=ray_box_blocked(Q,origin,centers,halves,axes)
        a9blocked=ray_box_blocked(Q,origin,centers[a9indices],halves[a9indices],axes[a9indices]) if a9indices else np.zeros(n,bool)
        # Triangles are rebuilt at the new point from this frame's actual official ROAD/LANE points.
        fi=np.flatnonzero(t['source_raw']==raw);Vall=t['world_xyz'][fi];vpin,vz=project(Vall,K,E);vgood=(vz>.1)&np.isfinite(vpin).all(1);fi=fi[vgood];Vall=Vall[vgood];vpin=vpin[vgood];vz=vz[vgood]
        _,un=np.unique(np.round(vpin,9),axis=0,return_index=True);fi,Vall,vpin,vz=fi[un],Vall[un],vpin[un],vz[un]
        tri=Delaunay(vpin);simplex=tri.find_simplex(uvpin);tri_inside=simplex>=0;triids=tri.simplices[np.maximum(simplex,0)];V=Vall[triids];P=vpin[triids];Z=vz[triids]
        delta=uvpin-P[:,0];aa=P[:,1]-P[:,0];bb=P[:,2]-P[:,0];det=aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0];safe=np.where(abs(det)>1e-10,det,1)
        b1=(delta[:,0]*bb[:,1]-delta[:,1]*bb[:,0])/safe;b2=(aa[:,0]*delta[:,1]-aa[:,1]*delta[:,0])/safe;bary=np.c_[1-b1-b2,b1,b2]
        measured=1/np.maximum((bary/np.maximum(Z,1e-8)).sum(1),1e-8);edge=np.stack([np.linalg.norm(V[:,i]-V[:,j],axis=1) for i,j in [(0,1),(1,2),(2,0)]],1).max(1)
        pedge=np.stack([np.linalg.norm(P[:,i]-P[:,j],axis=1) for i,j in [(0,1),(1,2),(2,0)]],1).max(1);tn=np.cross(V[:,1]-V[:,0],V[:,2]-V[:,0]);tnlen=np.linalg.norm(tn,axis=1);normal_cos=abs((tn*norm).sum(1))/np.maximum(tnlen,1e-8)
        signed_plane=((V-Q[:,None,:])*norm[:,None,:]).sum(2)
        triok=tri_inside&(abs(det)>1e-10)&(bary>=-1e-6).all(1)&(bary<=1+1e-6).all(1)&(edge<=1.6)&(pedge<=24)&(tnlen>1e-8)&(normal_cos>=.9)&(abs(measured-depth)<=.3)&(abs(signed_plane)<=.15).all(1)
        # Full unlabelled LiDAR remains in the visibility veto; no height-residual filtering of the cloud.
        cloud=np.fromfile(lidp,dtype='<f4').reshape(-1,6)[:,:3];cloud=transform(cloud,np.asarray(o['ego_world']).reshape(4,4));luv,ld=project(cloud,K,E)
        li=np.flatnonzero((ld>.1)&np.isfinite(luv).all(1)&((luv>=[0,0])&(luv<[479,319])).all(1));pix=np.floor(luv[li]+.5).astype(int);order=np.argsort(ld[li],kind='stable');_,first=np.unique(pix[order,1]*480+pix[order,0],return_index=True);nearest=li[order[first]]
        dist,hit=cKDTree(luv[nearest]).query(uvpin);hit=nearest[hit];hit_plane=abs(((cloud[hit]-Q)*norm).sum(1))
        conflict=(dist<=1.5)&((hit_plane>.15)|(abs(ld[hit]-depth)>.3))
        goodangle=(cos>=.12)&(area>=.25)&(foot>=.25)&(quality[raw]>=.2)
        qualified=inbounds&roadclear&guardclear&~shadowbad&~sourceS&~blocked&triok&~conflict&goodangle
        colors=np.full((n,3),np.nan,np.float32);colors[inbounds]=bilinear(image,uvB[inbounds]);common=colors*gains[raw]
        sharp=sample(abs(laplace(gaussian_filter(lum,.6))),uvc);weight=(quality[raw]*cos/np.maximum(depth,.1)**2*np.minimum(foot,1)).astype(np.float32)
        xy=np.clip(np.floor(uvB).astype(int),[0,0],[w-2,h-2]);sem4=np.stack([semfull[xy[:,1]+j,xy[:,0]+i] for i,j in [(0,0),(1,0),(0,1),(1,1)]],1)
        record=dict(source_raw=np.full(n,raw,np.int16),global_uv=UV,new_world_xyz=Q,source_uv_Brown=uvB,source_uv_pinhole480=uvpin,query_depth_m=depth,
            raw_linear=colors,canonical_linear_rgb=common,exposure_gain=np.full(n,gains[raw],np.float32),weight=weight,sharpness_proxy=sharp,
            in_image=inbounds,photo_semantic4=sem4,road_all4=roadclear,dynamic_rectangle_guard_clear_all4=guardclear,shadow_proxy_bad=shadowbad,near_vehicle_shadow_proxy=near_shadow,
            target_A9_near_shadow_proxy=a9shadow,target_A9_ray_blocked=a9blocked,all_GT_OBB_ray_blocked=blocked,explicit_source_target_shadow_S=sourceS,
            triangle_contains=tri_inside,triangle_qualified=triok,triangle_source_lidar_row_ids=t['source_lidar_row_id'][fi[triids]],triangle_official_semantic=t['official_semantic'][fi[triids]],triangle_official_front_uv=t['official_front_uv'][fi[triids]],
            triangle_barycentric=bary,triangle_world_edge_m=edge,triangle_pinhole480_edge_px=pedge,triangle_normal_cosine=normal_cos,triangle_vertex_plane_offset_m=signed_plane,measured_depth_m=measured,
            nearest_return_row_id=hit.astype(np.int32),nearest_return_distance_pinhole480_px=dist,nearest_return_depth_m=ld[hit],nearest_return_plane_distance_m=hit_plane,nearest_return_conflict=conflict,
            footprint_qualified=goodangle,qualified_proxy_certificate=qualified,strict_shadow_GT_verified=np.zeros(n,bool))
        np.savez_compressed(a.output/f'{raw:03d}_candidates.npz',**record);records.append(record)
        reason_masks=dict(out_of_image=~inbounds,nonroad=~roadclear,dynamic_or_rectangle_guard=~guardclear,shadow_proxy=shadowbad,explicit_target_shadow_S=sourceS,GT_OBB_blocked=blocked,triangle=~triok,nearest_return_conflict=conflict,footprint=~goodangle)
        row=dict(raw=raw,candidates=n,qualified_proxy=int(qualified.sum()),rejections_nonexclusive={k:int(v.sum()) for k,v in reason_masks.items()},target_A9_ray_blocked=int(a9blocked.sum()),target_A9_near_shadow_proxy=int(a9shadow.sum()),quality=float(quality[raw]),exposure_gain=float(gains[raw]))
        frame_summary.append(row);allpins.append(dict(raw=raw,inputs=[pin(p) for p in [op,annp,cp,pp,semp,fgp,labp,lidp]]+extra_pins))
        overlay=Image.fromarray(image);draw=ImageDraw.Draw(overlay)
        for i,(x,y) in enumerate(uvB):
            if inbounds[i]:draw.ellipse((x-3,y-3,x+3,y+3),outline=(0,255,70) if qualified[i] else (255,60,40),width=2)
        overlay.save(a.output/f'{raw:03d}_original_Brown_qualification.png');print(json.dumps(row),flush=True)
    qualified=np.stack([r['qualified_proxy_certificate'] for r in records]);common=np.stack([r['canonical_linear_rgb'] for r in records]);weights=np.stack([r['weight'] for r in records]);sharp=np.stack([r['sharpness_proxy'] for r in records])
    count=qualified.sum(0);fresh=np.zeros(n,bool);rgb=np.zeros((n,3),np.float32);agree=np.zeros_like(qualified);nweights=np.zeros_like(weights);span=np.zeros(n,np.int16)
    for j in range(n):
        ix=np.flatnonzero(qualified[:,j]);c=common[ix,j]
        if len(ix):
            lum=c@np.array([.2126,.7152,.0722]);q75=np.quantile(lum,.75);upper=(lum>=q75-.025)&(lum<=q75+.06);proto=np.median(c[upper],axis=0);ok=upper&(abs(c-proto).max(1)<.035);ki=ix[ok];agree[ki,j]=True
            if len(ki):span[j]=frames[ki[-1]]-frames[ki[0]]
            if len(ki)>=3 and span[j]>=6:
                fresh[j]=True;ww=weights[ki,j];ww=ww/max(ww.sum(),1e-20);nweights[ki,j]=ww;rgb[j]=(common[ki,j]*ww[:,None]).sum(0)
    np.savez_compressed(a.output/'fresh_qualification75.npz',global_uv=UV,new_world_xyz=Q,frames=np.asarray(frames,np.int16),all_candidate_canonical_linear_rgb=common,all_candidate_qualified_proxy=qualified,
        all_candidate_weight=weights,all_candidate_sharpness_proxy=sharp,all_candidate_agree=agree,all_candidate_normalized_weight=nweights,fresh_known=fresh,fresh_canonical_linear_rgb=rgb,
        qualified_distinct_count=count.astype(np.int16),agree_distinct_count=agree.sum(0).astype(np.int16),agree_raw_span=span,strict_shadow_GT_verified=np.zeros(n,bool))
    summary=dict(schema='R10.displaced75.fresh_Brown_photo_qualification.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,new_diffusion_calls=0,updates=0,displaced_ancestor_real_addresses=n,
        examined_frames=frames,candidate_records=n*len(frames),qualified_proxy_records=int(qualified.sum()),at_least_one_qualified=int((count>0).sum()),fresh_known=int(fresh.sum()),unrecertified=int((~fresh).sum()),
        inherited_unaffected_real_locks=6332,new_branch_real_lock_total=6332+int(fresh.sum()),minimum_distinct_frames=3,minimum_raw_span=6,frames=frame_summary,
        exposure='common_linear=bilinear(EOTF(original_Brown_RGB))*R7_gain(raw), exactly once',geometry='fresh pinhole triangle from fixed late17 actual official ROAD18/LANE19 cache; all original unfiltered LiDAR used for nearest-return visibility veto',
        triangle_limits=dict(world_edge_m=1.6,pinhole480_edge_px=24,normal_cosine=.9,vertex_plane_distance_m=.15,query_depth_difference_m=.3),
        certificate_status='Fresh observation qualified under the same explicit proxy dynamic/shadow convention as R7; independent shadow GT is unavailable and is not claimed. Per-frame all candidates and reasons remain available.',
        no_parent_field_modified=True,branch_field_not_yet_written=True,old_eb86_5eeb_6407_retained=True,mutable_future_scope='new material extension union these 75 qualification-repair addresses; other RGB and 6332 real locks exact',
        inputs=globalpins,frame_inputs=allpins,outputs=dict(fresh_qualification=pin(a.output/'fresh_qualification75.npz')),seconds=time.time()-start)
    write(a.output/'receipt.json',summary);print(json.dumps({k:summary[k] for k in ['complete','candidate_records','qualified_proxy_records','at_least_one_qualified','fresh_known','unrecertified','new_branch_real_lock_total','seconds']}),flush=True)

if __name__=='__main__':main()
