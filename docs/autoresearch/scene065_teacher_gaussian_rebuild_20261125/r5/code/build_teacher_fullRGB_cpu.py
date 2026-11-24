"""Image-driven full-RGB teacher targets and learnable Gaussian seed/candidates.

CPU only. No trained road field, optimizer, shared colour, LF colour, permanent
green carve or hard-final image fill is read or produced by this builder.
"""
from pathlib import Path
import copy, hashlib, json, os, time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

R=Path(__file__).resolve().parent
OLD=Path(r'C:/Users/dengyunlong/Documents/Codex/2026-11-22/065-file-c-users-dengyunlong-documents-2/work')
R4=R.parent/'native30k_red_rebuild_20261125_r4'
CPU=OLD/'cpu_fusion_acceptance_20261124_r2'
R1=OLD/'gpu_clean_rebuild_20261125_r1'
R2=OLD/'clean_local_reconstruction_20261125_r2'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_gaussian_rebuild_20261125_r5'
TRAIN=[106,107,108,109,110,120,137,165]
TEACHERS=[110,137,165]
EVAL=[134,179,181,182,183,184,185,198]
C0=.28209479177387814

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def pin(p,remote=False):
    p=Path(p).resolve()
    return dict(path=REMOTE+'/'+p.relative_to(R).as_posix() if remote else p.as_posix(),sha256=sha(p))
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy()for k in (keys if keys is not None else z.files)}
def eotf(a):return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
def oetf(a):return np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,0)**(1/2.4)-.055)
def sample(a,uv,order=1,mode='constant'):
    shape=uv.shape[:-1];q=np.nan_to_num(uv,nan=-1e6,posinf=-1e6,neginf=-1e6);coord=[q[...,1].ravel(),q[...,0].ravel()]
    if a.ndim==2:return map_coordinates(a.astype(np.float32),coord,order=order,mode=mode,cval=0,prefilter=False).reshape(shape)
    return np.stack([map_coordinates(a[...,k],coord,order=order,mode=mode,cval=0,prefilter=False).reshape(shape)for k in range(a.shape[-1])],-1)
def mask_sample(a,uv,all4=True):
    ok=np.isfinite(uv).all(-1)&(uv[...,0]>=0)&(uv[...,1]>=0)&(uv[...,0]<a.shape[1]-1)&(uv[...,1]<a.shape[0]-1)
    return ok&(sample(a,uv,1 if all4 else 0)>= (1-1e-6 if all4 else .5))
def project(xyz,c,brown=False):
    shape=xyz.shape[:-1];q=xyz.reshape(-1,3)@c['world_to_camera'][:3,:3].T+c['world_to_camera'][:3,3];z=q[:,2]
    den=np.where(abs(z)>1e-12,z,1);x=q[:,0]/den;y=q[:,1]/den
    if brown:
        fx,fy,cx,cy,k1,k2,p1,p2,k3=c['Brown_intrinsic'];r=x*x+y*y;f=1+k1*r+k2*r*r+k3*r*r*r
        uv=np.stack([fx*(x*f+2*p1*x*y+p2*(r+2*x*x))+cx,fy*(y*f+p1*(r+2*y*y)+2*p2*x*y)+cy],-1)
    else:
        k=c['camera_K'];uv=np.stack([k[0,0]*x+k[0,2],k[1,1]*y+k[1,2]],-1)
    return uv.reshape(*shape,2),z.reshape(shape)
def ray_points(c,uv,origin,normal):
    w=np.linalg.inv(c['world_to_camera']);r=np.concatenate([uv,np.ones((*uv.shape[:-1],1))],-1)@np.linalg.inv(c['camera_K']).T;d=r@w[:3,:3].T
    den=d@normal;t=(normal@(origin-w[:3,3]))/np.where(abs(den)>1e-12,den,1)
    return w[:3,3]+t[...,None]*d,(t>.2)&(abs(den)>1e-12),t
def native_rays(c,origin,normal):
    y,x=np.mgrid[:768,:1152];uv=np.stack([x+.5,y+.5],-1);xyz,ok,t=ray_points(c,uv,origin,normal);return xyz,ok,t
def native_full_rgb(path):
    raw=eotf(np.asarray(Image.open(path).convert('RGB'),np.float32)/255)
    h,w=raw.shape[:2];y,x=np.mgrid[:1280,:1920];uv=np.stack([(x+.5)*w/1920-.5,(y+.5)*h/1280-.5],-1)
    return sample(raw,uv,mode='nearest').astype(np.float32),[w,h]
def unknown_from_chart(xyz,geometry):
    o=geometry['cell_world_xyz'][0,0];du=geometry['cell_world_xyz'][0,1]-o;dv=geometry['cell_world_xyz'][1,0]-o
    uv=(xyz[...,:2]-o[:2])@np.linalg.inv(np.stack([du[:2],dv[:2]],1)).T
    ij=np.rint(np.nan_to_num(uv)).astype(np.int64);h,w=geometry['domain'].shape
    inside=(ij[...,0]>=0)&(ij[...,1]>=0)&(ij[...,0]<w)&(ij[...,1]<h)
    x=np.clip(ij[...,0],0,w-1);y=np.clip(ij[...,1],0,h-1)
    domain=inside&geometry['domain'][y,x];unknown=domain&geometry['structural_protection_unknown'][y,x]
    ids=np.where(domain,y*w+x,-1).astype(np.int64)
    return unknown,ids,domain

def make_points(raws,data,cams,origin,normal,pose,geometry,stride):
    chunks=[];inv=np.linalg.inv(pose)
    for raw in raws:
        d=data[raw];yy,xx=np.mgrid[stride//2:768:stride,stride//2:1152:stride];good=d['direct_support'][yy,xx];y=yy[good];x=xx[good]
        uv=np.stack([x+.5,y+.5],-1);xyz,ok,depth=ray_points(cams[raw],uv,origin,normal);assert ok.all()
        # Projected image footprint, not a manually inserted road lattice.
        dx=np.array([stride*.5,0]);dy=np.array([0,stride*.5])
        left=ray_points(cams[raw],uv-dx,origin,normal)[0];right=ray_points(cams[raw],uv+dx,origin,normal)[0]
        up=ray_points(cams[raw],uv-dy,origin,normal)[0];down=ray_points(cams[raw],uv+dy,origin,normal)[0]
        tangent=right-left;foot_x=np.linalg.norm(tangent,axis=-1);tangent/=np.maximum(foot_x[:,None],1e-9)
        bitangent=np.cross(np.broadcast_to(normal,tangent.shape),tangent);bitangent/=np.maximum(np.linalg.norm(bitangent,axis=-1,keepdims=True),1e-9)
        foot_y=np.abs(((down-up)*bitangent).sum(-1));foot=np.stack([foot_x,foot_y],-1)
        scales=np.concatenate([np.clip(foot/2.355,.004,.18),np.full((len(x),1),.003)],-1).astype(np.float32)
        mat=np.stack([tangent,bitangent,np.broadcast_to(normal,tangent.shape)],-1);mat=np.einsum('ij,njk->nik',inv[:3,:3],mat)
        quat=Rotation.from_matrix(mat).as_quat()[:,[3,0,1,2]].astype(np.float32)
        own=d['direct_rgb'][y,x];colour=own.copy();wsum=np.ones(len(x),np.float32);acc=own.copy();count=np.ones(len(x),np.uint8)
        # Keep the source teacher's full texture dominant. Other teachers are
        # weak same-position colour evidence, sampled at the depth prior.
        for other in TEACHERS:
            if other==raw:continue
            suv,z=project(xyz,cams[other]);valid=(z>.2)&mask_sample(data[other]['direct_support'],suv-.5,True)
            value=sample(data[other]['direct_rgb'],suv-.5);weight=.125*valid
            acc+=value*weight[:,None];wsum+=weight;count+=valid
        colour=(acc/wsum[:,None]).astype(np.float32)
        disagreement=np.linalg.norm(colour-own,axis=-1);centre=np.linalg.inv(cams[raw]['world_to_camera'])[:3,3];v=centre-xyz;v/=np.linalg.norm(v,axis=-1,keepdims=True)
        incidence=np.abs(v@normal);unknown,ids,domain=unknown_from_chart(xyz,geometry);assert not unknown.any()
        confidence=np.clip((.5+.25*incidence+.25*np.exp(-disagreement/.03))*np.where(domain,1.,.8),.05,1).astype(np.float32)
        local=(xyz@inv[:3,:3].T+inv[:3,3]).astype(np.float32);srgb=oetf(colour);coef=np.zeros((len(x),16,3),np.float32);coef[:,0]=(srgb-.5)/C0
        chunks.append(dict(xyz=local,world_xyz=xyz.astype(np.float64),rgb=colour,coefficients=coef,scales=scales,rotation=quat,opacity=np.full(len(x),.2,np.float32),active=np.ones(len(x),bool),prior_xyz=local.copy(),prior_world_xyz=xyz.astype(np.float64),normal_local=np.tile(inv[:3,:3]@normal,(len(x),1)).astype(np.float32),prior_normal=np.tile(inv[:3,:3]@normal,(len(x),1)).astype(np.float32),confidence=confidence,source_raw=np.full(len(x),raw,np.int64),source_uv=d['Brown_uv'][y,x].astype(np.float32),source_pinhole_uv=uv.astype(np.float32),source_pixel_xy=np.stack([x,y],-1).astype(np.int32),teacher_color=own.astype(np.float32),fused_teacher_color=colour,independent_teacher_support_count=count,geometry_cell_flat_row_ids=ids,trainable_rgb_rows=np.ones(len(x),bool),structure_unknown=np.zeros(len(x),bool),source_id=np.full(len(x),3,np.int64),teacher_prior_weight=np.zeros(len(x),np.float32),prior_camera_depth_m=depth.astype(np.float32),pixel_footprint_m=foot.astype(np.float32),sample_stride_pixels=np.full(len(x),stride,np.int32),source_independent_teacher=np.ones(len(x),bool),source_is_reprojection=np.zeros(len(x),bool)))
    return {k:np.concatenate([c[k]for c in chunks],axis=0)for k in chunks[0]}

def main():
    start=time.time();R.mkdir(parents=True,exist_ok=True)
    source_shared=CPU/'shared_field_v1/shared_field_CPU_candidate.npz'
    # Geometry and permission arrays only: shared colour is never loaded.
    geometry=load(source_shared,['cell_world_xyz','domain','structural_protection_unknown'])
    world=geometry['cell_world_xyz'];origin=world[0,0];normal=np.cross(world[0,1]-origin,world[1,0]-origin);normal/=np.linalg.norm(normal)
    pose=read(R4/'initialization/pose.json');pose_array=np.asarray(pose['world_from_patch'],np.float64)
    old=read(R2/'config/targets8.remote.json');oldrows={f['raw']:f for f in old['frames']}
    cameras={raw:load(R1/f'supervision/native_baseline/{raw:03}/actual_native_camera.npz')for raw in TRAIN}
    masks={raw:load(R4/f'masks/{raw:03}/native.npz')for raw in TRAIN}
    points={};data={};sources=[]
    for raw in TEACHERS:
        folder=CPU/f'correspondence/formal/gpt{raw:03}_native';receipt=read(folder/'receipt.json');teacher=Path(receipt['source_teacher_image']['path'])
        assert sha(teacher)==receipt['source_teacher_image']['sha256']
        fit_path=CPU/f'photo_anchor/raw{raw:03}_formal_selection_v3/receipt.json';fit=read(fit_path);assert fit['decision']=='PASS_VISIBLE_ASPHALT_COLOUR_MODULE'
        model=fit['selected_model'];full,source_wh=native_full_rgb(teacher)
        canonical_bias=np.array(oldrows[raw]['colour_transform']['bias_linear_RGB'],np.float32)
        full=(full*np.array(model['gain'],np.float32)+np.array(model['bias'],np.float32)+canonical_bias).astype(np.float32)
        fm=load(folder/'masks.npz');semantic_valid=fm['teacher_road_eroded']&~fm['teacher_dynamic']&~fm['teacher_protection']
        obs=load(R1/f'supervision/inputs/{raw:03}/real_observation.npz',['Brown_uv','mapping_valid'])
        xyz,ok,depth=native_rays(cameras[raw],origin,normal);unknown,_,_=unknown_from_chart(xyz,geometry)
        native_rgb=sample(full,obs['Brown_uv']);valid=masks[raw]['red_rebuild']&~masks[raw]['keep_green']&~masks[raw]['keep_orange']&obs['mapping_valid']&ok&mask_sample(semantic_valid,obs['Brown_uv'],True)&~unknown
        valid&=(native_rgb>=0).all(-1)&(native_rgb<=1).all(-1)&np.isfinite(native_rgb).all(-1)
        data[raw]=dict(direct_rgb=native_rgb.astype(np.float32),direct_support=valid,Brown_uv=obs['Brown_uv'])
        points[raw]=(xyz,ok,unknown)
        td=R/f'teachers/{raw:03}';td.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(td/'fullRGB_native.npz',rgb_common_linear=full,teacher_road_support=semantic_valid)
        source_dest=td/'raw_generated.png'
        import shutil
        shutil.copyfile(teacher,source_dest)
        assert sha(source_dest)==receipt['source_teacher_image']['sha256']
        sources.append(dict(raw=raw,independent_teacher=True,raw_teacher=pin(source_dest,True),source_teacher_original=receipt['source_teacher_image'],semantic_receipt=pin(folder/'receipt.json'),semantic_masks=pin(folder/'masks.npz'),colour_fit=pin(fit_path),fullRGB_native=pin(td/'fullRGB_native.npz',True),source_wh=source_wh,fit_gain=model['gain'],fit_bias=model['bias'],canonical_bias=canonical_bias.tolist(),encoding='EOTF once before half-pixel size mapping; FULL linear RGB * accepted fit gain + fit bias + canonical camera bias; no LF filter',direct_native_support_pixels=int(valid.sum()),structural_unknown_loss0_pixels=int((masks[raw]['red_rebuild']&unknown).sum())))
        print(json.dumps(dict(stage='direct_fullRGB',raw=raw,support_pixels=int(valid.sum()))),flush=True)
        del full,fm,native_rgb
    init=R/'initialization';init.mkdir(parents=True,exist_ok=True)
    seed=make_points(TEACHERS,data,cameras,origin,normal,pose_array,geometry,8)
    pool=make_points(TEACHERS,data,cameras,origin,normal,pose_array,geometry,2)
    np.savez_compressed(init/'teacher_seed.npz',**seed);np.savez_compressed(init/'candidate_pool.npz',**pool);write(init/'pose.json',pose)
    np.savez_compressed(init/'row_permissions.npz',trainable_rgb_rows=seed['trainable_rgb_rows'],structure_unknown=seed['structure_unknown'],geometry_cell_flat_row_ids=seed['geometry_cell_flat_row_ids'],initialization_linear_RGB=seed['rgb'])
    print(json.dumps(dict(stage='points',seed_rows=len(seed['xyz']),candidate_rows=len(pool['xyz']),seed_scale_quantiles=np.quantile(seed['scales'][:,:2],[.05,.5,.95]).tolist())),flush=True)
    rows=[];counts=[]
    for previous in old['frames']:
        raw=previous['raw'];row=copy.deepcopy(previous);row['categories']={}
        if raw not in TRAIN:
            row['role']='heldout_render_only';rows.append(row);continue
        m=masks[raw];photo_path=R1/f'supervision/inputs/{raw:03}/real_observation.npz'
        if not photo_path.is_file():photo_path=Path(previous['original_observation']['path'])
        photo=load(photo_path,['rgb_common_linear','Brown_uv','support','mapping_valid'])
        if raw in points:xyz,ok,unknown=points[raw]
        else:
            xyz,ok,_=native_rays(cameras[raw],origin,normal);unknown,_,_=unknown_from_chart(xyz,geometry)
        permit=m['red_rebuild']&~m['keep_green']&~m['keep_orange']&photo['mapping_valid']&ok&~unknown
        if raw in TEACHERS:
            pm=load(CPU/f'correspondence/formal/gpt{raw:03}_native/masks.npz',['photo_safe_dry_asphalt','edit_and_source_shadow_guard'])
            real_safe=pm['photo_safe_dry_asphalt']&~pm['edit_and_source_shadow_guard']
        else:real_safe=load(R1/f'supervision/photometry_pool12_native_v2/masks/{raw:03}/native_safe_asphalt.npz',['safe_dry_asphalt'])['safe_dry_asphalt']
        real=permit&photo['support']&mask_sample(real_safe,photo['Brown_uv'],True)&~m['A9']
        target=np.zeros((*permit.shape,3),np.float32);teacher_support=np.zeros(permit.shape,bool);independent=raw in TEACHERS
        if independent:
            target=data[raw]['direct_rgb'];teacher_support=data[raw]['direct_support']&permit&~real
        else:
            acc=np.zeros_like(target);weights=np.zeros(permit.shape,np.float32)
            for source_raw in TEACHERS:
                uv,z=project(xyz,cameras[source_raw]);valid=(z>.2)&mask_sample(data[source_raw]['direct_support'],uv-.5,True)&permit
                value=sample(data[source_raw]['direct_rgb'],uv-.5);centre=np.linalg.inv(cameras[source_raw]['world_to_camera'])[:3,3];v=centre-xyz;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9);weight=np.abs(v@normal).astype(np.float32)*valid
                acc+=value*weight[...,None];weights+=weight
            has=weights>0;target[has]=acc[has]/weights[has,None];teacher_support=has&permit&~real
        similar=np.zeros(permit.shape,bool)
        folder=R/f'supervision/{raw:03}';folder.mkdir(parents=True,exist_ok=True)
        real_value=photo['rgb_common_linear']+np.array(previous['colour_transform']['bias_linear_RGB'],np.float32)
        for name,value,support in [('real_visible',real_value,real),('teacher_pseudo',target,teacher_support),('similar_reference',target,similar)]:
            rgb=np.where(support[...,None],value,0).astype(np.float32);assert np.isfinite(rgb).all() and ((rgb[support]>=0)&(rgb[support]<=1)).all()
            assert not (support&(unknown|m['keep_green']|m['keep_orange'])).any()
            path=folder/(name+'.npz');np.savez_compressed(path,rgb_common_linear=rgb,support=support,loss_weight=support.astype(np.float32));row['categories'][name]=pin(path,True)
        p=folder/'target_masks.npz';np.savez_compressed(p,asphalt_RGB_permitted=real|teacher_support,target_local_range=m['red_rebuild']&ok,structural_veto=m['red_rebuild']&unknown,red_rebuild=m['red_rebuild'],retained_body=m['keep_green'],keep_orange=m['keep_orange'],real_photo_target=real,shared_teacher_target=teacher_support,similar_reference_target=similar)
        row['target_masks']=pin(p,True);p=folder/'protection.npz';np.savez_compressed(p,K=~m['red_rebuild'],body=m['keep_green'],keep_orange=m['keep_orange'],A9_deleted_instance=m['A9'],projection_valid=m['red_rebuild']&ok,material_permission=~unknown,structural_veto=m['red_rebuild']&unknown);row['protection']=pin(p,True)
        row.update(role='direct_independent_fullRGB_teacher' if independent else 'reprojected_fullRGB_teacher_supplement',independent_teacher=independent,teacher_target_is_geometry_reprojection=not independent,teacher_source_frames=[raw]if independent else TEACHERS,teacher_RGB_frequency='FULL',teacher_colour_encoding='canonical linear',real_visible_pixels=int(real.sum()),external_pseudo_pixels=int(teacher_support.sum()),similar_reference_pixels=0,structural_unknown_RGB_loss_weight=0,old_gray_or_LF_supervision_used=False)
        rows.append(row);counts.append(dict(raw=raw,independent_teacher=independent,real=int(real.sum()),teacher=int(teacher_support.sum()),similar=0,old_structural_unknown_loss0=int((m['red_rebuild']&unknown).sum())));print(json.dumps(counts[-1]),flush=True)
    binding=copy.deepcopy(old);binding.update(schema='scene065.clean_fixed_target_bindings.v1',binding_version='scene065.teacher_fullRGB_Gaussian.r5',frames=rows,complete=True,training_frames=TRAIN,evaluation_frames=EVAL,independent_teacher_frames=TEACHERS,independent_teacher_count=3,category_weights={'real_visible':1.,'teacher_pseudo':.25,'similar_reference':.15},structural_unknown_RGB_loss_weight=0,source_frequency='FULL original GPT-image RGB, never LF',source_teacher_manifest=pin(R/'teachers/manifest.json',True)if(R/'teachers/manifest.json').exists()else None,source_shared_RGB_loaded=False,shared_supervision_accepted=False,parameter_initialization_independent=True,old_student_RGB_read=False,geometry_prior_is_soft=True,permanent_A20_world_carve=False,new_field_initialization_sha256=sha(init/'teacher_seed.npz'),accepted_for='user-authorized image-driven three-dimensional Gaussian completion with full parameter optimization and topology changes')
    write(R/'teachers/manifest.json',dict(schema='scene065.three_original_FULL_RGB_teachers.v1',independent_teacher_count=3,rows=sources,generated_new_teachers=False,hard_final_fill=False))
    binding['source_teacher_manifest']=pin(R/'teachers/manifest.json',True);write(R/'config/targets8.remote.json',binding);write(R/'supervision/counts.json',counts)
    manifest=dict(schema='scene065.image_driven_teacher_gaussian_seed.v1',complete=True,CPU_only=True,GPU_calls=0,teacher_seed=pin(init/'teacher_seed.npz',True),candidate_pool=pin(init/'candidate_pool.npz',True),row_permissions=pin(init/'row_permissions.npz',True),pose=pin(init/'pose.json',True),targets=pin(R/'config/targets8.remote.json',True),teacher_manifest=pin(R/'teachers/manifest.json',True),seed_rows=len(seed['xyz']),candidate_rows=len(pool['xyz']),seed_pixel_stride=8,candidate_pixel_stride=2,all_parameters_learnable=['xyz','scale','rotation','opacity','DC'],geometry_source='teacher camera rays intersect previously verified affine road plane to initialize soft depth prior; not measured geometry and not a fixed planar field',prior_normal=normal.tolist(),prior_plane_origin=origin.tolist(),pixel_footprint_rule='ray intersections at +/-stride/2 pixels; tangent sigma=footprint/2.355, clipped4mm..18cm; normal sigma3mm',opacity_initial=.2,seed_colour='source original FULL RGB dominant, weak .125 weight per valid other teacher, all canonical linear; no median or LF',unknown_structure='old geometry chart explicit unknown pixels loss0; no seed/candidate drawn from them',source_shared_arrays_read=['cell_world_xyz','domain','structural_protection_unknown'],source_shared_RGB_read=False,old_trained_RGB_read=False,old_optimizer_read=False,old_road_render_RGB_read=False,permanent_A20_world_carve=False,direct_target='raw teacher FULL RGB sampled directly into its camera image; not a fixed-plane teacher re-render',other_five_views='three teachers reprojected via geometry depth prior, non-independent inferred supplement',training_frames=TRAIN,evaluation_frames=EVAL,seconds=time.time()-start)
    write(init/'manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
