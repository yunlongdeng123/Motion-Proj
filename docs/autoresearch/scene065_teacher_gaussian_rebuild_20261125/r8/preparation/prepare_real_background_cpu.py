"""CPU-only fixed visible-background observations and trainable Gaussian seed."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json, hashlib, time, shutil
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, binary_dilation
from scipy.spatial.transform import Rotation

R=Path(__file__).resolve().parent
CURRENT=R.parent.parent
OLD=Path('C:/Users/dengyunlong/Documents/Codex/2026-11-22/065-file-c-users-dengyunlong-documents-2/work')
R1=OLD/'gpu_clean_rebuild_20261125_r1'
R4=R.parent/'native30k_red_rebuild_20261125_r4'
R5=R.parent/'teacher_gaussian_rebuild_20261125_r5'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_background_continuation_20261125_r8'
TRAIN=[106,107,108,109,110,120,137,165]
SEED_FRAMES=[110,137,165]
C0=.28209479177387814

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p,remote=False):
    p=Path(p).resolve()
    return dict(path=REMOTE+'/'+p.relative_to(R).as_posix()if remote else p.as_posix(),sha256=sha(p),bytes=p.stat().st_size)
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def sample(a,uv):
    return map_coordinates(a.astype(np.float32),[uv[...,1],uv[...,0]],order=1,mode='constant',cval=0,prefilter=False)
def oetf(a):return np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,0)**(1/2.4)-.055)
def rays(c,uv,origin,normal):
    w=np.linalg.inv(c['world_to_camera']);ray=np.concatenate([uv,np.ones((*uv.shape[:-1],1))],-1)@np.linalg.inv(c['camera_K']).T
    direction=ray@w[:3,:3].T;den=direction@normal;t=(normal@(origin-w[:3,3]))/np.where(abs(den)>1e-12,den,1)
    return w[:3,3]+t[...,None]*direction,(t>.2)&(abs(den)>1e-12),t

def points(data,cameras,pose,origin,normal,stride):
    chunks=[];inv=np.linalg.inv(pose)
    for raw in SEED_FRAMES:
        d=data[raw];yy,xx=np.mgrid[0:768:stride,0:1152:stride];good=d['support'][yy,xx]
        y=yy[good];x=xx[good];uv=np.stack([x+.5,y+.5],-1);n=len(x)
        xyz,ok,depth=rays(cameras[raw],uv,origin,normal);assert ok.all()
        dx=np.array([stride/2,0]);dy=np.array([0,stride/2])
        tx=rays(cameras[raw],uv+dx,origin,normal)[0]-rays(cameras[raw],uv-dx,origin,normal)[0]
        ty=rays(cameras[raw],uv+dy,origin,normal)[0]-rays(cameras[raw],uv-dy,origin,normal)[0]
        fx=np.linalg.norm(tx,axis=-1);tangent=tx/np.maximum(fx[:,None],1e-12)
        bitangent=np.cross(np.broadcast_to(normal,tangent.shape),tangent);bitangent/=np.maximum(np.linalg.norm(bitangent,axis=-1,keepdims=True),1e-12)
        fy=np.abs((ty*bitangent).sum(-1));foot=np.stack([fx,fy],-1)
        scale=np.concatenate([np.clip(foot/2.355,.004,.18),np.full((n,1),.003)],-1).astype(np.float32)
        mat=np.stack([tangent,bitangent,np.broadcast_to(normal,tangent.shape)],-1);mat=np.einsum('ij,njk->nik',inv[:3,:3],mat)
        quat=Rotation.from_matrix(mat).as_quat()[:,[3,0,1,2]].astype(np.float32)
        local=(xyz@inv[:3,:3].T+inv[:3,3]).astype(np.float32);rgb=d['rgb'][y,x].copy()
        coef=np.zeros((n,16,3),np.float32);coef[:,0]=(oetf(rgb)-.5)/C0
        ids=10**12+raw*1_000_000+y.astype(np.int64)*1152+x
        chunks.append(dict(xyz=local,world_xyz=xyz.astype(np.float64),scales=scale,rotation=quat,opacity=np.full(n,.2,np.float32),
            coefficients=coef,coeffs=coef,rgb=rgb,active=np.ones(n,bool),prior_xyz=local.copy(),prior_world_xyz=xyz.astype(np.float64),
            prior_normal=np.tile(inv[:3,:3]@normal,(n,1)).astype(np.float32),normal_local=np.tile(inv[:3,:3]@normal,(n,1)).astype(np.float32),
            confidence=np.ones(n,np.float32),source_raw=np.full(n,raw,np.int64),source_uv=d['Brown_uv'][y,x].astype(np.float32),
            source_pinhole_uv=uv.astype(np.float32),source_pixel_xy=np.stack([x,y],-1).astype(np.int32),source_candidate_ids=ids,
            source_is_real_background=np.ones(n,bool),source_independent_teacher=np.zeros(n,bool),source_is_reprojection=np.zeros(n,bool),
            source_id=np.full(n,8,np.int64),trainable_rgb_rows=np.ones(n,bool),structure_unknown=np.zeros(n,bool),
            teacher_prior_weight=np.zeros(n,np.float32),prior_camera_depth_m=depth.astype(np.float32),pixel_footprint_m=foot.astype(np.float32),
            sample_stride_pixels=np.full(n,stride,np.int32)))
    result={k:np.concatenate([c[k]for c in chunks],axis=0)for k in chunks[0]}
    assert len(result['xyz'])>0 and np.unique(result['source_candidate_ids']).size==len(result['xyz'])
    assert all(np.isfinite(v).all()for v in result.values())
    return result

def main():
    begin=time.time();observations={v['raw']:Path(v['local_path'])for v in read(R1/'supervision/real_observation_takeback.json')}
    binding=read(OLD/'clean_local_reconstruction_20261125_r2/config/targets8.remote.json');oldrows={v['raw']:v for v in binding['frames']}
    geometry_path=OLD/'road_underbody_local_20261123_r10/audit/timewindow17_geometry_v1/candidate_reprojected_v1/candidate_geometry.npz'
    geo=load(geometry_path);coef=geo['fit_coefficients'];centre=geo['fit_XY_centre'];origin=np.array([centre[0],centre[1],coef[0]])
    normal=np.array([-coef[1],-coef[2],1.]);normal/=np.linalg.norm(normal)
    pose_path=R5/'initialization/pose.json';pose=read(pose_path);pose_array=np.asarray(pose['world_from_patch'],np.float64)
    damage_manifest=read(R/'inputs/damage/manifest.json');damage_rows={v['raw']:v for v in damage_manifest['frames']}
    data={};cameras={};rows=[]
    for raw in TRAIN:
        photo_path=observations[raw];p=load(photo_path);uv=p['Brown_uv'];camera_path=R1/f'supervision/native_baseline/{raw:03}/actual_native_camera.npz'
        camera=load(camera_path);cameras[raw]=camera
        mask_path=R4/f'masks/{raw:03}/native.npz';m=load(mask_path);orange=m['keep_orange'];green=m['keep_green']
        instance_path=R4/f'source_masks/photo_instances/{raw:03}.png';instances=np.asarray(Image.open(instance_path))
        dynamic=sample(binary_dilation(instances>0,iterations=1),uv)>1e-6
        a9=sample(binary_dilation(instances==10,iterations=1),uv)>1e-6
        guard_path=OLD/f'cpu_fusion_acceptance_20261124_r2/correspondence/formal/gpt{raw:03}_native/masks.npz'
        official=guard_path.exists()
        if official:
            guards=load(guard_path);shadow=sample(guards['edit_and_source_shadow_guard'],uv)>1e-6
            road=sample(guards['photo_road'],uv)>=1-1e-6;lane=sample(guards['photo_lane'],uv)>=1-1e-6
            guard_kind='official edit_and_source_shadow_guard'
        else:
            guard_path=R1/f'supervision/photometry_pool12_native_v2/masks/{raw:03}/native_safe_asphalt.npz';guards=load(guard_path)
            shadow=sample(guards['dark_or_shadow'],uv)>1e-6
            shadow|=sample(guards['original_dynamic_guard'],uv)>1e-6
            road=sample(guards['road'],uv)>=1-1e-6;lane=sample(guards['lane'],uv)>=1-1e-6
            guard_kind='conservative original-photo dark_or_shadow + original_dynamic_guard; official A9 shadow unavailable'
        damage_path=R/f'inputs/damage/{raw:03}.npz';damage=load(damage_path)['rgb_changed']
        scope=orange&damage&~green&~dynamic&~a9&~m['A9']&~shadow
        y,x=np.mgrid[:768,:1152];_,ray_ok,_=rays(camera,np.stack([x+.5,y+.5],-1),origin,normal)
        rgb=p['rgb_common_linear']+np.asarray(oldrows[raw]['colour_transform']['bias_linear_RGB'],np.float32)
        support=scope&p['support']&p['mapping_valid']&road&ray_ok&np.isfinite(rgb).all(-1)&(rgb>=0).all(-1)&(rgb<=1).all(-1)
        value=np.where(support[...,None],rgb,0).astype(np.float32)
        assert not(support&(~orange|green|dynamic|shadow|a9|m['A9'])).any()
        folder=R/f'supervision/{raw:03}';folder.mkdir(parents=True,exist_ok=True)
        target=folder/'real_visible_background.npz';np.savez_compressed(target,rgb_common_linear=value,support=support,
            loss_weight=support.astype(np.float32),confidence=support.astype(np.float32),unknown_RGB_loss0=scope&~support)
        masks_path=folder/'fixed_bg_masks.npz';np.savez_compressed(masks_path,bg_rebuild=scope,bg_geometry_scope=scope,
            real_visible=support,keep_orange=orange,keep_green=green,baseline_changed=damage,A9=a9|m['A9'],
            source_shadow_guard=shadow,dynamic=dynamic,unknown_RGB_loss0=scope&~support,photo_lane=lane&support)
        counts=dict(orange_pixels=int(orange.sum()),orange_changed_pixels=int((orange&damage).sum()),
            bg_scope_pixels=int(scope.sum()),real_visible_pixels=int(support.sum()),unknown_RGB_loss0_pixels=int((scope&~support).sum()),
            lane_GT_pixels=int((lane&support).sum()),overlap_outside_orange=0,overlap_green_dynamic_A9_shadow=0)
        row=dict(raw=raw,original_observation=pin(photo_path),camera=pin(camera_path),original_orange_masks=pin(mask_path),
            original_photo_instances=pin(instance_path),shadow_guard=pin(guard_path),shadow_guard_kind=guard_kind,
            damage_sources=damage_rows[raw],damage_mask=pin(damage_path,True),colour_transform=oldrows[raw]['colour_transform'],
            real_visible=pin(target,True),fixed_masks=pin(masks_path,True),weight=1.0,RGB_source='same-position original photo canonical linear; no rendered or teacher RGB',**counts)
        rows.append(row);data[raw]=dict(rgb=rgb,support=support,Brown_uv=uv)
        print(json.dumps(dict(raw=raw,**counts)),flush=True)
        if raw in SEED_FRAMES:
            img=(np.clip(oetf(rgb),0,1)*255).astype(np.uint8);over=img.copy();over[orange]=(over[orange]*.55+np.array([255,145,20])*.45).astype(np.uint8)
            over[support]=(over[support]*.35+np.array([40,180,255])*.65).astype(np.uint8)
            preview=Image.fromarray(over);draw=ImageDraw.Draw(preview);draw.rectangle((0,0,1151,30),fill=(15,20,28));draw.text((12,10),f'raw {raw} | orange: original keep scope | blue: fixed real-photo BG support ({support.sum()} px)',fill='white')
            (R/'previews').mkdir(exist_ok=True);preview.save(R/f'previews/{raw:03}_fixed_BG_scope.png')
    init=R/'initialization';init.mkdir(exist_ok=True)
    seed=points(data,cameras,pose_array,origin,normal,4);pool=points(data,cameras,pose_array,origin,normal,2)
    assert np.isin(seed['source_candidate_ids'],pool['source_candidate_ids']).all()
    seed_path=init/'bg_seed.npz';pool_path=init/'bg_candidate_pool.npz';np.savez_compressed(seed_path,**seed);np.savez_compressed(pool_path,**pool);write(init/'pose.json',pose)
    manifest=dict(schema='scene065.real_visible_background_continuation.v1',complete=True,CPU_only=True,GPU_calls=0,
        training_frames=TRAIN,seed_frames=SEED_FRAMES,candidate_frames=SEED_FRAMES,source_photo_only=True,teacher_generated=False,
        geometry_initialization_is_soft=True,A9_and_shadow_excluded=True,dynamic_and_green_excluded=True,fixed_damage_masks=True,
        fixed_scope_rule='original R4 orange AND max(abs(original Native30k baseline RGB - R7 frozen context RGB)) > 0.04; exclude original-photo dynamic/A9/shadow and green',
        context_total_alpha_used=False,real_visible_weight=1.,unknown_RGB_loss_weight=0.,seed_pixel_stride=4,candidate_pixel_stride=2,
        seed_rows=len(seed['xyz']),candidate_rows=len(pool['xyz']),seed=pin(seed_path,True),candidate_pool=pin(pool_path,True),pose=pin(init/'pose.json',True),
        source_candidate_id_rule='1000000000000 + raw*1000000 + native_y*1152 + native_x; seed and pool identical pixel IDs',
        source_pixel_xy_convention='integer native pinhole x,y at 1152x768; source_pinhole_uv=x+.5,y+.5; source_uv original Brown photo coordinates only provenance',
        geometry_source=pin(geometry_path),prior_plane_origin_world=origin.tolist(),prior_normal_world=normal.tolist(),
        geometry_description='camera-ray intersection with previously verified affine road plane is soft inferred initial geometry, not measured depth',
        all_parameters_learnable=['xyz','log_scales','raw_quaternion','opacity_logits','dc'],opacity_initial=.2,
        scale_rule='stride-pixel ray-plane footprint/2.355, tangent sigma clipped 0.004 to 0.18 m, normal sigma 0.003 m',
        old_student_RGB_used=False,render_RGB_used_as_GT=False,A20_source_replaced=False,frames=rows,seconds=time.time()-begin)
    write(R/'manifest.json',manifest);write(init/'manifest.json',manifest)
    print(json.dumps(dict(complete=True,seed_rows=len(seed['xyz']),candidate_rows=len(pool['xyz']),manifest=pin(R/'manifest.json',True))),flush=True)

if __name__=='__main__':main()
