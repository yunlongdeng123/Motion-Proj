"""View-conditioned surface resampling: persistent3D raw106 chart prototype.

Depth is the PVD proposal's expected camera-z, never observed depth truth.
No optimization, PVD update, image overlay, semantic half-object rule, or sigma search.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import hashlib,json
from pathlib import Path
import sys,time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.sparse import load_npz
from scipy.spatial import cKDTree
import torch

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_a6_appearance_formal_20261105'
DEST=RUN/'output_adapter/projective_surface'
SOURCE=RUN/'output_adapter/sphere_strong_v2/complete_a6.npz'
PROPOSAL=RUN/'inputs/raw106/strong_candidate_v2_512.png'
PACKET=RUN/'output_adapter/sphere_inputs_hr/raw106'
STRIDE=1;SIGMA_PIXELS=.6;OPACITY=.99;RAW=106
sys.path.insert(0,str(RUN/'source'))
from highres_fit import camera_hr,RAW as RGBROOT
from scene_runtime import Runtime,raster,tagged
from sh_asset import SHAsset,hash_geometry,C0
from full_pvd import GEOMETRY_FIELDS


def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
    a=np.asarray(a,float);q=np.quantile(a,[0,.1,.5,.9,.99,1])
    return dict(min=q[0],p10=q[1],median=q[2],p90=q[3],p99=q[4],max=q[5],mean=float(a.mean()))
def project(model,full,w=1920,h=1280):
    hp=np.column_stack((model,np.ones(len(model))))@full
    ndc=hp[:,:2]/hp[:,3,None]
    return ((ndc+1)*np.array([w,h])-1)/2


@torch.no_grad()
def main():
    DEST.mkdir(exist_ok=False);start=time.time();rt=Runtime()
    with np.load(SOURCE,allow_pickle=False) as saved:base={k:saved[k].copy() for k in saved.files}
    source_hash=sha(SOURCE);proposal_hash=sha(PROPOSAL);N=len(base['xyz'])
    source=SHAsset.from_geometry(SOURCE).cuda();d=rt.raw(RAW);cam=camera_hr(d['cam'])
    # Use the actual same prior/rasterizer, not a2D pasted final rendering.
    out=raster(cam,rt.pipe,0,tagged(source.rows(d['pose'],cam,rt.pipe,1,precomputed=True),'asset'),rt.n.bg.new_zeros(3))
    alpha=out['alpha'][0].cpu().numpy();depth_model=(out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-6)).cpu().numpy()
    support=alpha>.1
    yy,xx=np.mgrid[:1280,:1920];use=support&(xx%STRIDE==0)&(yy%STRIDE==0)
    pixels=np.flatnonzero(use);xy=np.column_stack((pixels%1920,pixels//1920))
    z_model=depth_model.ravel()[pixels].astype(np.float64)
    assert np.isfinite(z_model).all() and bool((z_model>0).all())
    view=cam.world_view_transform.cpu().numpy().astype(np.float64)
    full=cam.full_proj_transform.cpu().numpy().astype(np.float64)
    center=cam.camera_center.cpu().numpy().astype(np.float64)
    pose=d['pose'].cpu().numpy().astype(np.float64)
    pose_scale=float(np.linalg.norm(pose[:3,:3],axis=0).mean())
    ndc=np.column_stack(((2*xy[:,0]+1)/1920-1,(2*xy[:,1]+1)/1280-1,np.full(len(xy),.5),np.ones(len(xy))))
    homogeneous=ndc@np.linalg.inv(full);world_on_ray=homogeneous[:,:3]/homogeneous[:,3,None]
    direction=world_on_ray-center
    depth_per_ray=direction@view[:3,2]
    assert bool((depth_per_ray>0).all())
    model=center+direction*(z_model/depth_per_ray)[:,None]
    local=(model-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T
    local=local.astype(np.float32)
    reconstructed=local.astype(np.float64)@pose[:3,:3].T+pose[:3,3]
    reprojection_error=np.linalg.norm(project(reconstructed,full)-xy,axis=1)
    depth_error=np.abs(reconstructed@view[:3,2]+view[3,2]-z_model)
    assert float(reprojection_error.max())<.01
    assert float(depth_error.max())<1e-4
    sigma=(SIGMA_PIXELS*z_model/float(cam.fx)/pose_scale).astype(np.float32)
    distance,nearest=cKDTree(base['xyz']).query(local,k=1)
    # Every retired row is registered by an actual native contribution to
    # the chart reference domain. Entire PVD source survives on disk unchanged.
    W=load_npz(PACKET/'design_W_premultiplied.npz')
    with np.load(PACKET/'operator_samples.npz') as samples:
        assert np.array_equal(samples['pixels'],np.flatnonzero(support))
    contribution=np.asarray(W.sum(0)).ravel();retired=contribution>0
    retained=np.flatnonzero(~retired)
    assert W.shape==(int(support.sum()),N)
    real=np.asarray(Image.open(RGBROOT/'image/image_0'/'0008106.png').convert('RGB'))
    visible=np.repeat(np.repeat(d['a6_visible'].cpu().numpy().astype(bool),4,0),4,1)
    known=visible.ravel()[pixels]
    generated_image=np.asarray(Image.open(PROPOSAL).convert('RGB'))
    with np.load(PACKET/'transform.npz') as tr:affine=tr['highres_to_generated_pixel_center'].copy()
    generated_xy=(np.column_stack((xy,np.ones(len(xy))))@affine.T)[:,:2]
    near=np.floor(generated_xy+.5).astype(int);near=np.clip(near,[0,0],[511,511])
    fg=np.asarray(Image.open(RUN/'inputs/raw106/object_mask_512.png'))>0
    ip=np.asarray(Image.open(RUN/'inputs/raw106/inpaint_mask_512.png'))>0
    generated=(~known)&fg[near[:,1],near[:,0]]&ip[near[:,1],near[:,0]]
    fallback=(~known)&~generated
    rgb=base['rgb'][nearest].copy()
    sampled=np.column_stack([map_coordinates(generated_image[...,c].astype(np.float64)/255,
        [generated_xy[:,1],generated_xy[:,0]],order=1,mode='nearest',prefilter=False) for c in range(3)]).astype(np.float32)
    rgb[generated]=sampled[generated]
    rgb[known]=real.reshape(-1,3)[pixels[known]].astype(np.float32)/255
    M=len(local);result={}
    reserved={'canonical_sh','coefficients','sh','rgb','colour_inferred','geometry_hash','provenance_hash','metadata_json'}
    for key,value in base.items():
        if key in reserved:continue
        if value.ndim and value.shape[0]==N:
            result[key]=np.concatenate((value[retained],value[nearest]),axis=0)
        else:result[key]=value.copy()
    cut=len(retained);chart=slice(cut,None)
    result['xyz'][chart]=local
    result['sigma'][chart]=sigma;result['thickness'][chart]=sigma;result['opacity'][chart]=OPACITY
    toward_camera=(center-reconstructed)@np.linalg.inv(pose[:3,:3]).T
    toward_camera/=np.linalg.norm(toward_camera,axis=1,keepdims=True)
    result['normal'][chart]=toward_camera.astype(np.float32)
    result['normal_valid'][chart]=False;result['normal_confidence'][chart]=0
    result['normal_radius'][chart]=sigma*3;result['knn_spacing'][chart]=(z_model/float(cam.fx)/pose_scale).astype(np.float32)
    result['out_of_box'][chart]=(np.abs(local)>base['box'][None]/2).any(1)
    result['generated'][chart]=True;result['conditioning'][chart]=False;result['conditioning_held'][chart]=False
    for key in ('conditioning_observation_index','conditioning_cloud_row'):result[key][chart]=-1
    result['conditioning_raw_frame'][chart]=RAW
    result['canonical_sh']=np.concatenate((base['canonical_sh'][retained],np.zeros((M,16,3),np.float32)),axis=0)
    result['canonical_sh'][chart,0]=(rgb-.5)/C0
    result['rgb']=np.concatenate((base['rgb'][retained],rgb))
    result['colour_inferred']=np.concatenate((base['colour_inferred'][retained],~known))
    result['representation_kind']=np.concatenate((np.zeros(cut,np.int16),np.ones(M,np.int16)))
    result['resample_source_row']=np.concatenate((retained,nearest)).astype(np.int64)
    result['resample_source_parent']=np.concatenate((base['source_index'][retained],base['source_index'][nearest])).astype(np.int64)
    result['resample_source_distance_local_m']=np.concatenate((np.zeros(cut,np.float32),distance.astype(np.float32)))
    result['resample_reference_pixel_xy']=np.concatenate((np.full((cut,2),-1,np.int32),xy.astype(np.int32)))
    result['resample_reference_raw']=np.concatenate((np.full(cut,-1,np.int16),np.full(M,RAW,np.int16)))
    result['resample_rgb_real']=np.concatenate((np.zeros(cut,bool),known))
    result['resample_rgb_generated']=np.concatenate((np.zeros(cut,bool),generated))
    result['resample_rgb_prior_fallback']=np.concatenate((np.ones(cut,bool),fallback))
    result['geometry_hash']=np.asarray(hash_geometry(result))
    metadata=dict(representation='view-conditioned surface resampling',reference_raw=RAW,reference_split='BUILD',
        source_asset=str(SOURCE),source_asset_sha256=source_hash,source_geometry_hash=source.geometry_hash,
        geometry='New local3D persistent chart from frozen sphere expected camera-z, plus unreferenced unchanged PVD source rows',
        reference_rgb_proposal=str(PROPOSAL),reference_rgb_proposal_sha256=proposal_hash,
        source_rows=N,retired_source_rows=int(retired.sum()),retained_source_rows=cut,chart_rows=M,
        stride=STRIDE,sigma_reference_HR_pixels=SIGMA_PIXELS,opacity=OPACITY,
        sigma_formula='local_sigma=0.6*camera_z_model/fx_HR/actor_pose_uniform_scale',
        source_retirement='Retire exactly source rows with positive native alpha*T contribution in raw106 sphere alpha>0.1 object domain',
        color='BUILD106 true HR RGB for known; same512affine strongv2 only explicit unknown-object; sourceprior fallback outside supplied proposal',
        PVD_weight_updates=0,colour_optimization_updates=0,source_point_positions_modified=False,
        depth_is_prior_prediction_not_truth=True,HELD_RGB_used=False,full360_recovery=False,
        final_renderer='Same native3D joint sorting; no screen overlay',
        limitation='Single-camera conditioned output representation upper-bound; novel viewpoints can expose depth or coverage errors')
    asset=SHAsset(result,metadata).cuda();export=asset.export(DEST/'projective_surface_asset.npz')
    reload=SHAsset.from_geometry(DEST/'projective_surface_asset.npz')
    assert reload.geometry_hash==asset.geometry_hash and len(reload.xyz)==cut+M
    assert np.array_equal(reload.xyz.numpy()[:cut],base['xyz'][retained])
    np.savez_compressed(DEST/'reference_chart.npz',reference_raw=np.int64(RAW),pixels=pixels,pixel_xy=xy,
        local_xyz=local,sigma_local_m=sigma,expected_depth_model=z_model,expected_depth_m=z_model/rt.scale,
        color_rgb=rgb,color_real=known,color_generated=generated,color_prior_fallback=fallback,
        generated_xy=generated_xy,nearest_source_row=nearest,nearest_source_parent=base['source_index'][nearest],
        nearest_distance_local_m=distance,source_native_reference_contribution=contribution,
        source_retired=retired,source_retained=retained,reference_pose=pose,
        reference_world_view=view,reference_full_projection=full,reference_camera_center=center,
        highres_to_generated_pixel_center=affine)
    assert rt.modelunchanged() and source.assert_geometry_unchanged() and asset.assert_geometry_unchanged()
    assert sha(SOURCE)==source_hash and sha(PROPOSAL)==proposal_hash
    receipt=dict(status='PASS',**metadata,export=export,
        geometry_hash=asset.geometry_hash,real_color_rows=int(known.sum()),generated_color_rows=int(generated.sum()),prior_fallback_rows=int(fallback.sum()),
        reprojection_error_HR_pixels=stats(reprojection_error),depth_roundtrip_error_model=stats(depth_error),
        nearest_source_distance_local_m=stats(distance),sigma_local_m=stats(sigma),
        pose_uniform_scale=pose_scale,meter_to_model=rt.scale,seconds=time.time()-start)
    write(DEST/'prepare_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
