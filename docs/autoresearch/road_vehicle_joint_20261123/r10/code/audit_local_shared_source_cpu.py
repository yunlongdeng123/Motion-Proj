"""Independent locked-parent/real, fixed-domain band, raw-sampling and joint-source QA."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats

FIELD='eb86cb6d61b711366d8c30052fd02b823433a6e940b80d6f4e0463685e38a5dd'


def nf(x,m,sigma):
    den=gaussian_filter(m.astype(np.float64),sigma)
    num=gaussian_filter(np.where(m[...,None],x,0).astype(np.float64),(sigma,sigma,0))
    return np.divide(num,den[...,None],out=np.zeros_like(num),where=den[...,None]>1e-10)


def sample(image,uv):
    H,W=image.shape[:2];x,y=uv[...,0],uv[...,1];u=np.floor(np.nan_to_num(x)).astype(int);v=np.floor(np.nan_to_num(y)).astype(int)
    good=np.isfinite(uv).all(-1)&(u>=0)&(v>=0)&(u+1<W)&(v+1<H);u=np.clip(u,0,W-2);v=np.clip(v,0,H-2);dx=x-u;dy=y-v
    out=((1-dx)*(1-dy))[...,None]*image[v,u]+(dx*(1-dy))[...,None]*image[v,u+1]+((1-dx)*dy)[...,None]*image[v+1,u]+(dx*dy)[...,None]*image[v+1,u+1]
    return out,good


def all4(m,uv):
    H,W=m.shape;u=np.floor(np.nan_to_num(uv[...,0])).astype(int);v=np.floor(np.nan_to_num(uv[...,1])).astype(int)
    good=np.isfinite(uv).all(-1)&(u>=0)&(v>=0)&(u+1<W)&(v+1<H);u=np.clip(u,0,W-2);v=np.clip(v,0,H-2)
    return good&m[v,u]&m[v,u+1]&m[v+1,u]&m[v+1,u+1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();C,out=a.campaign_root,a.output
    out.mkdir(parents=True,exist_ok=False);started=time.time();R=C/'road_underbody_local_20261123_r10';R9=C/'old_raw_teacher_joint_20261123_r9';R7=C/'road_shared_appearance_20261122_r7'
    fpath=R/'local_shared_continuous_v1/final_atlas.npz';recpath=fpath.parent/'receipt.json';rec=read(recpath);assert pin(fpath)['sha256']==FIELD
    parentpath=Path(rec['parent']['path']);seedpath=Path(rec['seed']['path']);geopath=Path(rec['geometry']['path'])
    f,parent,seed,g=map(load,[fpath,parentpath,seedpath,geopath]);state=load(fpath.parent/'radiometry_state.npz');solve=load(fpath.parent/'solve_state.npz')
    ext=seed['parent_conditioned_extension'].astype(bool);known=f['real_locks'].astype(bool);perm=f['permission_mask'].astype(bool);shape=perm.shape
    contract=dict(schema='R10.independent_shared_input_contract.v1',complete=True,written_before_independent_metric_recomputation=True,
        field_sha256=FIELD,geometry_sha256=pin(geopath)['sha256'],scope='Only shared RGB input to local zero-step control, not final joint scene quality',
        invariant_hard_gates=['SHA pins','exact parent RGB outside declared12328 extension','real6407 mask and RGB exact','finite in-range RGB','geometry/permission/source kind correct','direct Brown raw source RGB replay, no hard-final'],
        local_fixed_pair='all local observation overlap, no robust/influence/permission denominator reduction',
        frequency='LF normalized sigma6texel=.12m; HF separate normalized2cm/4cm correlation, amplitude and delta; no blur or colour scaling to chase metrics',
        seam='LF boundary .005004 is descriptive, not a preregistered .005 hard rejection; do not tune solver to the number',
        joint='both original raw views jointly affine-corrected and world-bias updated synchronously; only accepted parent/real held fixed, no first-writer teacher lock',
        source_order_affine_check='reverse source axis, map back, compare actual corrected values; bounded float tolerance1e-5 linear RGB',
        no_new_GPU=True)
    write(out/'acceptance_contract.json',contract)
    assert int(ext.sum())==12328 and not (ext&parent['permission_mask']).any()
    assert np.array_equal(ext,perm&~parent['permission_mask'])
    assert f['atlas_linear'][~ext].tobytes()==parent['atlas_linear'][~ext].tobytes()
    assert seed['atlas_linear'][~ext].tobytes()==parent['atlas_linear'][~ext].tobytes()
    assert known.tobytes()==parent['real_locks'].tobytes() and int(known.sum())==6407
    true=load(R7/'quilt_seed/atlas_seed.npz');assert np.array_equal(known,true['same_world_lock'])
    assert f['atlas_linear'][known].tobytes()==true['atlas_seed_linear'][known].tobytes()
    assert np.isfinite(f['atlas_linear']).all() and f['atlas_linear'].min()>=0 and f['atlas_linear'].max()<=1
    assert not (known&ext).any()
    for key in ['source_kind','seed_source_kind','teacher_adopted','effective_teacher_blend','teacher_support_count']:
        assert f[key][~ext].tobytes()==parent[key][~ext].tobytes()
    yy,xx=g['uv_global'][...,1].astype(int),g['uv_global'][...,0].astype(int);scope=g['candidate_mask'].astype(bool)
    global_scope=np.zeros(shape,bool);global_scope[yy[scope],xx[scope]]=True;assert not (ext&~global_scope).any()
    assert np.array_equal(state['overwritten_geometry_scope'],global_scope)
    frames=state['frames'].tolist();assert frames==[86,110]
    before=[];valids=[];source_replays=[]
    import cv2
    for i,raw in enumerate(frames):
        localpath=R/f'local_world_observations_v1/{raw:03d}/world_observation.npz';b=load(localpath);old=load(R9/f'world_observations_v2/{raw:03d}/world_observation.npz')
        expected_raw=old['canonical_linear_RGB'].copy();expected_valid=old['valid_mask'].copy();expected_weight=old['weight'].copy()
        expected_raw[global_scope]=0;expected_valid[global_scope]=False;expected_weight[global_scope]=0
        m=b['valid_mask']&scope;expected_raw[yy[m],xx[m]]=b['canonical_linear_RGB'][m];expected_valid[yy[m],xx[m]]=True;expected_weight[yy[m],xx[m]]=b['weight'][m]
        expected_raw[~expected_valid]=0;expected_weight[~expected_valid]=0
        assert expected_raw.tobytes()==state['raw_RGB'][i].tobytes() and expected_valid.tobytes()==state['valid'][i].tobytes()
        assert expected_weight.tobytes()==state['raw_weights'][i].tobytes()
        view=load(R7/f'projection_cache_v1/{raw:03d}/view_cache.npz');E=view['world_to_camera'];intr=view['Brown_intrinsics'];fx,fy,cx,cy=intr[:4]
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],np.float64);rv,_=cv2.Rodrigues(E[:3,:3]);uv,_=cv2.projectPoints(g['world_xyz'].reshape(-1,3),rv,E[:3,3],K,intr[4:]);brown=uv.reshape(*scope.shape,2)*np.array([1152/1920,768/1280])-.5
        source=read(R/f'local_world_observations_v1/{raw:03d}/receipt.json');rawpath=Path(source['raw_model']['path']);assert rawpath.name=='raw_model.png' and pin(rawpath)['sha256']==source['raw_model']['sha256']
        image=np.asarray(Image.open(rawpath).convert('RGB'),np.float64)/255;linear=np.where(image<=.04045,image/12.92,((image+.055)/1.055)**2.4)
        sampled,bounds=sample(linear,brown);sampled*=source['exposure_gain'];error=float(abs(sampled[m]-b['canonical_linear_RGB'][m]).max());assert error<2e-6
        uv_error=float(abs(brown[m]-b['model_Brown_xy'][m]).max());assert uv_error<.0002
        sem=np.asarray(Image.open(R9/f'raw_road_semantic_v2/{raw:03d}/road_eroded_model_Brown.png'))>0
        assert np.array_equal(m,b['pre_semantic_geometric_candidate']&all4(sem,brown))
        before.append(b['canonical_linear_RGB'].astype(float));valids.append(m)
        source_replays.append(dict(raw=raw,source=pin(rawpath),local_observation=pin(localpath),valid_texels=int(m.sum()),RGB_max_abs_linear=error,OpenCV_Brown_model_UV_max_abs_px=uv_error,original_raw_only=True))
    before=np.asarray(before);valids=np.asarray(valids);after=state['corrected_RGB'][:,yy,xx].astype(float);pair=valids[0]&valids[1]
    assert int(pair.sum())==24237
    band={}
    for name,sigma in [('full_RGB',None),('LF_sigma0p12m',6.)]:
        A=before if sigma is None else np.array([nf(x,m,sigma) for x,m in zip(before,valids)])
        B=after if sigma is None else np.array([nf(x,m,sigma) for x,m in zip(after,valids)])
        v0=float(abs(A[0]-A[1])[pair].mean());v1=float(abs(B[0]-B[1])[pair].mean());band[name]=dict(before_MAE=v0,after_MAE=v1,reduction=1-v1/v0,pair_texels=int(pair.sum()),channel_samples=int(pair.sum())*3)
    assert band['full_RGB']['reduction']>=.2 and band['LF_sigma0p12m']['reduction']>=.2
    def boundary(image):
        p=nf(parent['atlas_linear'],parent['permission_mask'],6.);q=nf(image,ext,6.);lf=[];full=[]
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
            m=ext&np.roll(parent['permission_mask'],(-dy,-dx),(0,1))
            if dy>0:m[-1]=False
            if dy<0:m[0]=False
            if dx>0:m[:,-1]=False
            if dx<0:m[:,0]=False
            lf.append(abs(q-np.roll(p,(-dy,-dx),(0,1)))[m]);full.append(abs(image-np.roll(parent['atlas_linear'],(-dy,-dx),(0,1)))[m])
        return dict(LF=stats(np.concatenate(lf).ravel()),full_texture_plus_colour=stats(np.concatenate(full).ravel()))
    boundaries=dict(seed=boundary(seed['atlas_linear']),final=boundary(f['atlas_linear']))
    grain={}
    for metres in [.02,.04]:
        h0=(seed['atlas_linear']-nf(seed['atlas_linear'],ext,metres/.02))[ext].ravel()
        h1=(f['atlas_linear']-nf(f['atlas_linear'],ext,metres/.02))[ext].ravel();rms0=float(np.sqrt(np.mean(h0*h0)));rms1=float(np.sqrt(np.mean(h1*h1)))
        grain[str(metres)]=dict(texels=int(ext.sum()),HF_correlation=float(np.corrcoef(h0,h1)[0,1]),HF_RMS_ratio=rms1/rms0,HF_delta_RMS_over_seed_RMS=float(np.sqrt(np.mean((h1-h0)**2))/rms0))
    final_replay=(seed['atlas_linear'].astype(float)+solve['RGB_residual'].astype(float)).astype(np.float32);final_replay[~ext]=seed['atlas_linear'][~ext]
    residual_replay_error=float(abs(final_replay-f['atlas_linear']).max());assert residual_replay_error<5e-7
    corrected_replay=state['affine_RGB'].astype(float)+state['world_bias'].astype(float);correction_error=float(abs(corrected_replay-state['corrected_RGB']).max());assert correction_error<5e-7
    sys.path.insert(0,str(R9/'code'));from fuse_old_raw_joint import affine_calibrate
    real=true['atlas_seed_linear'];rgb=state['raw_RGB'].astype(float);weights=state['raw_weights'].astype(float);vv=state['valid']
    forward,params=affine_calibrate(rgb,vv,weights,real,known,8)
    reverse,rparams=affine_calibrate(rgb[::-1],vv[::-1],weights[::-1],real,known,8)
    order_error=float(abs(forward-reverse[::-1]).max());affine_error=float(abs(forward-state['affine_RGB']).max());assert order_error<1e-5 and affine_error<5e-7
    direct=ext&vv.any(0);adopt=ext&f['teacher_adopted'];tail=ext&~adopt&(f['effective_teacher_blend']>0)
    assert np.array_equal(direct,f['extension_teacher_direct_observation'])
    viewpath=R/'local_shared_surface_review_v1/receipt.json';view=read(viewpath);assert view['final_atlas_sha256']==FIELD and view['geometry_sha256']==pin(geopath)['sha256']
    viewchecks=[]
    for row in view['views']:
        raw=row['raw'];pr=load(R/f'local_shared_surface_review_v1/{raw:03d}/surface_review.npz');shown=pr['shown'];assert not pr['unavailable_material'].any()
        image=np.asarray(Image.open(R/f'local_shared_surface_review_v1/{raw:03d}/direct_surface_material.png'))
        assert not (shown&(image.max(-1)<40)).any()
        tri=pr['triangle_index'];N=g['cell_asset_mask'].size;expected=pr['projection_valid']&g['cell_asset_mask'].ravel()[np.maximum(tri,0)%N]
        assert np.array_equal(pr['projection_valid'],expected)
        viewchecks.append(dict(raw=raw,material_pixels=int(shown.sum()),unavailable_material=0,near_black_RGB_pixels_in_material=0,exact_cell_asset_triangles=True))
    numeric=dict(schema='R10.independent_local_shared_source_numeric_gate.v1',complete=True,decision='PASS',final_atlas_sha256=FIELD,
        geometry_sha256=pin(geopath)['sha256'],scope='Shared colour input for local zero-step probe only',parent_SHA256=pin(parentpath)['sha256'],
        parent_RGB_outside_extension_byteexact=True,extension_texels=int(ext.sum()),real6407_mask_and_RGB_byteexact=True,finite_in_range_RGB=True,
        current_new_geometry_RGB_sampling_replay=source_replays,fixed_domain_frequency_metrics=band,parent_extension_boundary=boundaries,
        boundary_LF_p95_is_diagnostic_not_preregistered_0p005_hard_gate=True,extension_full_texture_HF=grain,
        joint_radiometry=dict(first_teacher_locked=False,both_views_affine_joint_optimized=True,synchronous_world_bias_steps=8,
            fresh_affine_replay_max_abs=affine_error,reversed_affine_source_order_max_abs=order_error,
            source_order_check_does_not_claim_full_PDE_reexecution=True,affine_bias_sum_replay_max_abs=correction_error,
            affine_parameters=params,conditional_influence_not_probability=True),
        seed_plus_continuous_delta_replay_max_abs=residual_replay_error,pointwise_alpha_multiplication_not_used=True,
        extension_teacher_direct_observed_texels=int(direct.sum()),extension_teacher_adopted_texels=int(adopt.sum()),
        extension_source2_weak_teacher_tail_texels=int(tail.sum()),similar_photo_not_same_world_or_dense_teacher_GT=True,
        five_view_material_domain_checks=viewchecks,source_receipt=pin(recpath),surface_receipt=pin(viewpath),
        full_joint_image_quality_acceptance=False,native_Gaussian_evaluation_performed=False,new_GPU_forward=0,new_GPU_backward=0,Gaussian_updates=0,seconds=time.time()-started)
    write(out/'numeric_gate.json',numeric)
    visual=dict(schema='R10.independent_local_shared_source_visual_gate.v1',complete=True,decision='PASS',final_atlas_sha256=FIELD,
        geometry_sha256=pin(geopath)['sha256'],scope='Exact cell_asset_mask material input only, not whole scene or native joint restoration',reviewer='native_clean_photo_regions',
        frames=[86,110,134,179,198],original_resolution=[1152,768],root_concurred=False,
        observations='Inspected all five original-size direct material PNGs and photo-context illustrations. No obvious interior near-black rim, broad LF colour block, or lattice stripe. Grain remains visible. Two upper dark cutouts in material-only view are outside the exact asset support and have zero unavailable-material pixels within the support; photo context is not target road RGB.',
        frame_images=[dict(raw=r,material=pin(R/f'local_shared_surface_review_v1/{r:03d}/direct_surface_material.png'),context=pin(R/f'local_shared_surface_review_v1/{r:03d}/photo_context_surface_illustration.png')) for r in [86,110,134,179,198]],
        contactsheet=pin(R/'local_shared_surface_review_v1/five_view_direct_surface_context.png'),numeric_gate=pin(out/'numeric_gate.json'),
        source_is_approved_for_next_zero_step_comparison=True,actual_native_occlusion_and_final_joint_quality='SEPARATE_PENDING',new_GPU_forward=0)
    write(out/'visual_gate.json',visual)
    print(json.dumps(dict(complete=True,numeric=pin(out/'numeric_gate.json'),visual=pin(out/'visual_gate.json'),metrics=band,grain=grain)),flush=True)


if __name__=='__main__':main()
