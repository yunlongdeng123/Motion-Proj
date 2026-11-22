"""Joint old/new raw-world calibration, extension-only continuous RGB residual.

CPU only. The accepted parent is a locked prior, not a real-photo relabel.
No hard-final image, failed sequential atlas, Gaussian or GPU is read/run.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt

PARENT_SHA='2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def edge_metrics(parent,field,oldmask,newmask,normalize,sigma):
    a=normalize(parent,oldmask,sigma);b=normalize(field,newmask,sigma);raw=[];lf=[]
    for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
        ok=newmask&np.roll(oldmask,(-dy,-dx),(0,1))
        if dy>0:ok[-1]=False
        if dy<0:ok[0]=False
        if dx>0:ok[:,-1]=False
        if dx<0:ok[:,0]=False
        raw.append(abs(field-np.roll(parent,(-dy,-dx),(0,1)))[ok]);lf.append(abs(b-np.roll(a,(-dy,-dx),(0,1)))[ok])
    stat=lambda x:dict(channel_samples=int(x.size),median=float(np.median(x)) if x.size else None,p95=float(np.quantile(x,.95)) if x.size else None,max=float(x.max()) if x.size else None)
    return dict(raw_adjacent_texture_and_colour=stat(np.concatenate(raw).ravel()),
        separate_side_normalized_LF=stat(np.concatenate(lf).ravel()),LF_sigma_texels=sigma)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['r7','r9','geometry','seed','observations','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);started=time.time()
    sys.path.insert(0,str(a.r9/'code'))
    from fuse_old_raw_joint import affine_calibrate,normalized_filter,robust_mean,pair_metrics
    from solve_continuous_shared_field import system,solve
    settings=read(a.r9/'acceptance_contract_v1.json')['joint_solver'];pde=read(a.r9/'continuous_field_contract_v2.json')
    parent_path=a.r9/'joint_shared_continuous_v2/final_atlas.npz';assert sha(parent_path)==PARENT_SHA
    parent=load(parent_path);seed=load(a.seed);g=load(a.geometry)
    H,W=parent['permission_mask'].shape;original=seed['atlas_linear'].copy();base=original.astype(np.float64)
    extension=seed['parent_conditioned_extension'].astype(bool);parent_lock=parent['permission_mask'].astype(bool)
    assert extension.any() and not (extension&parent_lock).any()
    assert original[~extension].tobytes()==parent['atlas_linear'][~extension].tobytes()
    assert seed['real_locks'].tobytes()==parent['real_locks'].tobytes()
    true_seed_path=a.r7/'quilt_seed/atlas_seed.npz';true_seed=load(true_seed_path)
    known=true_seed['same_world_lock'].astype(bool);true_real=true_seed['atlas_seed_linear']
    assert np.array_equal(known,parent['real_locks']) and parent['atlas_linear'][known].tobytes()==true_real[known].tobytes()
    uv=g['uv_global'];scope=g['candidate_mask'].astype(bool) if 'candidate_mask' in g else g['geometry_valid'].astype(bool)
    assert uv.shape==(*scope.shape,2) and g['world_xyz'].shape==(*scope.shape,3)
    integer=np.rint(uv).astype(np.int64);assert np.array_equal(uv[scope],integer[scope])
    assert np.isfinite(g['world_xyz'][scope]).all();iy,ix=integer[...,1],integer[...,0]
    assert ((ix>=0)&(ix<W)&(iy>=0)&(iy<H))[scope].all()
    assert len(np.unique((iy*W+ix)[scope]))==scope.sum()
    global_scope=np.zeros((H,W),bool);global_scope[iy[scope],ix[scope]]=True
    assert not (extension&~global_scope).any()
    source_receipt=read(a.observations/'receipt.json');old_observations=a.r9/'world_observations_v2';old_receipt=read(old_observations/'receipt.json')
    frames=sorted(old_receipt['frames']);assert frames==[86,110]
    old_banks=[load(old_observations/f'{raw:03d}/world_observation.npz') for raw in frames]
    old_rgb=np.stack([x['canonical_linear_RGB'] for x in old_banks]).astype(np.float64)
    old_valid=np.stack([x['valid_mask'] for x in old_banks]);rgb=old_rgb.copy();valid=old_valid.copy()
    weight=np.stack([x['weight'] for x in old_banks]).astype(np.float64);local_rgb=[];local_valid=[];local_weight=[];input_rows=[]
    for i,raw in enumerate(frames):
        path=a.observations/f'{raw:03d}/world_observation.npz';b=load(path)
        assert b['canonical_linear_RGB'].shape==(*scope.shape,3) and b['valid_mask'].shape==scope.shape and b['weight'].shape==scope.shape
        if 'uv_global' in b:assert np.array_equal(b['uv_global'],uv)
        m=b['valid_mask'].astype(bool)&scope;w=np.where(m,b['weight'],0).astype(np.float64)
        assert np.isfinite(b['canonical_linear_RGB'][m]).all() and np.isfinite(w).all() and (w>=0).all()
        # A new geometry query replaces old query samples at the same UV address.
        # Old data outside the new geometry scope and true real RGB anchors remain.
        rgb[i,global_scope]=0;valid[i,global_scope]=False;weight[i,global_scope]=0
        rgb[i,iy[m],ix[m]]=b['canonical_linear_RGB'][m];valid[i,iy[m],ix[m]]=True;weight[i,iy[m],ix[m]]=w[m]
        local_rgb.append(b['canonical_linear_RGB']);local_valid.append(m);local_weight.append(w)
        input_rows.append(dict(raw=raw,original_observation=pin(old_observations/f'{raw:03d}/world_observation.npz'),local_observation=pin(path),
            original_valid_overwritten=int((old_valid[i]&global_scope).sum()),local_valid=int(m.sum()),
            local_valid_new_extension=int((valid[i]&extension).sum()),true_real_anchor_samples=int((valid[i]&known).sum()),
            canonical_RGB_already_contains_original_photo_exposure_gain=True,photo_gain_applied_again=False))
    rgb=np.where(valid[...,None],rgb,0);weight=np.where(valid,weight,0)
    print('JOINT_GLOBAL_AFFINE_AND_SYNCHRONOUS_WORLD_BIAS',flush=True)
    affine,parameters=affine_calibrate(rgb,valid,weight,true_real,known,settings['affine_IRLS_steps'])
    bias=np.zeros_like(affine);base_weight=weight.copy();texel=float(pde['grid_texel_m'])
    for i in range(len(frames)):
        margin=np.clip(distance_transform_edt(valid[i])*texel/.1,0,1);base_weight[i]*=np.sin(np.pi*.5*margin)**2
    tone_history=[]
    for iteration in range(settings['synchronous_world_bias_steps']):
        current=affine+bias;shared,robust_weights=robust_mean(current,base_weight)
        support=valid.any(0)|known;target=np.where(valid.any(0)[...,None],shared,base).copy();target[known]=true_real[known]
        shared_lf=normalized_filter(target,support,settings['shared_tone_sigma_m']/texel);next_bias=[]
        for i in range(len(frames)):
            tone=normalized_filter(current[i],valid[i],settings['shared_tone_sigma_m']/texel)
            coupled=valid[i]&((valid.sum(0)>1)|known);w=np.where(coupled,base_weight[i],0)
            den=gaussian_filter(w,settings['world_bias_sigma_m']/texel)
            num=gaussian_filter((shared_lf-tone)*w[...,None],(settings['world_bias_sigma_m']/texel,settings['world_bias_sigma_m']/texel,0))
            d=np.divide(num,den[...,None],out=np.zeros_like(num),where=den[...,None]>1e-10)
            update=bias[i]+settings['world_bias_relaxation']*d*(den/(den+.02))[...,None]
            next_bias.append(np.where(valid[i,...,None],np.clip(update,-.025,.025),0))
        bias=np.stack(next_bias);tone_history.append(dict(iteration=iteration+1,max_abs=float(abs(bias).max())))
    data=affine+bias;delta=np.zeros_like(base);unknown=extension&~known
    positive=weight[weight>0];assert len(positive)>0;normalizer=float(np.median(positive));raw_weight=weight/normalizer;history=[]
    print('EXTENSION_ONLY_CONTINUOUS_RESIDUAL_DIRICHLET_ZERO',flush=True)
    for iteration in range(pde['IRLS_iterations']):
        residual=np.sqrt(((base[None]+delta[None]-data)**2).mean(-1))
        fit_weights=raw_weight*np.minimum(1.,pde['Huber_delta_linear_RGB']/np.maximum(residual,1e-12))
        Wdata=fit_weights.sum(0);rhs=(fit_weights[...,None]*(data-base[None])).sum(0)
        matrix,pre=system(unknown,Wdata,pde['seed_residual_prior'],pde['laplacian_lambda']);channels=[]
        for c in range(3):
            x,record=solve(matrix,pre,rhs[...,c][unknown],delta[...,c][unknown],pde['CG_relative_tolerance'],pde['CG_max_iterations'])
            delta[...,c][unknown]=x;channels.append(record)
        history.append(dict(iteration=iteration+1,channels=channels,RGB_residual_abs_max=float(abs(delta).max())))
        print(history[-1],flush=True)
    influence,influence_record=solve(matrix,pre,Wdata[unknown],None,pde['CG_relative_tolerance'],pde['CG_max_iterations'])
    tau=np.zeros((H,W),np.float32);tau[unknown]=np.clip(influence,0,1).astype(np.float32)
    final=(base+delta).astype(np.float32);final[~unknown]=original[~unknown]
    assert np.isfinite(final).all() and final.min()>=0 and final.max()<=1,'Do not clip an invalid solution into a false pass'
    assert final[~extension].tobytes()==parent['atlas_linear'][~extension].tobytes() and final[known].tobytes()==parent['atlas_linear'][known].tobytes()
    output={k:v.copy() for k,v in seed.items()};output['atlas_linear']=final
    adopted=unknown&(tau>.001);direct=valid.any(0)&extension
    output['source_kind'][extension]=2;output['source_kind'][adopted]=3
    output['teacher_adopted'][extension]=adopted[extension];output['effective_teacher_blend'][extension]=tau[extension]
    output['teacher_support_count'][extension]=valid.sum(0)[extension].astype(np.uint8)
    output['teacher_directly_observed'][extension]=direct[extension];output['teacher_inferred_field'][extension]=(adopted&~direct)[extension]
    output['confidence'][extension]=np.where(adopted,.35*tau,.25)[extension]
    output['extension_teacher_direct_observation']=direct;output['extension_teacher_inferred_influence']=extension&~direct&(tau>0)
    output['extension_parent_colour_conditioning']=extension.copy();output['extension_teacher_mixed_source']=np.asarray('joint original-raw teacher continuous LF residual + complete similar-photo seed grain; parent conditioning is not a new real observation')
    assert output['real_locks'].tobytes()==parent['real_locks'].tobytes()
    np.savez_compressed(a.output/'final_atlas.npz',**output)
    np.savez_compressed(a.output/'radiometry_state.npz',frames=np.asarray(frames),raw_RGB=rgb.astype(np.float32),affine_RGB=affine.astype(np.float32),
        world_bias=bias.astype(np.float32),corrected_RGB=data.astype(np.float32),valid=valid,raw_weights=weight.astype(np.float32),
        base_weights=base_weight.astype(np.float32),original_old_valid=old_valid,overwritten_geometry_scope=global_scope)
    np.savez_compressed(a.output/'solve_state.npz',RGB_residual=delta.astype(np.float32),data_weights=fit_weights.astype(np.float32),
        conditional_teacher_influence_extension=tau,unknown_extension=unknown,frames=np.asarray(frames))
    local_raw=np.stack(local_rgb).astype(np.float64);local_masks=np.stack(local_valid)
    local_corrected=data[:,iy,ix];old_corrected=np.zeros_like(old_rgb)
    for i,pv in enumerate(parameters):old_corrected[i]=np.where(old_valid[i,...,None],old_rgb[i]*np.asarray(pv['gain'])+np.asarray(pv['bias_linear'])+bias[i],0)
    metrics=dict(old_original_fixed_overlap=pair_metrics(old_rgb,old_corrected,old_valid,settings['shared_tone_sigma_m']/texel),
        local_new_geometry_fixed_overlap=pair_metrics(local_raw,local_corrected,local_masks,settings['shared_tone_sigma_m']/texel),
        seed_parent_boundary=edge_metrics(parent['atlas_linear'],original,parent_lock,extension,normalized_filter,settings['shared_tone_sigma_m']/texel),
        final_parent_boundary=edge_metrics(parent['atlas_linear'],final,parent_lock,extension,normalized_filter,settings['shared_tone_sigma_m']/texel))
    grain={}
    for metres in [.02,.04]:
        s0=normalized_filter(original,extension,metres/texel);s1=normalized_filter(final,extension,metres/texel)
        h0=(original-s0)[extension].ravel();h1=(final-s1)[extension].ravel();corr=float(np.corrcoef(h0,h1)[0,1])
        grain[str(metres)]=dict(extension_pixels=int(extension.sum()),correlation=corr,RMS_ratio=float(np.sqrt(np.mean(h1*h1))/max(np.sqrt(np.mean(h0*h0)),1e-12)),
            normalized_within_same_extension_mask=True)
    metrics['extension_photo_grain']=grain;write(a.output/'metrics.json',metrics)
    record=dict(schema='R10.local_extension_joint_continuous_appearance.v1',complete=True,CPU_only=True,
        parent=pin(parent_path),seed=pin(a.seed),geometry=pin(a.geometry),new_observation_receipt=pin(a.observations/'receipt.json'),
        original_observation_receipt=pin(old_observations/'receipt.json'),inputs=input_rows,frames=frames,
        reused_joint_solver_source=pin(a.r9/'code/fuse_old_raw_joint.py'),reused_PDE_source=pin(a.r9/'code/solve_continuous_shared_field.py'),
        affine_settings=settings,PDE_settings=pde,affine_parameters=parameters,synchronous_history=tone_history,IRLS_CG_history=history,influence_CG=influence_record,
        true_same_world_real_anchor_source=pin(true_seed_path),true_real_lock_count=int(known.sum()),parent_inferred_pixels_not_relabelled_real=True,
        parent_RGB_and_real_locks_byteexact=True,extension_only_write_pixels=int(extension.sum()),
        teacher_direct_extension_pixels=int(direct.sum()),teacher_inferred_extension_pixels=int((adopted&~direct).sum()),
        source2_weak_teacher_tail_retained_pixels=int((extension&~adopted&(tau>0)).sum()),conditional_influence_is_not_calibrated_probability=True,
        no_pointwise_support_multiplication_of_final_RGB=True,parent_boundary_residual_Dirichlet_zero=True,
        harmonic_seed_coordination_applied=False,HF_texture_not_blurred_to_pass_seam=True,
        accepted_parent_field_read=True,old_hard_final_images_or_failed_v1_atlas_read=False,new_geometry_acceptance=False,
        final_atlas=pin(a.output/'final_atlas.npz'),radiometry_state=pin(a.output/'radiometry_state.npz'),solve_state=pin(a.output/'solve_state.npz'),metrics=pin(a.output/'metrics.json'),
        source_photo_seed_lineage=pin(a.seed.parent/'new_texture_lineage.npz'),source_photo_seed_trace=pin(a.seed.parent/'quilt_trace.json'),
        Gaussian_fit_authorized=False,numeric_acceptance='PENDING independent fixed-domain LF/seam/grain checks',visual_acceptance='PENDING independent and root review',
        nativeF=0,nativeB=0,new_semantic_calls=0,new_diffusion_calls=0,Gaussian_updates=0,seconds=time.time()-started)
    assert sha(parent_path)==PARENT_SHA
    write(a.output/'receipt.json',record);print(dict(complete=True,final=record['final_atlas'],metrics=metrics,seconds=record['seconds']))


if __name__=='__main__':main()
