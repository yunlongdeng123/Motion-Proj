"""Joint radiometric correction and robust world-surface fusion, CPU only."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt
from scipy.optimize import minimize
from project_old_raw_candidates import load,sha,write


def normalized_filter(x,mask,sigma):
    w=gaussian_filter(mask.astype(np.float64),sigma)
    value=gaussian_filter(np.where(mask[...,None],x,0).astype(np.float64),(sigma,sigma,0))
    return np.divide(value,w[...,None],out=np.zeros_like(value),where=w[...,None]>1e-10)


def huber_weight(r,delta):return np.minimum(1.,delta/np.maximum(np.abs(r),1e-12))


def affine_calibrate(rgb,valid,weights,real,known,steps):
    n=len(rgb);theta=np.tile([1.,0.],(3,n))
    pairs=[(i,j,valid[i]&valid[j]) for i in range(n) for j in range(i+1,n)]
    anchor_records=[]
    for c in range(3):
        t=theta[c].copy()
        for _ in range(steps):
            diagonal=np.tile([1e-4,.01],n);Q=np.diag(diagonal);b=np.tile([1e-4,0.],n)
            for i in range(n):
                m=valid[i]&known
                if not m.any():continue
                x=rgb[i,...,c][m];y=real[...,c][m]
                a=np.zeros((len(x),2*n));a[:,2*i]=x;a[:,2*i+1]=1
                w=weights[i][m]*huber_weight(a@t-y,.01);w*=2/max(w.sum(),1e-12)
                Q+=a.T@(w[:,None]*a);b+=a.T@(w*y)
            for i,j,m in pairs:
                if not m.any():continue
                a=np.zeros((int(m.sum()),2*n));a[:,2*i]=rgb[i,...,c][m];a[:,2*i+1]=1
                a[:,2*j]=-rgb[j,...,c][m];a[:,2*j+1]=-1
                w=np.sqrt(weights[i][m]*weights[j][m])*huber_weight(a@t,.01);w/=max(w.sum(),1e-12)
                Q+=a.T@(w[:,None]*a)
            objective=lambda x: (float(.5*x@Q@x-b@x),Q@x-b)
            solved=minimize(objective,t,jac=True,method='L-BFGS-B',bounds=[(.8,1.25),(-.025,.025)]*n,
                options={'ftol':1e-14,'gtol':1e-10,'maxiter':300})
            assert np.isfinite(solved.x).all();t=solved.x
        theta[c]=t
    corrected=rgb.copy().astype(np.float64)
    parameters=[]
    for i in range(n):
        gain=theta[:,2*i];bias=theta[:,2*i+1]
        corrected[i]=np.where(valid[i,...,None],rgb[i]*gain+bias,0)
        parameters.append(dict(gain=gain.tolist(),bias_linear=bias.tolist(),real_anchor_texels=int((known&valid[i]).sum())))
    return corrected,parameters


def robust_mean(rgb,weights,passes=3):
    w=weights.copy();mean=np.zeros_like(rgb[0])
    for _ in range(passes):
        den=w.sum(0);mean=np.divide((rgb*w[...,None]).sum(0),den[...,None],out=np.zeros_like(mean),where=den[...,None]>1e-12)
        residual=np.sqrt(((rgb-mean[None])**2).mean(-1))
        w=weights*huber_weight(residual,.01)
    return mean,w


def pair_metrics(before,after,valid,sigma):
    rows=[];before_lf=[normalized_filter(x,m,sigma) for x,m in zip(before,valid)]
    after_lf=[normalized_filter(x,m,sigma) for x,m in zip(after,valid)]
    for i in range(len(before)):
        for j in range(i+1,len(before)):
            m=valid[i]&valid[j]
            if not m.any():continue
            a=float(abs(before[i]-before[j])[m].mean());b=float(abs(after[i]-after[j])[m].mean())
            l0=float(abs(before_lf[i]-before_lf[j])[m].mean());l1=float(abs(after_lf[i]-after_lf[j])[m].mean())
            rows.append(dict(source_indices=[i,j],fixed_pair_texels=int(m.sum()),fixed_channel_samples=int(m.sum())*3,
                full_RGB_before_MAE=a,full_RGB_after_MAE=b,full_RGB_reduction=1-b/max(a,1e-12),
                LF_before_MAE=l0,LF_after_MAE=l1,LF_reduction=1-l1/max(l0,1e-12),
                LF_absolute_or_relative_pass=bool(l1<=.005 or l1<=.8*l0)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--r7',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists();args.output.mkdir(parents=True)
    started=time.time();contract=json.loads(args.contract.read_text(encoding='utf-8-sig'));settings=contract['joint_solver']
    source_receipt=json.loads((args.observations/'receipt.json').read_text(encoding='utf-8-sig'))
    frames=sorted(source_receipt['frames']);assert frames==contract['source_views']
    banks=[load(args.observations/f'{raw:03d}/world_observation.npz') for raw in frames]
    rgb=np.stack([b['canonical_linear_RGB'] for b in banks]).astype(np.float64)
    valid=np.stack([b['valid_mask'] for b in banks]);weight=np.stack([b['weight'] for b in banks]).astype(np.float64)
    seed_path=args.r7/'quilt_seed/atlas_seed.npz';seed=load(seed_path);base=seed['atlas_seed_linear'];known=seed['same_world_lock'].astype(bool)
    permission=seed['write_domain'].astype(bool);kind=seed['source_label'];texel=.02
    assert all((v&~permission).sum()==0 for v in valid)
    corrected,parameters=affine_calibrate(rgb,valid,weight,base,known,settings['affine_IRLS_steps'])
    bias=np.zeros_like(corrected);base_weight=weight.copy()
    for i in range(len(valid)):
        margin=np.clip(distance_transform_edt(valid[i])*texel/.1,0,1)
        base_weight[i]*=np.sin(np.pi*.5*margin)**2
    shared=np.zeros_like(base,dtype=np.float64);history=[]
    for iteration in range(settings['synchronous_world_bias_steps']):
        current=corrected+bias;shared,robust_weight=robust_mean(current,base_weight)
        support=valid.any(0)|known;target=np.where(valid.any(0)[...,None],shared,base).copy();target[known]=base[known]
        shared_lf=normalized_filter(target,support,settings['shared_tone_sigma_m']/texel)
        next_bias=[]
        for i in range(len(valid)):
            tone=normalized_filter(current[i],valid[i],settings['shared_tone_sigma_m']/texel)
            coupled=valid[i]&((valid.sum(0)>1)|known)
            w=np.where(coupled,base_weight[i],0)
            denominator=gaussian_filter(w,settings['world_bias_sigma_m']/texel)
            numerator=gaussian_filter((shared_lf-tone)*w[...,None],(settings['world_bias_sigma_m']/texel,settings['world_bias_sigma_m']/texel,0))
            delta=np.divide(numerator,denominator[...,None],out=np.zeros_like(numerator),where=denominator[...,None]>1e-10)
            influence=denominator/(denominator+.02)
            update=bias[i]+settings['world_bias_relaxation']*delta*influence[...,None]
            next_bias.append(np.where(valid[i,...,None],np.clip(update,-.025,.025),0))
        bias=np.stack(next_bias)
        history.append(dict(iteration=iteration+1,world_bias_abs_max=float(abs(bias).max())))
    current=corrected+bias;shared,robust_weight=robust_mean(current,base_weight)
    union=valid.any(0);distance=np.clip(distance_transform_edt(union)*texel/settings['teacher_to_seed_boundary_feather_m'],0,1)
    alpha=distance*distance*(3-2*distance);alpha[known|~permission]=0
    final=((1-alpha[...,None])*base+alpha[...,None]*shared).astype(np.float32)
    final[known]=base[known];final[~permission]=base[~permission]
    assert np.isfinite(final).all();final=np.clip(final,0,1)
    adopted=(alpha>1e-8)&permission&~known;source=kind.copy();source[adopted]=3
    np.savez_compressed(args.output/'final_atlas.npz',atlas_linear=final,source_kind=source,seed_source_kind=kind,
        confidence=np.where(adopted,.35,seed.get('confidence',np.ones_like(alpha))).astype(np.float32),
        real_locks=known,write_domain=permission,permission_mask=permission,teacher_adopted=adopted,
        effective_teacher_blend=alpha.astype(np.float32),teacher_support_count=valid.sum(0).astype(np.uint8),
        teacher_mixed_source=np.asarray('full RGB of jointly corrected original 2D raw teachers, with photo-seed boundary blend; not hard final composites'))
    np.savez_compressed(args.output/'solve_state.npz',raw_RGB=rgb.astype(np.float32),affine_RGB=corrected.astype(np.float32),
        corrected_RGB=current.astype(np.float32),world_bias=bias.astype(np.float32),valid=valid,base_weights=base_weight.astype(np.float32),
        robust_weights=robust_weight.astype(np.float32),teacher_alpha=alpha.astype(np.float32),frames=np.asarray(frames))
    pairs=pair_metrics(rgb,current,valid,settings['shared_tone_sigma_m']/texel)
    receipt=dict(schema='R9.original_raw_joint_shared_appearance.v1',complete=True,CPU_only=True,
        contract_sha256=sha(args.contract),source_observation_receipt_sha256=sha(args.observations/'receipt.json'),
        source_observation_NPZ_SHA=[sha(args.observations/f'{raw:03d}/world_observation.npz') for raw in frames],
        seed_sha256=sha(seed_path),final_atlas_sha256=sha(args.output/'final_atlas.npz'),solve_state_sha256=sha(args.output/'solve_state.npz'),
        frames=frames,affine_parameters=parameters,synchronous_solver_history=history,fixed_pair_metrics=pairs,
        source1_locked_texels=int(known.sum()),teacher_support_texels=int(union.sum()),teacher_adopted_texels=int(adopted.sum()),
        source2_fallback_texels=int((source==2).sum()),real_locks_RGB_bitexact=bool(np.array_equal(final[known],base[known])),
        outside_permission_RGB_bitexact=bool(np.array_equal(final[~permission],base[~permission])),
        teacher_to_teacher_locks=False,first_writer_policy=False,raw_native_or_hard_final_used_as_RGB=False,
        source_order='canonical frame order; all view corrections updated synchronously from the same shared estimate',
        new_diffusion_calls=0,nativeF=0,nativeB=0,optimizer_updates=0,
        numeric_acceptance='PENDING independent full-surface QA',visual_acceptance='PENDING independent and root review',seconds=time.time()-started)
    write(args.output/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
