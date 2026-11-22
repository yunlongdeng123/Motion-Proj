"""A continuous world RGB residual solves sparse-view support without RGB stamps."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg,LinearOperator
from project_old_raw_candidates import load,sha,write


def system(unknown,data_weight,mu,lam):
    h,w=unknown.shape;idx=np.full((h,w),-1,np.int64);idx[unknown]=np.arange(unknown.sum())
    degree=np.full((h,w),4.,np.float64);degree[0]-=1;degree[-1]-=1;degree[:,0]-=1;degree[:,-1]-=1
    row=[];col=[]
    for a,b in [(idx[:-1],idx[1:]),(idx[:,:-1],idx[:,1:])]:
        good=(a>=0)&(b>=0);i=a[good];j=b[good]
        row.extend([i,j]);col.extend([j,i])
    rr=np.concatenate(row);cc=np.concatenate(col)
    diagonal=(mu+data_weight+lam*degree)[unknown]
    matrix=diags(diagonal,format='csr')+coo_matrix((np.full(len(rr),-lam),(rr,cc)),shape=(int(unknown.sum()),)*2).tocsr()
    pre=LinearOperator(matrix.shape,matvec=lambda x:x/diagonal,dtype=np.float64)
    return matrix,pre


def solve(matrix,pre,b,x,rtol,maxiter):
    count=[0]
    def tick(_):count[0]+=1
    value,info=cg(matrix,b,x0=x,M=pre,rtol=rtol,atol=1e-10,maxiter=maxiter,callback=tick)
    residual=float(np.linalg.norm(matrix@value-b)/max(np.linalg.norm(b),1e-12))
    assert info==0 and np.isfinite(value).all(),dict(CG_info=info,relative_residual=residual)
    return value,dict(CG_iterations=count[0],relative_residual=residual)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--r7',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--radiometry',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True);started=time.time()
    contract=json.loads(args.contract.read_text(encoding='utf-8-sig'));upstream=json.loads((args.radiometry/'receipt.json').read_text())
    seed_path=args.r7/'quilt_seed/atlas_seed.npz';seed=load(seed_path);state_path=args.radiometry/'solve_state.npz';state=load(state_path)
    base=seed['atlas_seed_linear'].astype(np.float64);known=seed['same_world_lock'].astype(bool);permission=seed['write_domain'].astype(bool)
    unknown=permission&~known;data=state['corrected_RGB'].astype(np.float64);valid=state['valid'].astype(bool);frames=state['frames']
    assert sha(seed_path)==upstream['seed_sha256'] and sha(state_path)==upstream['solve_state_sha256']
    raw_weights=np.stack([load(args.observations/f'{raw:03d}/world_observation.npz')['weight'] for raw in frames]).astype(np.float64)
    scale=float(np.median(raw_weights[raw_weights>0]));raw_weights/=scale
    delta=np.zeros_like(base);history=[];weights=raw_weights
    for iteration in range(contract['IRLS_iterations']):
        residual=np.sqrt(((base[None]+delta[None]-data)**2).mean(-1))
        weights=raw_weights*np.minimum(1.,contract['Huber_delta_linear_RGB']/np.maximum(residual,1e-12))
        W=weights.sum(0);rhs=(weights[...,None]*(data-base[None])).sum(0)
        matrix,pre=system(unknown,W,contract['seed_residual_prior'],contract['laplacian_lambda'])
        records=[]
        for channel in range(3):
            x,record=solve(matrix,pre,rhs[...,channel][unknown],delta[...,channel][unknown],contract['CG_relative_tolerance'],contract['CG_max_iterations'])
            delta[...,channel][unknown]=x;records.append(record)
        history.append(dict(iteration=iteration+1,channels=records,RGB_residual_abs_max=float(abs(delta).max())))
        print(json.dumps(history[-1]),flush=True)
    influence,record=solve(matrix,pre,W[unknown],None,contract['CG_relative_tolerance'],contract['CG_max_iterations'])
    teacher_influence=np.zeros_like(W);teacher_influence[unknown]=np.clip(influence,0,1)
    final=(base+delta).astype(np.float32);final[known]=seed['atlas_seed_linear'][known];final[~permission]=seed['atlas_seed_linear'][~permission]
    assert np.isfinite(final).all() and final.min()>=0 and final.max()<=1,'Do not clip an invalid colour solution into a false PASS'
    source=seed['source_label'].copy();adopted=unknown&(teacher_influence>.001);source[adopted]=3
    confidence=np.where(source==1,1.,np.where(source==2,.5,np.where(source==3,.35*teacher_influence,0))).astype(np.float32)
    union=valid.any(0)
    np.savez_compressed(args.output/'final_atlas.npz',atlas_linear=final,source_kind=source,seed_source_kind=seed['source_label'],
        real_locks=known,write_domain=permission,permission_mask=permission,confidence=confidence,teacher_adopted=adopted,
        effective_teacher_blend=teacher_influence.astype(np.float32),teacher_support_count=valid.sum(0).astype(np.uint8),
        teacher_directly_observed=union,teacher_inferred_field=adopted&~union,
        teacher_mixed_source=np.asarray('continuous variational teacher colour residual with complete similar-photo gradient/detail reference; conditional influence is not calibrated confidence'))
    np.savez_compressed(args.output/'solve_state.npz',RGB_residual=delta.astype(np.float32),data_weights=weights.astype(np.float32),
        conditional_teacher_influence=teacher_influence.astype(np.float32),valid=valid,frames=frames)
    receipt=dict(schema='R9.continuous_shared_appearance.v2',complete=True,CPU_only=True,
        contract_sha256=sha(args.contract),seed_sha256=sha(seed_path),radiometry_receipt_sha256=sha(args.radiometry/'receipt.json'),
        radiometry_state_sha256=sha(state_path),source_observation_receipt_sha256=sha(args.observations/'receipt.json'),
        final_atlas_sha256=sha(args.output/'final_atlas.npz'),solve_state_sha256=sha(args.output/'solve_state.npz'),
        unchanged_fixed_pair_metrics=upstream['fixed_pair_metrics'],IRLS_CG_history=history,influence_CG=record,
        same_world_real_locks=int(known.sum()),teacher_direct_observation_texels=int(union.sum()),teacher_inferred_field_texels=int((adopted&~union).sum()),
        teacher_adopted_texels=int(adopted.sum()),source2_fallback_texels=int((source==2).sum()),
        real_locks_RGB_bitexact=bool(np.array_equal(final[known],seed['atlas_seed_linear'][known])),
        outside_permission_RGB_bitexact=bool(np.array_equal(final[~permission],seed['atlas_seed_linear'][~permission])),
        raw_or_hard_final_images_read=False,geometry_or_Gaussian_modified=False,new_diffusion_calls=0,nativeF=0,nativeB=0,optimizer_updates=0,
        acceptance='PENDING independent numerical and visual shared-input acceptance',seconds=time.time()-started)
    write(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
