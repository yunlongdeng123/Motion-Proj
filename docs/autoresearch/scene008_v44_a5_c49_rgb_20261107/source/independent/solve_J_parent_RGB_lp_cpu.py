"""Single sealed J parentRGB LP, CPU only; inspect never calls a solver."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linprog

PROTOCOL_SHA='f1adea94500d1488b3c266ca4c2645596181a08654f6bdb0892df050e8f65a76'
EXPECTED_MANIFEST_SHA='0894a5e2d9e63f872829e8128b40a3822f32135c7a841bb92769b590a9321c25'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def key(r):return (int(r['raw']),r['state'],r['kind'],r['edge'],int(r['tangent_bin']))


def inputs(args):
    assert sha(args.protocol)==PROTOCOL_SHA
    protocol=read(args.protocol);assert protocol['status']=='SEALED_BEFORE_J_CPU_SOLVE'
    assert sha(args.manifest)==EXPECTED_MANIFEST_SHA
    manifest=read(args.manifest);assert manifest['schema']=='J-road-parent-rgb-signed-seam-operator-v1'
    op=args.manifest.parent/manifest['matrix_file'];assert sha(op)==manifest['matrix_sha256']
    with np.load(op,allow_pickle=False) as z:data={n:z[n].copy() for n in z.files}
    ids=np.asarray(data['eligible_parent_ids'],np.int64);A=np.asarray(data['A'],np.float64);j0=np.asarray(data['baseline_J'],np.float64)
    base=np.asarray(data['initial_parent_RGB']);full=np.asarray(data['full_parent_RGB']);mu=np.asarray(data['eligible_parent_mass'],np.float64)
    assert ids.shape==(2119,) and np.array_equal(ids,np.unique(ids)) and ids.min()>=0 and ids.max()<10098
    assert A.shape==(106,2119) and j0.shape==(106,3) and base.shape==(2119,3) and full.shape==(10098,3) and mu.shape==(2119,)
    assert base.dtype==full.dtype==np.float32 and np.array_equal(base,full[ids])
    assert all(np.isfinite(v).all() for v in (A,j0,base,full,mu)) and (mu>0).all() and full.min()>=0 and full.max()<=1
    if 'full_parent_A' in data:
        fA=np.asarray(data['full_parent_A'],np.float64);assert fA.shape==(106,10098) and np.isfinite(fA).all()
        assert np.array_equal(A,fA[:,ids]),'Eligible operator columns differ from full-parent closure'
    records=manifest['seam_records'];assert len(records)==106 and len({key(r) for r in records})==106
    for r in records:
        assert r['state'] in ('keep','delete') and r['kind'] in ('world','primary128')
        assert int(r['minus_pixels'])>=3 and int(r['plus_pixels'])>=3 and len(r['minus_sha256'])==len(r['plus_sha256'])==64
    # Exact original helper metadata/mask hashes; no synthetic bin membership.
    from registered_seam_pairs import seam_pairs
    expected={key(p):p for raw in (94,105,128,137) for state in ('keep','delete') for p in [{**v,'state':state} for v in seam_pairs(raw,args.world_masks)]}
    assert set(expected)=={key(r) for r in records}
    for r in records:
        e=expected[key(r)]
        for n in ('minus_pixels','plus_pixels','minus_sha256','plus_sha256'):assert r[n]==e[n],(key(r),n)
    pools=[]
    for state in ('keep','delete'):
        for name,kind,edge,n in [('primary128','primary128','near_x',6),('world_near','world','near_x',24),('world_lateral','world','lateral_y',23)]:
            rows=np.array([i for i,r in enumerate(records) if r['state']==state and r['kind']==kind and r['edge']==edge],np.int64)
            assert len(rows)==n
            ref=protocol['published_reference_metrics'][name][state];latest=float(ref['latest']);current=float(ref['current_H2_C2000'])
            original=min(current,latest*.95,latest-1e-4) if name=='primary128' else min(current,latest+1e-6)
            measured=float(abs(j0[rows]).mean());assert abs(measured-current)<=1e-6,('j0 is not published H2-C2K residual',name,state,measured,current)
            pools.append(dict(name=name,state=state,rows=rows.tolist(),bins=n,current=current,latest=latest,
                original_target=original,tightened_target=original-1e-7,measured_j0=measured))
    assert sum(len(p['rows']) for p in pools)==106 and len({i for p in pools for i in p['rows']})==106
    return protocol,manifest,op,ids,A,j0,base,full,mu,records,pools


def compile_problem(A,j0,base,mu,pools):
    nd=base.size;ne=j0.size;I=sparse.eye(nd,format='csr');E=sparse.eye(ne,format='csr');B=sparse.kron(sparse.csr_matrix(A),sparse.eye(3),format='csr')
    Zde=sparse.csr_matrix((nd,ne));Zed=sparse.csr_matrix((ne,nd));P=sparse.lil_matrix((6,ne),dtype=np.float64)
    for i,p in enumerate(pools):
        for row in p['rows']:P[i,row*3:row*3+3]=1/(3*p['bins'])
    M=sparse.vstack([sparse.hstack([I,-I,Zde]),sparse.hstack([-I,-I,Zde]),
        sparse.hstack([B,Zed,-E]),sparse.hstack([-B,Zed,-E]),
        sparse.hstack([sparse.csr_matrix((6,2*nd)),P.tocsr()])],format='csr')
    rhs=np.r_[np.zeros(2*nd),-j0.ravel(),j0.ravel(),[p['tightened_target'] for p in pools]]
    weights=mu/mu.mean();objective=np.r_[np.zeros(nd),np.repeat(weights,3),np.zeros(ne)]
    bounds=list(zip(-base.astype(np.float64).ravel(),1-base.astype(np.float64).ravel()))+[(0,None)]*(nd+ne)
    assert M.shape==(13356,13032) and rhs.shape==(13356,) and objective.shape==(13032,)
    assert np.isfinite(M.data).all() and np.isfinite(rhs).all() and np.isfinite(objective).all()
    return M,rhs,objective,bounds,weights


def pool_certificate(A,j0,delta,pools):
    j=j0+A@delta;rows=[]
    for p in pools:
        value=float(abs(j[p['rows']]).mean())
        rows.append(dict(name=p['name'],state=p['state'],bins=p['bins'],actual_mean_absolute_residual=value,
            original_target=p['original_target'],tightened_target=p['tightened_target'],
            margin_to_original=p['original_target']-value,margin_to_tightened=p['tightened_target']-value,
            original_pass=bool(value<=p['original_target']),tightened_pass_with_1e_9_audit_tolerance=bool(value<=p['tightened_target']+1e-9)))
    return j,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['inspect','solve'],default='inspect')
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--verification',type=Path)
    p.add_argument('--world-masks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists(),'Keep prior result/attempt immutable';args.output.mkdir(parents=True)
    protocol,manifest,op,ids,A,j0,base,full,mu,records,pools=inputs(args)
    M,rhs,c,bounds,weights=compile_problem(A,j0,base,mu,pools)
    source=[args.manifest,args.protocol,op,Path(__file__),Path(__file__).with_name('registered_seam_pairs.py')]
    before={str(v):sha(v) for v in source}
    result={'schema':'J_PARENT_RGB_SINGLE_LP_CPU_RESULT_V1','status':'COMPILED_NOT_SOLVED','actual_solver_calls':0,
        'scipy_version':scipy.__version__,'protocol_sha256':sha(args.protocol),'operator_manifest_sha256':sha(args.manifest),
        'operator_npz_sha256':sha(op),'source_sha256':sha(__file__),'matrix_shape':list(M.shape),'matrix_nnz':M.nnz,
        'delta_variables':6357,'absolute_delta_variables':6357,'residual_epigraph_variables':318,
        'original_mu_stats':dict(min=float(mu.min()),max=float(mu.max()),mean=float(mu.mean())),
        'mean_one_mu_stats':dict(min=float(weights.min()),max=float(weights.max()),mean=float(weights.mean())),
        'pools':pools,'original_scene_and_Gaussian_updates':0,'Adam_updates':0,'GPU_forwards_by_this_CPU_script':0,
        'photo_protection_quality':'PENDING_REAL8F_FULL4560_AND_IMAGES','quality_accepted':False,'human_verdict':None}
    sparse.save_npz(args.output/'compiled_A_ub.npz',M)
    np.savez_compressed(args.output/'compiled_vectors.npz',b_ub=rhs,objective=c,delta_lower=-base,delta_upper=1-base,eligible_parent_ids=ids,mu_mean_one=weights)
    if args.mode=='solve':
        assert args.verification is not None
        v=read(args.verification);source.append(args.verification);before[str(args.verification)]=sha(args.verification)
        assert v['status']=='PASS_J_OPERATOR_AND_BOUNDED_NATIVE_AFFINITY_NO_FIT'
        assert v['operator_manifest_sha256']==sha(args.manifest) and v['perturbation_forwards']==8
        assert float(v['max_native_linearity_error'])<=1e-6 and v['input_bytes_mtime_and_model_binding_unchanged'] is True
        assert scipy.__version__=='1.13.1'
        lock=args.manifest.parent/'J_single_CPU_LP_solve_started.json'
        fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w') as f:json.dump(dict(protocol_sha256=sha(args.protocol),source_sha256=sha(__file__),verification_sha256=sha(args.verification),operator_sha256=sha(op),started_unix=time.time(),maximum_calls=1),f,indent=2)
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,time_limit=120)
        result.update(actual_solver_calls=1,solver_method='highs',solver_options=options,native_affinity_receipt_sha256=sha(args.verification),native_affinity_max_error=v['max_native_linearity_error'])
        write(args.output/'started.json',result);start=time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');sol=linprog(c,A_ub=M,b_ub=rhs,bounds=bounds,method='highs',options=options)
        result.update(solver_seconds=time.monotonic()-start,solver_status=int(sol.status),solver_message=str(sol.message),
            solver_success=bool(sol.success),solver_iterations=int(sol.nit),solver_warnings=[str(w.message) for w in caught])
        if sol.status==0:
            delta=sol.x[:6357].reshape(2119,3);_,cert64=pool_certificate(A,j0,delta,pools)
            violation=max(0.,float(np.max(M@sol.x-rhs)));box=max(0.,float((-base.astype(np.float64)-delta).max()),float((base.astype(np.float64)+delta-1).max()))
            passed64=all(r['tightened_pass_with_1e_9_audit_tolerance'] for r in cert64) and box<=1e-9 and violation<=1e-9
            selected=np.asarray(np.clip(base.astype(np.float64)+delta,0,1),np.float32);delta32=selected.astype(np.float64)-base.astype(np.float64)
            _,cert32=pool_certificate(A,j0,delta32,pools);passed32=all(r['original_pass'] for r in cert32)
            new_full=full.copy();new_full[ids]=selected;fixed=np.ones(10098,bool);fixed[ids]=False
            assert new_full[fixed].tobytes()==full[fixed].tobytes() and np.isfinite(new_full).all() and new_full.min()>=0 and new_full.max()<=1
            result.update(status='FEASIBLE_CPU_CERTIFIED_PENDING_REAL8F' if passed64 and passed32 else 'SOLVER_SUCCESS_EXPORT_CERTIFICATE_FAILED_STOP',
                float64_certificate_pass=passed64,float32_certificate_pass=passed32,float64_pools=cert64,float32_pools=cert32,
                solver_inequality_max_violation=violation,delta_box_max_violation=box,
                solver_objective=float(sol.fun),actual_weighted_L1_objective=float((weights[:,None]*abs(delta)).sum()),
                actual_original_mass_weighted_L1=float((mu[:,None]*abs(delta)).sum()),
                mean_absolute_RGB_change=float(abs(delta32).mean()),max_absolute_RGB_change=float(abs(delta32).max()),
                eligible_parent_rows_changed=int((abs(delta32).max(1)>0).sum()),fixed7979_bitwise_unchanged=True)
            np.savez_compressed(args.output/'LP_numeric_solution.npz',solver_x=sol.x,delta_float64=delta,delta_actual_float32=delta32)
            if passed64 and passed32:
                out=args.output/'J_candidate_parent_RGB.npz';np.savez_compressed(out,full_parent_RGB=new_full,parent_RGB=new_full,eligible_parent_ids=ids,eligible_parent_RGB=selected,delta_from_formal2k=delta32)
                result['candidate']=dict(path=str(out),sha256=sha(out),dtype='float32',full_parent_rows=10098,eligible_parent_rows=2119,source_formal_material_steps=2000,new_Adam_updates=0)
        elif sol.status==2:result['status']='REGISTERED_RESTRICTED_LP_REPORTED_INFEASIBLE_STOP'
        elif sol.status==1:result['status']='SOLVER_TIME_OR_ITERATION_LIMIT_INCONCLUSIVE_STOP'
        else:result['status']='SOLVER_NUMERICAL_OR_OTHER_INCONCLUSIVE_STOP'
        result['scope_limit']='Only2119eligible bounded parentRGB variables/7979fixed and this measured operator/6targets; no claim about all10098parents or all representations.'
    assert all(sha(v)==before[str(v)] for v in source)
    result['source_integrity']='PASS_READ_INPUT_SHA_UNCHANGED';result['input_files']=[dict(path=str(v),sha256=before[str(v)]) for v in source]
    write(args.output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','actual_solver_calls','solver_status','solver_seconds','float64_certificate_pass','float32_certificate_pass','candidate']}))


if __name__=='__main__':main()
