"""Single-factor S pair: same Rchild RGB basis/LS, remove extra seam-colour guard."""
import argparse,hashlib,importlib.util,json,os,signal,time
from pathlib import Path
from types import SimpleNamespace
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import lsq_linear

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def expired(sig,frame):raise TimeoutError('600s total LS budget exhausted')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-protocol',type=Path,required=True);p.add_argument('--guarded-child-run',type=Path,required=True);p.add_argument('--child-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();ref=read(a.input_protocol);source=Path(ref['r2_source']);assert sha(source)==ref['r2_source_sha256']
    s=importlib.util.spec_from_file_location('immutable_parent_inputs',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    inp=SimpleNamespace(**{k:Path(v) for k,v in ref['input_paths'].items()});d=m.inputs(inp);assert scipy.__version__=='1.15.3'
    meta=m.load(a.child_root/'children.npz');ids=meta['variable_child_ids'];mapping=meta['child_to_parent_ids'];c0=meta['initial_child_RGB'].astype(np.float64)
    assert ids.dtype==np.int64 and ids.shape==(20725,) and mapping.shape==(40392,)
    assert np.array_equal(ids,np.unique(ids)) and np.array_equal(c0,meta['full_child_RGB'][ids].astype(np.float64))
    Kparent=m.load(inp.k_palette)['full_parent_RGB'];Kfull=Kparent[mapping].copy();guard=np.zeros(len(ids),bool);blocks=[];Smats=[]
    for frame in d['frames']:
        raw,state=frame['raw'],frame['state'];root=a.child_root/f'{raw:03d}_{state}'
        A=sparse.load_npz(root/'A_child_rgb_csr.npz').tocsr().astype(np.float64);v=m.load(root/'pixel_domains_baseline.npz')
        assert A.shape==(len(v['pixel_index']),len(ids)) and np.isfinite(A.data).all() and (A.data>=0).all()
        union=v['protected_nonroad']|v['originalroad'];active=union.ravel()[v['pixel_index']] & (np.asarray(A.sum(axis=1)).ravel()>0)
        assert np.array_equal(v['pixel_index'][active],frame['pixel_index'])
        assert np.array_equal(m.chw(v['full_baseline_rgb']),frame['rgb0']) and np.array_equal(m.chw(v['full_observation_rgb']),frame['obs'])
        assert np.array_equal(v['protected_nonroad'],frame['masks']['protected_nonroad']) and np.array_equal(v['originalroad'],frame['masks']['originalroad'])
        S=A[v['registered_seam_support'].ravel()[v['pixel_index']]].tocoo();guard[np.unique(S.col[S.data>0])]=True;Smats.append(S.tocsr());blocks.append(A[active])
    PA=sparse.vstack(blocks,format='csr');assert PA.shape==(9966,20725)
    visible=np.asarray(PA.sum(axis=0)).ravel()>0;free=np.flatnonzero(visible);Af=PA[:,free].tocsr();assert free.size==20725
    sk=m.load(a.child_root/'seam_child_operator.npz');SA=np.asarray(sk['A_child'],np.float64);j0=np.asarray(sk['baseline_J'],np.float64);assert SA.shape==(106,20725) and j0.shape==(106,3)
    dk=Kfull[ids].astype(np.float64)-c0;RK=d['r0']+PA@dk;parentRK=d['r0']+d['PA']@d['delta_k']
    a.output.mkdir(parents=True);out=a.output;source_sha=sha(__file__)
    protocol=dict(schema='S_CHILD_ALL_PROTECTION_MSE_SINGLE_FACTOR_CPU_V1',status='SEALED_BEFORE_THREE_BOUNDED_LS_CALLS',source_sha256=source_sha,
        run_id='S_child_all_protection_MSE_v1',material_basis='child_RGB',guarded_child_pair_protocol_sha256=sha(a.guarded_child_run/'protocol.json'),
        immutable_input_source_sha256=sha(source),inherited_input_protocol=str(a.input_protocol),inherited_input_protocol_sha256=sha(a.input_protocol),
        parent_operator_manifest_sha256=sha(inp.pixel_root/'operator_manifest.json'),operator_manifest_sha256=sha(a.child_root/'operator_manifest.json'),child_metadata_sha256=sha(a.child_root/'children.npz'),
        variable_child_count=len(ids),free_child_count=len(free),seam_frozen_child_count=0,seam_supported_child_count=int(guard.sum()),protected_pixels=9966,
        single_changed_factor='Remove only the artificial entire-seam-pixel RGB freeze used by Rchild. Same20725 visible children, fixed original geometry/opacity/actors/identity, masks,9966target, K reference, equal-pixel MSE, TRF parameters and3call600s budget.',
        seam_guard='REMOVED_FOR_THIS_ONLY_COUNTERFACTUAL; registered seam scientific K6+106+Kactual6 targets remain unchanged and require native evaluation',
        objective='Equal MSE over same9966 fixed protection pixels and3channels to original observation, one world-child table for all4 KEEP/DELETE',
        reference='Kchild=Kparent[immutable child_to_parent_ids]; solve delta around K; standard lsq_linear internal initialization',
        bounds='delta in[-Kchild,1-Kchild], no clipping/shrink/reweight/rollback',
        solver=dict(library='SciPy1.15.3',function='scipy.optimize.lsq_linear',method='trf',tol=1e-8,lsmr_tol=1e-8,max_iter=100,lsmr_maxiter=200,maximum_calls=3,total_seconds_budget=600,CPU_threads=1,no_retries=True),
        parent_render_colour_arrays='Historical provenance unchanged; only actual render_colours_srgb40392 may change',
        hidden_DELETE_policy='Original fixed NR/road masks exclude visible37; no unknown deletion-road observation-car supervision',
        original94_and16_gates_unchanged=True,geometry_opacity_identity_H2_support_changes=0,diagnostic=True,automatic49=False,quality_accepted=False,human_verdict=None)
    write(out/'protocol.json',protocol);proto_sha=sha(out/'protocol.json');np.savez_compressed(out/'selection.npz',variable_child_ids=ids,free_child_ids=ids[free],seam_frozen_child_ids=np.empty(0,np.int64),seam_supported_child_ids=ids[guard])
    K=Kfull[ids[free]].astype(np.float64);delta=np.zeros_like(K);calls=[];deadline=time.monotonic()+600;signal.signal(signal.SIGALRM,expired)
    for ch,name in enumerate(['red','green','blue']):
        remain=deadline-time.monotonic()
        if remain<=0:break
        rec=dict(call_index=ch+1,channel=name,started_unix=time.time());write(out/f'call{ch+1}_{name}_started.json',rec);print(json.dumps(dict(event='START_CHILD_LS',**rec)),flush=True);t=time.monotonic();signal.setitimer(signal.ITIMER_REAL,remain)
        try:
            sol=lsq_linear(Af,-RK[:,ch],bounds=(-K[:,ch],1-K[:,ch]),method='trf',tol=1e-8,lsmr_tol=1e-8,max_iter=100,lsmr_maxiter=200)
            assert np.isfinite(sol.x).all() and np.all(sol.x>=-K[:,ch]) and np.all(sol.x<=1-K[:,ch]);delta[:,ch]=sol.x
            rec.update(status=int(sol.status),success=bool(sol.success),approximate=not bool(sol.success),iterations=int(sol.nit),cost=float(sol.cost),optimality=float(sol.optimality),message=str(sol.message))
        except TimeoutError as exc:rec.update(status='TOTAL_BUDGET_TIMEOUT',success=False,approximate=True,message=str(exc),channel_delta_retained_zero=True)
        finally:signal.setitimer(signal.ITIMER_REAL,0)
        rec['seconds']=time.monotonic()-t;calls.append(rec);np.savez_compressed(out/f'call{ch+1}_{name}_delta.npz',free_child_ids=ids[free],delta_from_K=delta[:,ch]);write(out/f'call{ch+1}_{name}_result.json',rec);print(json.dumps(dict(event='END_CHILD_LS',**rec)),flush=True)
        if rec['status']=='TOTAL_BUDGET_TIMEOUT':break
    full=Kfull.copy();full[ids[free]]=np.asarray(K+delta,np.float32);assert np.isfinite(full).all() and full.min()>=0 and full.max()<=1
    frozen=np.ones(40392,bool);frozen[ids[free]]=False;assert full[frozen].tobytes()==Kfull[frozen].tobytes()
    deltaK=full[ids].astype(np.float64)-Kfull[ids].astype(np.float64);seam_RGB_delta=max(float(abs(S@deltaK).max(initial=0)) for S in Smats)
    palette=out/'S_child_all_protection_MSE_RGB.npz';np.savez_compressed(palette,full_child_RGB=full,variable_child_ids=ids,child_to_parent_ids=mapping,variable_child_RGB=full[ids],free_child_ids=ids[free],seam_frozen_child_ids=np.empty(0,np.int64),seam_supported_child_ids=ids[guard])
    dc=d.copy();dc.update(PA=PA,SA=SA,j0=j0,c0=c0,delta_k=dk,ids=ids);total=full[ids].astype(np.float64)-c0;RN=d['r0']+PA@total;cert=m.certificate(dc,total,slack=0,reserve=False)
    result=dict(schema='S_CHILD_ALL_PROTECTION_MSE_FINITE_CPU_RESULT_V1',status='FINITE_BOUNDED_CHILD_ALL_PROTECTION_PALETTE_READY_NATIVE8F',diagnostic=True,material_basis='child_RGB',export_integrity_pass=True,source_sha256=source_sha,protocol_sha256=proto_sha,
        candidate=dict(path=str(palette),sha256=sha(palette),full_child_rows=40392,variable_child_rows=len(ids)),actual_solver_calls=len(calls),solver_calls=calls,solver_seconds=sum(c['seconds'] for c in calls),bounded_LS_converged_all3=bool(len(calls)==3 and all(c['success'] for c in calls)),optimality_proof_claim=False,
        variable_child_count=len(ids),free_child_count=len(free),seam_frozen_child_count=0,seam_supported_child_count=int(guard.sum()),changed_child_count=int(np.any(full!=Kfull,axis=1).sum()),all_outside_free_child_RGB_byte_exact_K=True,seam_guard_applied=False,registered_seam_pixel_operator_RGB_change_max_abs=seam_RGB_delta,signed106_seam_operator_change_max_abs=float(abs(SA@deltaK).max(initial=0)),
        same_protection_pixel_indices_masks_baseline_observation_byte_exact_parent=True,child_K_model_vs_parent_K_model_max_abs=float(abs(RK-parentRK).max(initial=0)),
        prediction=dict(K_equal_pixel_channel_MSE=float((RK*RK).mean()),Schild_equal_pixel_channel_MSE=float((RN*RN).mean()),original16_photo_pass=sum(r['passed'] for r in cert['domains']),original118_seam_pass=sum(r['passed'] for r in cert['seams']),strict_main_pixel_cap_failures=cert['pixel_constraint_failed_count'],strict_main_pixel_cap_max_excess=cert['max_pixel_excess_over_original_cap']),
        new_Adam_updates=0,neural_steps=0,GPU_forwards=0,backwards=0,original_model_updates=0,geometry_opacity_identity_H2_support_changes=0,native8F_diagnostic_authorized=True,automatic49=False,quality_accepted=False,human_verdict=None)
    write(out/'result.json',result);print(json.dumps(result),flush=True)
    write(out/'predicted_original16_photo.json',cert['domains']);write(out/'predicted_original118_seam.json',cert['seams']);np.savez_compressed(out/'predicted_protection_residuals.npz',K_residual=RK,Rchild_residual=RN,pixel_caps=d['pixel_cap'])

if __name__=='__main__':main()
