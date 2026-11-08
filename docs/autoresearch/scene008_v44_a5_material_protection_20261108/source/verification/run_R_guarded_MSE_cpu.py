"""Finite R control: exact seam-pixel guard + equal protection-pixel bounded LS."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import lsq_linear


POINTS = [(128,301,221,'white'),(128,310,222,'adjacent_dark'),
          (137,300,227,'white'),(137,311,228,'adjacent_dark')]


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def timeout_handler(signum, frame): raise TimeoutError('Registered600s total bounded-LS budget exhausted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    inherited=read(a.input_protocol);source=Path(inherited['r2_source'])
    assert sha(source)==inherited['r2_source_sha256']
    spec=importlib.util.spec_from_file_location('immutable_parent_inputs',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    inp=SimpleNamespace(**{k:Path(v) for k,v in inherited['input_paths'].items()})
    d=m.inputs(inp);assert scipy.__version__=='1.15.3'
    kfull=m.load(inp.k_palette)['full_parent_RGB'];assert kfull.dtype==np.float32 and kfull.shape==(10098,3)
    assert np.array_equal(kfull[d['ids']].astype(np.float64)-d['c0'],d['delta_k'])
    guard=np.zeros(len(d['ids']),bool);guard_records=[];guard_matrices=[]
    for rec in d['operator_receipt']['records']:
        A=sparse.load_npz(inp.pixel_root/rec['operator_file']).tocsr().astype(np.float64)
        data=m.load(inp.pixel_root/rec['pixel_domains_file']);pix=data['pixel_index']
        seam=np.asarray(data['registered_seam_support'],bool)
        warm=m.load(inp.warm_root/f"frames/{rec['raw']:03d}/{rec['state']}_float.npz")
        represented=np.zeros((320,480),bool);represented.ravel()[pix]=True
        assert not (seam & (warm['group_alpha_T'][7]>0) & ~represented).any(), 'Seam support missing from pixel operator'
        S=A[seam.ravel()[pix]].tocoo()
        guard[np.unique(S.col[S.data>0])]=True
        guard_matrices.append(S.tocsr())
        guard_records.append(dict(raw=rec['raw'],state=rec['state'],registered_pixels=int(seam.sum()),
            registered_positive_road_pixels=int((seam & (warm['group_alpha_T'][7]>0)).sum()),
            represented_seam_rows=S.shape[0]))
    visible=np.asarray(d['PA'].sum(axis=0)).ravel()>0
    free=np.flatnonzero(visible & ~guard)
    assert free.size>0
    Afree=d['PA'][:,free].tocsr()
    assert Afree.shape[0]==9966 and np.isfinite(Afree.data).all() and (Afree.data>=0).all()
    for S in guard_matrices:
        assert np.count_nonzero(S[:,free].data)==0, 'Exact zero guard failed'
    for frame in d['frames']:
        obs=m.load(inp.obs_root/f"{frame['raw']:03d}.npz")
        assert not ((frame['masks']['protected_nonroad']|frame['masks']['originalroad']) & obs['visible'][1]).any()
    RK=d['r0']+d['PA']@d['delta_k']
    K=kfull[d['ids'][free]].astype(np.float64)
    a.output.mkdir(parents=True)
    source_sha=sha(__file__)
    protocol=dict(schema='R_EXACT_SEAM_PIXEL_GUARDED_PROTECTION_MSE_CPU_V1',status='SEALED_BEFORE_THREE_BOUNDED_LS_CALLS',
        task_id='SCENE008-V44-A5-MATERIAL-PROTECTION-20261108',run_id='R_guarded_MSE_v1',source_sha256=source_sha,
        inherited_input_protocol=str(a.input_protocol),inherited_input_protocol_sha256=sha(a.input_protocol),
        immutable_input_source=str(source),immutable_input_source_sha256=sha(source),
        operator_manifest_sha256=sha(inp.pixel_root/'operator_manifest.json'),K_palette_sha256=sha(inp.k_palette),
        variable_parent_count=len(d['ids']),free_parent_count=len(free),seam_frozen_parent_count=int(guard.sum()),
        zero_protection_mass_parents_frozen=int((~visible & ~guard).sum()),protected_pixels=Afree.shape[0],
        seam_guard='Freeze K RGB for every parent having any exact A>0 coefficient at any fixed registered_seam_support pixel, all4 KEEP/DELETE; no coefficient threshold',
        objective='All9966 protection pixels and RGB channels equal-weight MSE to original observation; one shared world-parent RGB table for every frame/state',
        reference='K is the zero-delta reference; solve delta around K. SciPy lsq_linear exposes no x0; its standard internal TRF initialization is retained.',
        bounds='per channel delta in [-K,1-K]; no clip, shrink, refit, repair, weighting or rollback',
        solver=dict(library='SciPy1.15.3',function='scipy.optimize.lsq_linear',method='trf',tol=1e-8,lsmr_tol=1e-8,max_iter=100,lsmr_maxiter=200,maximum_calls=3,total_seconds_budget=600,CPU_threads=1,no_retries=True),
        hidden_DELETE_policy='Only fixed original protected_nonroad/originalroad supervision. Original visible37 is excluded; unknown deleted road never receives observation car RGB supervision.',
        frozen='All original models, identities, geometry, opacity, H2 masks, camera/time, topology and all non-free road parents',
        numeric_acceptance='Original94, added16 and prior cap gates unchanged; real8F diagnostic allowed for finite bounded seam-guarded colour even if old gates fail; no49 on failure',
        documentation='https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.lsq_linear.html',
        neural_steps=0,GPU_forwards=0,diagnostic=True,automatic49=False,quality_accepted=False,human_verdict=None)
    write(a.output/'protocol.json',protocol);protocol_sha=sha(a.output/'protocol.json')
    np.savez_compressed(a.output/'selection.npz',variable_parent_ids=d['ids'],free_parent_ids=d['ids'][free],
        seam_frozen_parent_ids=d['ids'][guard],free_variable_columns=free)
    delta=np.zeros((len(free),3),np.float64);calls=[];start=time.monotonic();deadline=start+600
    signal.signal(signal.SIGALRM,timeout_handler)
    for ch,name in enumerate(['red','green','blue']):
        remaining=deadline-time.monotonic()
        if remaining<=0: break
        call=dict(channel=name,call_index=len(calls)+1,started_unix=time.time(),time_remaining_seconds=remaining)
        write(a.output/f'call{ch+1}_{name}_started.json',call);print(json.dumps(dict(event='START_BOUNDED_LS',**call)),flush=True)
        t=time.monotonic();signal.setitimer(signal.ITIMER_REAL,remaining)
        try:
            sol=lsq_linear(Afree,-RK[:,ch],bounds=(-K[:,ch],1-K[:,ch]),method='trf',tol=1e-8,lsmr_tol=1e-8,max_iter=100,lsmr_maxiter=200)
            assert np.isfinite(sol.x).all() and np.all(sol.x>=-K[:,ch]) and np.all(sol.x<=1-K[:,ch])
            delta[:,ch]=sol.x
            call.update(status=int(sol.status),success=bool(sol.success),iterations=int(sol.nit),cost=float(sol.cost),optimality=float(sol.optimality),message=str(sol.message),approximate=not bool(sol.success))
        except TimeoutError as exc:
            call.update(status='TOTAL_BUDGET_TIMEOUT',success=False,approximate=True,message=str(exc),channel_delta_retained_zero=True)
        finally: signal.setitimer(signal.ITIMER_REAL,0)
        call['seconds']=time.monotonic()-t;calls.append(call)
        write(a.output/f'call{ch+1}_{name}_result.json',call);print(json.dumps(dict(event='END_BOUNDED_LS',**call)),flush=True)
        if call['status']=='TOTAL_BUDGET_TIMEOUT': break
    full=kfull.copy();selected=np.asarray(K+delta,np.float32)
    assert np.isfinite(selected).all() and selected.min()>=0 and selected.max()<=1
    full[d['ids'][free]]=selected
    frozen=np.ones(10098,bool);frozen[d['ids'][free]]=False
    assert full[frozen].tobytes()==kfull[frozen].tobytes()
    assert full[d['ids'][guard]].tobytes()==kfull[d['ids'][guard]].tobytes()
    delta_from_K=full[d['ids']].astype(np.float64)-kfull[d['ids']].astype(np.float64)
    assert all(np.count_nonzero(S@delta_from_K)==0 for S in guard_matrices)
    palette=a.output/'R_guarded_MSE_parent_RGB.npz'
    np.savez_compressed(palette,full_parent_RGB=full,parent_RGB=full,variable_parent_ids=d['ids'],
        variable_parent_RGB=full[d['ids']],eligible_parent_ids=d['ids'],eligible_parent_RGB=full[d['ids']],
        free_parent_ids=d['ids'][free],seam_frozen_parent_ids=d['ids'][guard],delta_from_K=delta_from_K)
    total_delta=full[d['ids']].astype(np.float64)-d['c0'];Rnew=d['r0']+d['PA']@total_delta
    cert=m.certificate(d,total_delta,slack=0,reserve=False)
    photos=[]
    for rec in cert['domains']:
        source_rec=next(v for v in d['domains'] if (v['raw'],v['state'],v['name'])==(rec['raw'],rec['state'],rec['name']))
        photos.append(dict(rec,positive_A_pixels=len(source_rec['active_photo_rows']),prediction_only=True))
    table(a.output/'predicted_original16_photo.csv',photos)
    table(a.output/'predicted_original118_seam.csv',cert['seams'])
    points=[]
    for raw,x,y,role in POINTS:
        for frame in [f for f in d['frames'] if f['raw']==raw]:
            q=np.flatnonzero(frame['pixel_index']==y*480+x);assert q.size==1
            row=int(q[0])+frame['photo_start'];gt=frame['obs'][:,y,x].astype(np.float64)
            rgb=gt+Rnew[row];krgb=gt+RK[row]
            points.append(dict(raw=raw,state=frame['state'],x=x,y=y,role=role,
                K_operator_MAE=float(abs(krgb-gt).mean()),R_operator_MAE=float(abs(rgb-gt).mean()),
                predicted_red=float(rgb[0]),predicted_green=float(rgb[1]),predicted_blue=float(rgb[2])))
    table(a.output/'predicted_white_dark_points.csv',points)
    np.savez_compressed(a.output/'predicted_protection_residuals.npz',K_residual=RK,R_residual=Rnew,pixel_caps=d['pixel_cap'])
    result=dict(schema='R_GUARDED_MSE_FINITE_CPU_RESULT_V1',status='FINITE_BOUNDED_SEAM_GUARDED_PALETTE_READY_NATIVE8F',
        diagnostic=True,export_integrity_pass=True,source_sha256=source_sha,protocol_sha256=protocol_sha,
        candidate=dict(path=str(palette),sha256=sha(palette),full_parent_rows=10098,variable_parent_rows=len(d['ids'])),
        solver_calls=calls,actual_solver_calls=len(calls),solver_seconds=sum(c['seconds'] for c in calls),
        bounded_LS_converged_all3=bool(len(calls)==3 and all(c['success'] for c in calls)),
        optimality_proof_claim=False,all_nonconverged_results_explicitly_approximate=True,
        variable_parent_count=len(d['ids']),free_parent_count=len(free),seam_frozen_parent_count=int(guard.sum()),
        changed_parent_count=int(np.any(full!=kfull,axis=1).sum()),
        all_outside_free_parent_RGB_byte_exact_K=True,seam_guard_parent_RGB_byte_exact_K=True,
        exact_registered_seam_pixel_operator_RGB_change_max_abs=0.0,
        signed106_seam_operator_change_max_abs=float(abs(d['SA']@delta_from_K).max(initial=0)),
        prediction=dict(K_equal_pixel_channel_MSE=float((RK*RK).mean()),R_equal_pixel_channel_MSE=float((Rnew*Rnew).mean()),
            original16_photo_pass=sum(r['passed'] for r in photos),original118_seam_pass=sum(r['passed'] for r in cert['seams']),
            strict_main_pixel_cap_failures=cert['pixel_constraint_failed_count'],strict_main_pixel_cap_max_excess=cert['max_pixel_excess_over_original_cap']),
        seam_guard_records=guard_records,frozen_geometry_identity_opacity_and_H2_masks_changed=False,
        original94_gates_require_actual_native_float_replay=True,original_threshold_changes=False,
        new_Adam_updates=0,neural_steps=0,original_model_updates=0,GPU_forwards=0,backwards=0,
        native8F_diagnostic_authorized=True,automatic49=False,quality_accepted=False,human_verdict=None)
    assert sha(source)==inherited['r2_source_sha256'] and sha(__file__)==source_sha
    write(a.output/'result.json',result)
    print(json.dumps({k:result[k] for k in ['status','candidate','actual_solver_calls','solver_seconds','free_parent_count','seam_frozen_parent_count','prediction']}),flush=True)


if __name__=='__main__':main()
