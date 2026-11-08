"""Finish readouts from already saved R palette; no solver invocation."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy import sparse

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original-source',type=Path,required=True);a=p.parse_args()
    out=a.run;assert not (out/'result.json').exists()
    protocol=read(out/'protocol.json');assert sha(a.original_source)==protocol['source_sha256']
    inherited=read(protocol['inherited_input_protocol']);source=Path(inherited['r2_source']);assert sha(source)==inherited['r2_source_sha256']
    s=importlib.util.spec_from_file_location('immutable_R_inputs',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    inp=SimpleNamespace(**{k:Path(v) for k,v in inherited['input_paths'].items()});d=m.inputs(inp)
    palette=out/'R_guarded_MSE_parent_RGB.npz';z=m.load(palette);full=z['full_parent_RGB'];selection=m.load(out/'selection.npz')
    K=m.load(inp.k_palette)['full_parent_RGB'];assert full.dtype==np.float32 and full.shape==(10098,3) and np.isfinite(full).all() and full.min()>=0 and full.max()<=1
    assert np.array_equal(z['variable_parent_ids'],d['ids'])
    frozen=np.ones(10098,bool);frozen[selection['free_parent_ids']]=False
    assert full[frozen].tobytes()==K[frozen].tobytes()
    assert full[selection['seam_frozen_parent_ids']].tobytes()==K[selection['seam_frozen_parent_ids']].tobytes()
    deltaK=full[d['ids']].astype(np.float64)-K[d['ids']].astype(np.float64)
    for rec in d['operator_receipt']['records']:
        A=sparse.load_npz(inp.pixel_root/rec['operator_file']).tocsr();v=m.load(inp.pixel_root/rec['pixel_domains_file'])
        assert np.count_nonzero(A[v['registered_seam_support'].ravel()[v['pixel_index']]]@deltaK)==0
    total=full[d['ids']].astype(np.float64)-d['c0'];RK=d['r0']+d['PA']@d['delta_k'];RN=d['r0']+d['PA']@total
    cert=m.certificate(d,total,slack=0,reserve=False)
    calls=[read(out/f'call{i+1}_{c}_result.json') for i,c in enumerate(['red','green','blue'])]
    points=[]
    for raw,x,y,role in [(128,301,221,'white'),(128,310,222,'adjacent_dark'),(137,300,227,'white'),(137,311,228,'adjacent_dark')]:
        f=next(f for f in d['frames'] if f['raw']==raw and f['state']=='delete');q=np.flatnonzero(f['pixel_index']==y*480+x)
        if q.size:
            row=int(q[0])+f['photo_start'];gt=f['obs'][:,y,x].astype(np.float64);rgb=gt+RN[row]
            points.append(dict(raw=raw,state='delete',x=x,y=y,role=role,K_operator_MAE=float(abs(RK[row]).mean()),R_operator_MAE=float(abs(RN[row]).mean()),predicted_RGB=rgb.tolist()))
        else:points.append(dict(raw=raw,state='delete',x=x,y=y,role=role,status='OUTSIDE_FIXED_OPERATOR_PROTECTION_SUPPORT'))
    write(out/'predicted_white_dark_points.json',points)
    np.savez_compressed(out/'predicted_protection_residuals.npz',K_residual=RK,R_residual=RN,pixel_caps=d['pixel_cap'])
    result=dict(schema='R_GUARDED_MSE_FINITE_CPU_RESULT_V1',status='FINITE_BOUNDED_SEAM_GUARDED_PALETTE_READY_NATIVE8F',diagnostic=True,export_integrity_pass=True,
        source_sha256=protocol['source_sha256'],protocol_sha256=sha(out/'protocol.json'),finalization_source_sha256=sha(__file__),
        candidate=dict(path=str(palette),sha256=sha(palette),full_parent_rows=10098,variable_parent_rows=len(d['ids'])),
        solver_calls=calls,actual_solver_calls=3,solver_seconds=sum(c['seconds'] for c in calls),bounded_LS_converged_all3=all(c['success'] for c in calls),optimality_proof_claim=False,
        variable_parent_count=len(d['ids']),free_parent_count=len(selection['free_parent_ids']),seam_frozen_parent_count=len(selection['seam_frozen_parent_ids']),
        changed_parent_count=int(np.any(full!=K,axis=1).sum()),all_outside_free_parent_RGB_byte_exact_K=True,seam_guard_parent_RGB_byte_exact_K=True,
        exact_registered_seam_pixel_operator_RGB_change_max_abs=0.0,signed106_seam_operator_change_max_abs=float(abs(d['SA']@deltaK).max(initial=0)),
        prediction=dict(K_equal_pixel_channel_MSE=float((RK*RK).mean()),R_equal_pixel_channel_MSE=float((RN*RN).mean()),original16_photo_pass=sum(r['passed'] for r in cert['domains']),
            original118_seam_pass=sum(r['passed'] for r in cert['seams']),strict_main_pixel_cap_failures=cert['pixel_constraint_failed_count'],strict_main_pixel_cap_max_excess=cert['max_pixel_excess_over_original_cap']),
        original94_gates_require_actual_native_float_replay=True,original_threshold_changes=False,frozen_geometry_identity_opacity_and_H2_masks_changed=False,
        finalization_added_solver_calls=0,postprocessing_repair='Read existing finite saved palette; omit unsupported KEEP point readout. No colour or source/protocol rewrite, no solver replay.',
        new_Adam_updates=0,neural_steps=0,original_model_updates=0,GPU_forwards=0,backwards=0,native8F_diagnostic_authorized=True,automatic49=False,quality_accepted=False,human_verdict=None)
    write(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
