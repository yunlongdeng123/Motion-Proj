"""CPU-only audit of the one registered10px image-edge blend fallback C."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    parent=a.run/'targets';root=a.run/'targets_edgeblend_v1';contract=read(root/'edgeblend_contract.json');manifest=read(root/'target_manifest.json')
    assert contract['width_pixels']==10 and manifest['fusion_ready'] is True
    assert contract['parent_target_manifest_sha256']==sha(parent/'target_manifest.json')
    assert manifest['parent_target_manifest_sha256']==sha(parent/'target_manifest.json')
    prior=read(parent/'target_manifest.json');index={r['raw']:r for r in prior['rows']}
    stats={g:dict(pixels=0,weight_sum=0.,full_B_pixels=0,B_error_sum=0.,C_error_sum=0.) for g in ('BUILD40','HELD9_EVAL_TARGET_QUERY_RGB_ANCHORED')};rows=[];fit=0
    for r in manifest['rows']:
        raw=int(r['raw']);pfile=parent/index[raw]['file'];cfile=root/r['file'];old=load(pfile);new=load(cfile)
        assert sha(cfile)==r['sha256'];assert all(np.array_equal(new[k],old[k]) for k in old)
        M=old['fusion_replace_mask'];O=old['observation_rgb'];B=old['fusion_rgb'];C=new['edgeblend_rgb']
        w=np.clip(distance_transform_edt(M)/10,0,1).astype(np.float32)
        expected=(O+w[...,None]*(B-O)).astype(np.float32);error=float(np.max(np.abs(expected-C)))
        assert error<2e-7 and np.isfinite(C).all() and C.min()>=0 and C.max()<=1
        assert np.array_equal(C[~M],O[~M]) and np.array_equal(new['fit_mask'],old['fit_mask'])
        role='BUILD40' if raw in manifest['BUILD40'] else 'HELD9_EVAL_TARGET_QUERY_RGB_ANCHORED'
        s=stats[role];s['pixels']+=int(M.sum());s['weight_sum']+=float(w[M].astype(float).sum());s['full_B_pixels']+=int((M&(w==1)).sum())
        s['B_error_sum']+=float(np.abs(B[M]-O[M]).mean(1).astype(float).sum());s['C_error_sum']+=float(np.abs(C[M]-O[M]).mean(1).astype(float).sum())
        if role=='BUILD40':fit+=int(new['fit_mask'].sum())
        rows.append(dict(raw=raw,role=role,replace_pixels=int(M.sum()),fit_pixels=int(new['fit_mask'].sum()),formula_max_abs_diff=error,
            mean_weight_in_replace=float(w[M].mean()) if M.any() else None,max_weight=float(w.max()),full_B_pixels=int((M&(w==1)).sum()),target_sha256=sha(cfile),parent_target_sha256=sha(pfile)))
    assert len(rows)==49 and fit==714546
    for s in stats.values():
        n=s['pixels'];s.update(mean_weight=s['weight_sum']/n if n else None,B_MAE=s['B_error_sum']/n if n else None,C_MAE=s['C_error_sum']/n if n else None)
    desc='C is O+clip(distance_inside(common replacement mask)/10,0,1)*(B-O): state-shared encoded-sRGB desired full image, view-dependent mask weights and per-query real O anchors. B retains the190-nonHELD shared atlas; C itself is not a pure world-consistent atlas projection. HELD9 O enters evaluation-target construction only, not the material fit.'
    report=dict(schema='ONE_FIXED10PX_EDGEBLEND_TARGET_SOURCE_AUDIT_V1',status='DATA_AND_CONTRACT_CHECKS_PASS_WITH_LIMITATIONS',
        input_sha256={str(p):sha(p) for p in (root/'edgeblend_contract.json',root/'target_manifest.json',parent/'target_manifest.json',a.run/'red_review/source_and_fit_contract_audit_v1.json')},
        target_interpretation=desc,rows=rows,weight_and_replace_fidelity=stats,BUILD_fit_pixels=fit,parent_AB_arrays_unchanged=True,
        original_AB_review_unchanged=True,registered_width_pixels=10,width_search_performed=False,GPU_used=False,model_updates=0,
        limitations=['C adds same-view real RGB inside replace; target-vs-O error reduction is largely by construction, not independent fusion evidence.',
            'HELD9 C-target fidelity is query-RGB conditioned even inside replace; only the native3D render remains excluded from current HELD9 fitting.',
            'Image-mask distance weights are view-dependent; do not claim a purely shared world-atlas target or isolate Poisson as the cause of an improvement.',
            'The one C400-pair bake must be evaluated independently at identical fixed masks/parameters/budget; no legacy-static update or geometry change is allowed.'],
        executed_source_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(stats=stats,fit=fit,sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
