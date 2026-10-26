"""首对颜色拒绝的有界诊断：3帧交替块的中位颜色及留出块核验。
观察到首源阴影异常后追加的诊断；不改原初始化、门限或GPU主对照。
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,warnings
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);result={}
for name in ['window49','full199']:
    a=dict(np.load(p/f'{name}_initialization.npz'));raw=a['source_raw'];valid=a['source_valid'];rgb=a['source_rgb']
    fit=(raw//3)%2==0;fit_valid=valid&fit[:,None];check_valid=valid&~fit[:,None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(np.where(fit_valid[...,None],rgb,np.nan),axis=0)
    mae=np.abs(rgb-median).mean(2);fit_in=fit_valid&(mae<=.05);check_in=check_valid&(mae<=.05)
    fit_count=fit_in.sum(0);check_count=check_in.sum(0)
    lo=np.where(fit_in,raw[:,None],32767).min(0);hi=np.where(fit_in,raw[:,None],-1).max(0)
    accepted=(fit_count>=3)&(fit_count>=.6*fit_valid.sum(0))&(hi-lo>=4)&(check_count>=2)&(check_count>=.6*check_valid.sum(0))&a['surface_supported']
    np.savez_compressed(p/f'{name}_color_consensus.npz',rgb=np.nan_to_num(median,nan=.18),eligible=accepted,fit_inliers=fit_count,check_inliers=check_count,fit=fit,first_pair=a['eligible'])
    critical=[]
    if name=='full199':
        for c in json.loads((p/'coverage.json').read_text())['critical']:
            uid=c['variants'][name]['unit_id'];critical.append(dict(pixel=c['pixel'],unit_id=uid,accepted=bool(accepted[uid]),fit_sources=raw[fit_in[:,uid]].tolist(),check_sources=raw[check_in[:,uid]].tolist(),rgb=median[uid].tolist()))
    result[name]=dict(accepted=int(accepted.sum()),first_pair_accepted=int(a['eligible'].sum()),added=int((accepted&~a['eligible']).sum()),critical=critical)
result['protocol']=dict(after_observing_first_source_color_outlier=True,fit='raw//3 even',check='raw//3 odd',min_fit=3,min_fit_span_frames=4,min_check=2,min_fraction=.6,rgb_mae=.05,
    old_arrays_modified=False,used_in_gpu_primary_comparison=False,limitation='temporal-block consistency within same scene, not independent road GT; shadows/calibration error remain')
(p/'color_consensus.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
