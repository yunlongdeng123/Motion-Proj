"""Zero-update domain and sampling diagnostics, explicitly after the full native bench."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
import numpy as np
from scipy import ndimage as ndi
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1');G=Q/'gpu_20261119';B=G/'quality_bench_20261120_r1';H=B/'highres'
sys.path.insert(0,str(B/'metrics'))
from road_detail_metrics import _masked_hf,_hf_stats,_strict_core
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def box4_chw(x):return x.reshape(3,320,4,480,4).mean((2,4))

def main():
    assert (B/'native_detail_r1/completed.json').is_file()
    out=B/'frequency_diagnostic_r1';out.mkdir(exist_ok=False);start=time.monotonic()
    write(out/'contract.json',dict(purpose='separate new pre-alpha legal-domain expansion from fine sampling of the same trained field',
        timing='added after inspecting first full native benchmark; posthoc diagnosis, not primary blinded evidence',
        definitions='source-only old480 score NN labels split native full domain; identical context for all split scores',
        optimizer_updates=0,native_calls=0,no_new_method=True))
    rows=[];down=[]
    for raw in (110,86):
        t=load(H/f'targets/{raw:03d}.npz');alpha=load(H/f'rendered/M11/{raw:03d}.npz')['road_alpha'].squeeze();stable=alpha>=.95
        packets={a:load(H/f'rendered/{a}/{raw:03d}.npz')for a in ('M01','M11','M10','M11_noH')}
        for region in ('C','S'):
            legal=t[region+'_pre_alpha'];old=t['legacy480_'+region+'_texture_domain'];context=legal&stable
            teacher=t['T_'+region].transpose(1,2,0).astype(float)
            scores=dict(full=legal,in_legacy480=legal&old,outside_legacy480=legal&~old)
            ref={sigma:_masked_hf(teacher,context,sigma,3.)[0] for sigma in(1.,2.,4.)}
            cores={sigma:_strict_core(context,int(3*sigma+.5))for sigma in(1.,2.,4.)}
            for name,p in packets.items():
                rgb=p['road_rgb'].transpose(1,2,0).astype(float)
                straight=np.divide(rgb,alpha[...,None],out=np.zeros_like(rgb),where=stable[...,None])
                for sigma in(1.,2.,4.):
                    hp,_=_masked_hf(straight,context,sigma,3.)
                    for split,score in scores.items():
                        core=score&cores[sigma]
                        rows.append(dict(raw=raw,region=region,case=name,sigma_native_pixels=sigma,split=split,
                            expected_pixels=int(score.sum()),stable_pixels=int((score&stable).sum()),
                            core_pixels=int(core.sum()),HF=_hf_stats(hp,ref[sigma],core)))
            f=load(G/f'evaluation/frozen_r2/{raw:03d}.npz')
            for name in ('M01','M11','M10'):
                saved=load(G/f'formal_r1/{name}/C800/rendered/{raw:03d}.npz')
                delta=box4_chw(packets[name]['road_rgb'])-saved['road_rgb']
                down.append(dict(raw=raw,region=region,case=name,old_score_pixels=int(f[region].sum()),
                    native_box4_vs_actual480_RGB_MAE=float(np.abs(delta[:,f[region]]).mean()),
                    native_box4_vs_actual480_RGB_RMSE=float(np.sqrt((delta[:,f[region]]**2).mean())),
                    native_box4_vs_actual480_alpha_MAE=float(np.abs(packets[name]['road_alpha'].squeeze().reshape(320,4,480,4).mean((1,3))-saved['road_alpha'].squeeze())[f[region]].mean()),
                    warning='Actual native GS alphaT/lowpass and local material footprint differ from the480 renderer; this measures the combined sampling effect, not a universal resolution-invariance certificate.'))
    result=dict(rows=rows,native_downsample_consistency=down,seconds=time.monotonic()-start,
        native_calls=0,optimizer_updates=0,GPU_operations=0,primary_bench_unchanged=True,
        interpretation='Compare within a split at the same native pixel scale; the old480-source subset split does not make199BUILD a blind test.')
    write(out/'results.json',result);print(json.dumps(dict(output=str(out),seconds=result['seconds'])))
if __name__=='__main__':main()
