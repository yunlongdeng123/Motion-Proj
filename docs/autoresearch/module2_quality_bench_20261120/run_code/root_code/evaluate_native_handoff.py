"""Full native expected-road coverage and same-pass scene attribution, CPU only."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib
import numpy as np
from scipy import ndimage as ndi
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1');G=Q/'gpu_20261119';B=G/'quality_bench_20261120_r1';H=B/'highres'
sys.path.insert(0,str(B/'root_code'))
from evaluate_other_failures import summary,gap_components
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    assert (B/'native_detail_r1/completed.json').is_file(),'native detail stage first'
    out=B/'secondary_audit/native_r1';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    for raw in (110,86):
        t=load(H/f'targets/{raw:03d}.npz');p=load(H/f'rendered/M11/{raw:03d}.npz')
        r=load(H/f'native_r1/{raw:03d}/road_basis.npz');j=load(H/f'native_r1/{raw:03d}/joint_basis.npz')
        alpha=p['road_alpha'].squeeze();K=p['K'];w=j['entire_road_weight'];s=j['static_alphaT'];dy=j['dynamic_alphaT']
        source_sum=j['static_rgb']+j['dynamic_rgb']+j['whole_road_rgb']+j['env_rgb']
        row=dict(raw=raw,resolution=[1920,1280],domains={},K_pixels=int(K.sum()),
            native_basis_K_eligible_weight_max=float(j['eligible_weight'][K].max()),
            native_source_RGB_conservation_max=float(np.abs(source_sum-j['baseline_rgb']).max()),
            native_source_alpha_conservation_max=float(np.abs(s+dy+w-j['alpha']).max()))
        for region in ('A','C','S'):
            expected=t[region+'_pre_alpha'].astype(bool);stable=expected&(alpha>=.95)
            target=t['T_'+region]
            domain=dict(expected_pre_GS_alpha_pixels=int(expected.sum()),
                surface_hit_pixels=int((expected&t['surface_hit']).sum()),
                no_surface_hit_pixels=int((expected&~t['surface_hit']).sum()),
                source_valid_pixels=int((expected&t['source_material_valid']).sum()),
                coverage=[dict(alpha_threshold=a,pixels=int((expected&(alpha>=a)).sum()),below_pixels=int((expected&(alpha<a)).sum()))for a in(.05,.5,.9,.95,.99)],
                gaps={str(a):gap_components(expected&(alpha<a))for a in(.05,.9,.95)},
                road_alpha=summary(alpha[expected]),whole_road_joint_alphaT=summary(w[expected]),
                static_joint_alphaT=summary(s[expected]),dynamic_joint_alphaT=summary(dy[expected]),
                road_eligible_joint_alphaT=summary(j['eligible_weight'][expected]),
                high_road_alpha_but_joint_road_weight_below05=int((stable&(w<.05)).sum()),
                high_road_alpha_static_weight_ge50=int((stable&(s>=.5)).sum()),
                high_road_alpha_dynamic_weight_ge50=int((stable&(dy>=.5)).sum()),
                stable_road_static_ge50_joint_road_below05=int((stable&(s>=.5)&(w<.05)).sum()),
                static_dominant_over_expected=int((expected&(s>=.5)).sum()),
                road_color_error_auxiliary=float(np.abs(p['road_rgb']-alpha[None]*target)[:,expected].mean())if expected.any()else None,
                joint_color_error_auxiliary=float(np.abs(p['joint_rgb']-target)[:,expected].mean())if expected.any()else None,
                teacher_role='real_visible'if region=='A'else'frozen_pseudo_teacher_not_hidden_GT')
            row['domains'][region]=domain
        row['interpretation']='Source weights are from the same full-scene depth sort, not a semantic residual-car detector. High static contribution inside a legal deleted-target road domain supports static occlusion; inspect source image before labelling exact ghost components.'
        rows.append(row)
    result=dict(schema='native_road_quality_handoff_attribution.v1',rows=rows,seconds=time.monotonic()-start,
        native_calls=0,optimizer_updates=0,GPU_operations=0,source_components_not_counterfactual_renders=True)
    write(out/'results.json',result);print(json.dumps(dict(output=str(out),seconds=result['seconds'])))
if __name__=='__main__':main()
