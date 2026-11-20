"""Secondary CPU diagnostics after detail evaluation; no training or native render."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json,sys,time,hashlib
import numpy as np
from scipy import ndimage as ndi

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1');G=Q/'gpu_20261119';B=G/'quality_bench_20261120_r1'
sys.path.insert(0,str(B/'metrics'))
from road_detail_metrics import evaluate_detail,ResolutionSpec,TileLayout

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(x):
    a=np.asarray(x,np.float64)
    return dict(count=int(a.size),mean=float(a.mean()),p0_p05_p50_p95_p100=np.quantile(a,[0,.05,.5,.95,1]).tolist()) if a.size else dict(count=0,mean=None)
def gap_components(mask):
    labels,n=ndi.label(mask,np.ones((3,3),bool));counts=np.bincount(labels.ravel())[1:]
    return dict(components=int(n),largest_pixels=int(counts.max()) if n else 0,
                total_pixels=int(mask.sum()),largest10_pixels=sorted(counts.tolist(),reverse=True)[:10])

def main():
    # The ordering gate is evidentiary, not a task controller.
    assert (B/'lowres_r1/completed.json').is_file(),'detail evaluation first'
    out=B/'secondary_audit/lowres_r1';out.mkdir(parents=True,exist_ok=False)
    P=Q.parent/'road_capacity_cpu_20261117_r1/supervision_v4_candidate_r1'
    rows=[];binding={};started=time.monotonic()
    for raw in (86,110):
        files=dict(targets=P/f'{raw:03d}.npz',frozen=G/f'evaluation/frozen_r2/{raw:03d}.npz',
            packet=G/f'formal_r1/M11/C800/rendered/{raw:03d}.npz',road=G/f'native_qualification_r2/basis/{raw:03d}/road_basis.npz',
            joint=G/f'native_qualification_r2/basis/{raw:03d}/joint_basis.npz')
        d,f,p,r,j=[load(files[k]) for k in ('targets','frozen','packet','road','joint')]
        binding.update({str(v):sha(v) for v in files.values()})
        alpha=p['road_alpha'].squeeze();road_joint=j['entire_road_weight'];K=j['K']
        assert np.array_equal(alpha,f['baseline_road_alpha'])
        row=dict(raw=raw,domains={},K=dict(pixels=int(K.sum()),
                M11_minus_parent_RGB_max=float(np.abs(p['joint_rgb']-f['baseline_joint_rgb'])[:,K].max())))
        for region in ('A','C','S'):
            expected=d['coverage_'+region].astype(bool)
            stable=expected&(alpha>=.95)
            teacher=f['target_'+region].transpose(1,2,0).astype(float)
            domain=dict(expected_pre_alpha_pixels=int(expected.sum()),legacy_texture_score_pixels=int(f[region].sum()),
                alpha=summary(alpha[expected]),native_road_We=summary(r['eligible_weight'][expected]),
                joint_whole_road_alphaT=summary(road_joint[expected]),joint_eligible_road_alphaT=summary(j['eligible_weight'][expected]),
                expected_overlaps_K_pixels=int((expected&K).sum()),
                coverage_thresholds=[dict(alpha_threshold=t,pixels=int((expected&(alpha>=t)).sum()),
                    below_pixels=int((expected&(alpha<t)).sum())) for t in (.05,.5,.9,.95,.99)],
                low_coverage_components={str(t):gap_components(expected&(alpha<t)) for t in (.05,.9,.95)},
                stable_road_but_joint_whole_road_below05=int((stable&(road_joint<.05)).sum()),
                stable_road_but_joint_eligible_road_below05=int((stable&(j['eligible_weight']<.05)).sum()),
                warning='low joint road alphaT proves road handoff occlusion, not by itself which static Gaussian is a target-car ghost')
            if region in ('C','S'):
                joint_rgb=p['joint_rgb'].transpose(1,2,0).astype(float)
                domain['joint_color_MAE_auxiliary']=float(np.abs(joint_rgb-teacher)[expected].mean())
                # Full legally expected region; no alpha divide or exclusion of occluded road.
                rep=evaluate_detail(joint_rgb,teacher,context_mask=expected,score_mask=expected,
                    resolution=ResolutionSpec(480,320,'camera480_full_joint_expected_'+region,'srgb'),
                    tile_layout=TileLayout(32,32),teacher_role='frozen_pseudo_teacher',color_space='srgb',
                    input_conditioning=dict(road_alpha_not_used_to_hide_ghosts=True,frozen_expected_mask=True))
                write(out/f'{raw:03d}_{region}_joint_detail.json',rep)
                domain['joint_detail']=dict(scales=[dict(sigma=x['sigma_pixels'],core=x['core'],
                    full_score=x['full_score_pixels'],boundary_pixels=x['boundary_pixels']) for x in rep['scales']],
                    edges=[{k:v for k,v in x.items() if k!='tiles'} for x in rep['edges']['threshold_curves']])
            row['domains'][region]=domain
        rows.append(row)
    result=dict(schema='road_quality_secondary_480_diagnostics.v1',rows=rows,bindings=binding,
        seconds=time.monotonic()-started,optimizer_updates=0,native_calls=0,GPU_operations=0,
        limitations=['pre-alpha480 masks are existing conservatively pooled legal domains, not the new native1920 masks',
                    'static versus dynamic attribution requires new highres same-pass source decomposition; old readonly basis mixes branches',
                    'joint appearance under a removed car has frozen pseudo teacher, not real hidden-road GT'])
    write(out/'results.json',result);print(json.dumps(dict(output=str(out),seconds=result['seconds'])))

if __name__=='__main__':main()
